"""Clustering methods and scoring for the HPA tissue-microarray cells.

Library for Extended Data Fig. 2i-k / Supplementary Note 2: GPU k-means, Ward
(on a subsample, extended by nearest centroid), diagonal-covariance Gaussian
mixture and Leiden clustering of the Harmony-corrected 3,072-d features, plus the
metrics each labelling is scored on (see Supplementary Methods, "Clustering"):

  silhouette (25,000-cell subsample), Davies-Bouldin, Calinski-Harabasz in the
  clustered space; a second Calinski-Harabasz index on 22 withheld morphology and
  intensity measurements z-scored within core (``ext_ch``); resample stability
  (ARI against two 80% subsamples); spatial coherence of each cell's ten nearest
  tissue neighbours beyond the core composition; and batch confounding (the
  cell-weighted largest single-core share per cluster).

Entry points built on it: ``cluster_comparison`` (the four-family comparison),
``kmeans_k13`` (the reported k = 13 partition) and ``reference_labels`` (the
labelling that the manual quality assessment was done on). ``sweep`` here runs
every family over k = 10-30 on one feature matrix.

Inputs: ``config.TISSUE_INTENSITY`` (slide coordinates and withheld measurements),
a metadata table with ``id, slide, core, fov``. No PCA or feature preprocessing is
applied: the two 1,536-d channel blocks leave Harmony on the same scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

from subcell import config
from subcell.runlog import log
from subcell.utils import zscore

KS = list(range(10, 31))
KS_COARSE = [10, 12, 15, 18, 20, 22, 25, 28, 30]
# calibrated: res 0.25 -> k=5, res 4.0 -> k=59, so k=10..30 lives in ~0.4-2.0
RESOLUTIONS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00]
GRAPH_K = 30
SIL_N = 25000
WARD_N = 12000
SPATIAL_K = 10
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Independent measurements the clustering never sees -- used to check that
# clusters differ on something other than the SubCell embedding itself.
MORPH = ["cell_area", "cell_eccentricity", "cell_solidity", "nuc_area",
         "nuc_eccentricity", "nuc_solidity", "nuc_cell_area_ratio"]
INTENS = [f"{c}_{a}_{b}" for c in ("mt", "DAPI", "Vimentin")
          for a, b in (("cell", "mean"), ("cell", "cv"), ("cyto", "mean"),
                       ("nucl", "mean"), ("nucl_cyto", "ratio"))]


# The eight measurements drawn in Extended Data Fig. 2j (column, two-line header). The
# header text is also the key of the per-cluster table and of its source-data columns.
MEASURED = [("DAPI_nucl_mean", "Nucleus\nIntensity"), ("nuc_area", "Nucleus\nArea"),
            ("nuc_solidity", "Nucleus\nSolidity"), ("nuc_eccentricity", "Nucleus\nEccentr."),
            ("Vimentin_cell_mean", "Vimentin\nCell"), ("Vimentin_cell_cv", "Vimentin\nCell CV"),
            ("mt_cell_mean", "Microtub.\nCell"), ("mt_cell_cv", "Microtub.\nCell CV")]


def aligned_intensity(ids) -> pd.DataFrame:
    """Rows of ``config.TISSUE_INTENSITY`` in the order of cell ``ids``."""
    im = pd.read_csv(config.TISSUE_INTENSITY)
    im["id"] = im["id"].str.replace(r"_(Vimentin|mt|DAPI)$", "", regex=True)
    pos = pd.Index(im["id"]).get_indexer(ids)
    assert (pos >= 0).all(), f"{(pos < 0).sum()} cells missing from intensity_metrics"
    return im.iloc[pos].reset_index(drop=True)


def external_panel(im, meta):
    """Morphology + intensity, z-scored WITHIN core so batch cannot drive it."""
    cols = [c for c in MORPH + INTENS if c in im.columns]
    E = zscore.log_heavy_tailed(im[cols].to_numpy())
    return zscore.zscore_within(E, meta["core"].to_numpy(), np.float32), cols


def kmeans_gpu(Xg, k, seed, iters=150, tol=1e-6):
    n = Xg.shape[0]
    g = torch.Generator(device=Xg.device).manual_seed(seed)
    xsq = (Xg * Xg).sum(1)

    first = torch.randint(n, (1,), generator=g, device=Xg.device)
    C = Xg[first].clone()
    d2 = (xsq + xsq[first] - 2 * (Xg @ Xg[first].T).squeeze(1)).clamp_min_(0)
    for _ in range(k - 1):  # k-means++
        nxt = torch.multinomial(d2 / d2.sum().clamp_min(1e-12), 1, generator=g)
        C = torch.cat([C, Xg[nxt]], 0)
        d2 = torch.minimum(d2, (xsq + xsq[nxt] - 2 * (Xg @ Xg[nxt].T).squeeze(1)).clamp_min_(0))

    lab = None
    for _ in range(iters):
        d = xsq[:, None] - 2 * (Xg @ C.T) + (C * C).sum(1)[None, :]
        lab = d.argmin(1)
        cnt = torch.bincount(lab, minlength=k)
        newC = torch.zeros_like(C).index_add_(0, lab, Xg) / cnt.clamp_min(1).unsqueeze(1)
        empty = (cnt == 0).nonzero().squeeze(1)
        if empty.numel():  # respawn dead centroids on the worst-fit cells
            newC[empty] = Xg[d.gather(1, lab[:, None]).squeeze(1).topk(empty.numel()).indices]
        shift = (newC - C).norm()
        C = newC
        if shift < tol:
            break
    return lab.cpu().numpy(), C


def gmm_diag_gpu(Xg, k, seed, iters=60):
    n, d = Xg.shape
    lab, mu = kmeans_gpu(Xg, k, seed, iters=40)
    lab = torch.as_tensor(lab.astype(np.int64), device=Xg.device)
    var = torch.full((k, d), float(Xg.var()), device=Xg.device)
    pi = torch.bincount(lab, minlength=k).float().clamp_min(1) / n
    X2 = Xg * Xg
    floor = float(Xg.var()) * 1e-3

    lp = None
    for _ in range(iters):
        iv = 1.0 / var
        quad = X2 @ iv.T - 2 * (Xg @ (mu * iv).T) + (mu * mu * iv).sum(1)[None, :]
        lp = -0.5 * (quad + torch.log(var).sum(1)[None, :]) + torch.log(pi)[None, :]
        R = torch.softmax(lp, 1)
        Nk = R.sum(0).clamp_min(1e-6)
        mu = (R.T @ Xg) / Nk[:, None]
        var = ((R.T @ X2) / Nk[:, None] - mu * mu).clamp_min(floor)
        pi = Nk / n
    return lp.argmax(1).cpu().numpy(), mu


def ward_labels(Xg, ks, seed):
    """Ward on a subsample, cut at every k, extended by nearest centroid."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(Xg.shape[0], min(WARD_N, Xg.shape[0]), replace=False)
    S = Xg[torch.as_tensor(sub, device=Xg.device)]
    D = torch.cdist(S, S)
    iu = torch.triu_indices(len(sub), len(sub), offset=1, device=S.device)
    cond = D[iu[0], iu[1]].double().cpu().numpy()
    del D, iu
    torch.cuda.empty_cache()
    Z = sch.linkage(cond, method="ward")
    del cond

    out = {}
    for k in ks:
        sl = sch.fcluster(Z, k, criterion="maxclust") - 1
        cent = torch.stack([S[torch.as_tensor(sl == j, device=S.device)].mean(0)
                            for j in range(sl.max() + 1)])
        out[k] = assign_nearest(Xg, cent)
    return out


def assign_nearest(Xg, C):
    d = (Xg * Xg).sum(1)[:, None] - 2 * (Xg @ C.T) + (C * C).sum(1)[None, :]
    return d.argmin(1).cpu().numpy()


def knn_graph(Xg, k=GRAPH_K, chunk=2048):
    Z = torch.nn.functional.normalize(Xg, dim=1)
    n = Z.shape[0]
    idx = np.empty((n, k), np.int64)
    sim = np.empty((n, k), np.float32)
    for i in range(0, n, chunk):
        s = Z[i:i + chunk] @ Z.T
        rows = torch.arange(s.shape[0], device=Z.device)
        s[rows, torch.arange(i, i + s.shape[0], device=Z.device)] = -2  # drop self
        v, j = s.topk(k, dim=1)
        idx[i:i + chunk] = j.cpu().numpy()
        sim[i:i + chunk] = v.cpu().numpy()

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    lo, hi = np.minimum(src, idx.ravel()), np.maximum(src, idx.ravel())
    _, first = np.unique(lo * n + hi, return_index=True)  # symmetrise as union
    g = ig.Graph(n=n, edges=list(zip(lo[first].tolist(), hi[first].tolist())))
    g.es["weight"] = np.clip(sim.ravel()[first], 0, None).tolist()
    log(f"  knn graph: {g.vcount():,} nodes {g.ecount():,} edges")
    return g


def leiden(g, res, seed):
    p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                 weights=g.es["weight"], resolution_parameter=res,
                                 n_iterations=2, seed=seed)
    return np.asarray(p.membership, np.int64)


def relabel(lab):
    """Renumber clusters by descending size, so cluster 0 is the largest."""
    lab = np.asarray(lab, np.int64)
    order = np.argsort(-np.bincount(lab))
    remap = np.empty(order.max() + 1, np.int64)
    remap[order] = np.arange(len(order))
    return remap[lab]


def silhouette_gpu(Xg, lab, n_sub=SIL_N, seed=0):
    rng = np.random.default_rng(seed)
    sub = rng.choice(Xg.shape[0], min(n_sub, Xg.shape[0]), replace=False)
    S = Xg[torch.as_tensor(sub, device=Xg.device)]
    L = torch.as_tensor(lab[sub], device=Xg.device)
    k = int(L.max()) + 1
    cnt = torch.bincount(L, minlength=k).float()
    if (cnt < 2).any():
        return float("nan")

    ssq = (S * S).sum(1)
    tot = torch.zeros(len(sub), k, device=Xg.device)
    for i in range(0, len(sub), 2048):
        d = (ssq[i:i + 2048, None] - 2 * (S[i:i + 2048] @ S.T) + ssq[None, :]).clamp_min_(0).sqrt_()
        tot[i:i + 2048] = torch.zeros(d.shape[0], k, device=Xg.device).index_add_(1, L, d)

    own = torch.nn.functional.one_hot(L, k).bool()
    a = tot[own] / (cnt[L] - 1).clamp_min(1)
    other = tot / cnt[None, :]
    other[own] = float("inf")
    b = other.min(1).values
    return float(((b - a) / torch.maximum(a, b).clamp_min(1e-12)).mean())


def ch_db_gpu(Xg, lab):
    """Calinski-Harabasz (higher better), Davies-Bouldin (lower better)."""
    L = torch.as_tensor(lab, device=Xg.device)
    k, n = int(L.max()) + 1, Xg.shape[0]
    cnt = torch.bincount(L, minlength=k).float().clamp_min(1)
    cent = torch.zeros(k, Xg.shape[1], device=Xg.device).index_add_(0, L, Xg) / cnt[:, None]
    gm = Xg.mean(0)
    bgss = (cnt[:, None] * (cent - gm) ** 2).sum()
    r2 = ((Xg - cent[L]) ** 2).sum(1)
    wg = torch.zeros(k, device=Xg.device).index_add_(0, L, r2)
    ch = float((bgss / (k - 1)) / (wg.sum() / (n - k) + 1e-12))
    s = (wg / cnt).sqrt()
    cd = torch.cdist(cent, cent) + torch.eye(k, device=Xg.device)
    ratio = (s[:, None] + s[None, :]) / cd
    ratio.fill_diagonal_(-1e9)
    return ch, float(ratio.max(1).values.mean())


def ch_np(E, lab):
    k = lab.max() + 1
    cent = np.stack([E[lab == j].mean(0) for j in range(k)])
    cnt = np.bincount(lab, minlength=k)
    bgss = (cnt[:, None] * (cent - E.mean(0)) ** 2).sum()
    wg = ((E - cent[lab]) ** 2).sum()
    return float((bgss / (k - 1)) / (wg / (len(E) - k) + 1e-12))


def spatial_neighbors(meta, k=SPATIAL_K):
    """Indices of each cell's k nearest neighbours in tissue space, within core."""
    xy = meta[["slide_cell_x", "slide_cell_y"]].to_numpy()
    nbr = np.full((len(meta), k), -1, np.int64)
    for _, idx in meta.groupby("core").indices.items():
        if len(idx) <= k:
            continue
        _, j = cKDTree(xy[idx]).query(xy[idx], k=k + 1)
        nbr[idx] = idx[j[:, 1:]]  # drop self
    return nbr


def spatial_coherence(lab, nbr, core_codes, n_core):
    """Do neighbouring cells share a label more than the core's mix predicts?

    A real cell type is spatially organised (islets, ducts, portal tracts); a
    cluster that is just a continuum slice is spatially random. Baseline is the
    per-core composition, so a core dominated by one cluster gets no credit.
    """
    ok = nbr[:, 0] >= 0
    obs = float((lab[nbr[ok]] == lab[ok][:, None]).mean())
    k = lab.max() + 1
    M = np.zeros((n_core, k))
    np.add.at(M, (core_codes, lab), 1)
    p = M / M.sum(1, keepdims=True).clip(1)
    exp = float(p[core_codes[ok], lab[ok]].mean())
    return obs - exp, obs / max(exp, 1e-9)


def core_confound(lab, core_codes, n_core):
    """Cell-weighted mean of each cluster's largest single-core share."""
    k = lab.max() + 1
    M = np.zeros((k, n_core))
    np.add.at(M, (lab, core_codes), 1)
    frac = M / M.sum(1, keepdims=True).clip(1)
    w = M.sum(1) / M.sum()
    ent = -(frac * np.log(frac + 1e-12)).sum(1) / np.log(n_core)
    return float((w * frac.max(1)).sum()), float((w * ent).sum())


@dataclass
class Cells:
    """Everything about the cells that does not depend on the feature matrix."""
    meta: pd.DataFrame        # metadata plus slide/FOV coordinates and ``row``
    ext: np.ndarray           # withheld measurements, z-scored within core
    ext_cols: list
    nbr: np.ndarray           # tissue-space neighbours within core
    core_codes: np.ndarray
    n_core: int


def load_cells(meta: pd.DataFrame) -> Cells:
    """Attach coordinates and the withheld measurement panel to ``meta``."""
    meta = meta.copy()
    im = aligned_intensity(meta["id"])
    for c in ("slide_cell_x", "slide_cell_y", "fov_cell_x", "fov_cell_y"):
        meta[c] = im[c].to_numpy()
    meta["row"] = np.arange(len(meta))
    ext, ext_cols = external_panel(im, meta)
    core_codes = pd.Categorical(meta["core"]).codes.astype(np.int64)
    nbr = spatial_neighbors(meta)
    log(f"{len(meta):,} cells · external panel {ext.shape} ({len(ext_cols)} "
        f"measurements, within-core z-scored) · spatial graph k={SPATIAL_K}")
    return Cells(meta, ext, ext_cols, nbr, core_codes, int(core_codes.max()) + 1)


def sweep_methods(Xg, cells: Cells, ks, ks_gmm, resolutions, extra=None):
    """Score k-means, Ward, GMM and Leiden labellings of the features ``Xg``.

    ``ks`` applies to k-means and Ward, ``ks_gmm`` to the mixture; Leiden is run at
    each resolution and kept when it yields 10-30 clusters. ``extra`` adds constant
    columns (e.g. the harmonization name) to every row. Returns ``(rows, labels)``
    with one row of metrics and one relabelled partition per labelling.
    """
    rng = np.random.default_rng(SEED)
    n = Xg.shape[0]
    subs = [rng.choice(n, int(0.8 * n), replace=False) for _ in range(2)]
    subg = [Xg[torch.as_tensor(s, device=DEV)] for s in subs]
    rows, labels = [], {}

    def score(family, method, lab, stab_fn):
        """stab_fn(i) -> labels for subs[i]; ARI against the full labelling."""
        lab = relabel(lab)
        k = lab.max() + 1
        sil = silhouette_gpu(Xg, lab)
        ch, db = ch_db_gpu(Xg, lab)
        ech = ch_np(cells.ext, lab)
        mx, ent = core_confound(lab, cells.core_codes, cells.n_core)
        sx, sr = spatial_coherence(lab, cells.nbr, cells.core_codes, cells.n_core)
        stab = float(np.mean([adjusted_rand_score(lab[subs[i]], relabel(stab_fn(i)))
                              for i in range(len(subs))]))
        mn = float(np.bincount(lab).min() / len(lab))
        name = f"{method}_k{k}"
        labels[name] = lab
        rows.append({**(extra or {}), "family": family, "method": method, "k": k,
                     "silhouette": sil, "ch": ch, "db": db, "ext_ch": ech,
                     "stability": stab, "spatial_excess": sx, "spatial_ratio": sr,
                     "core_maxfrac": mx, "core_entropy": ent, "min_frac": mn})
        log(f"  {name:18s} sil {sil:+.4f}  DB {db:.3f}  extCH {ech:7.2f}  stab {stab:.3f}  "
            f"spatial {sx:+.4f} ({sr:.2f}x)  coreMax {mx:.3f}  minClust {mn:.4f}")

    log("k-means")
    for k in ks:
        lab, _ = kmeans_gpu(Xg, k, SEED)
        score("kmeans", "kmeans", lab, lambda i, k=k: kmeans_gpu(subg[i], k, SEED + 1)[0])

    log(f"ward ({WARD_N} subsample + nearest-centroid extension)")
    ward_full = ward_labels(Xg, ks, SEED)
    ward_sub = [ward_labels(subg[i], ks, SEED + 1 + i) for i in range(len(subs))]
    for k in ks:
        score("ward", "ward", ward_full[k], lambda i, k=k: ward_sub[i][k])

    log("gmm (diagonal covariance)")
    for k in ks_gmm:
        lab, _ = gmm_diag_gpu(Xg, k, SEED)
        score("gmm", "gmm", lab, lambda i, k=k: gmm_diag_gpu(subg[i], k, SEED + 1)[0])

    log("leiden (graph)")
    g = knn_graph(Xg)
    gsub = {}
    for res in resolutions:
        lab = leiden(g, res, SEED)
        k = lab.max() + 1
        if not (10 <= k <= 30):
            log(f"  res {res}: k={k}, outside 10-30, skipped")
            continue

        def stab_fn(i, res=res):
            if i not in gsub:
                gsub[i] = knn_graph(subg[i])
            return leiden(gsub[i], res, SEED + 1)

        score("leiden", f"leiden(r{res})", lab, stab_fn)
    return rows, labels


def sweep(features_path, cells: Cells, out_dir, resolutions=RESOLUTIONS):
    """All four families over k = 10-30 on one feature matrix.

    Writes ``cluster_scores.csv`` (one row of metrics per labelling) and
    ``all_labels.npz`` (every labelling, keyed ``<method>_k<k>``) to ``out_dir``.
    """
    out_dir = config.ensure(out_dir)
    X = np.load(features_path).astype(np.float32)
    assert len(X) == len(cells.meta), f"{len(X)} rows vs {len(cells.meta)} cells"
    Xg = torch.as_tensor(X, device=DEV)
    del X
    rows, labels = sweep_methods(Xg, cells, KS, KS_COARSE, resolutions)
    pd.DataFrame(rows).to_csv(out_dir / "cluster_scores.csv", index=False)
    np.savez_compressed(out_dir / "all_labels.npz",
                        **{k: v.astype(np.int16) for k, v in labels.items()})
    log(f"scored {len(rows)} labellings -> {out_dir}")


def finalize(meta, lab, cells: Cells, outdir):
    """Per-cluster measurement profiles and cluster-by-core composition tables.

    Writes ``cluster_profiles.csv`` (mean withheld measurements, z-scored within
    core, per cluster) and the ``cluster_by_core_*`` / ``core_by_cluster_pct`` /
    ``cluster_core_summary`` tables to ``outdir``.
    """
    outdir = config.ensure(outdir)
    # Labels need not be contiguous: transferred identities leave gaps. Index by
    # the id so a cluster keeps its name, and skip ids that are absent.
    k = lab.max() + 1
    present = np.flatnonzero(np.bincount(lab, minlength=k) > 0)
    prof = np.full((k, cells.ext.shape[1]), np.nan)
    for j in present:
        prof[j] = cells.ext[lab == j].mean(0)
    pd.DataFrame(prof[present], columns=cells.ext_cols,
                 index=[f"c{j}" for j in present]).to_csv(outdir / "cluster_profiles.csv")
    core_stats(meta, lab, outdir)


def core_stats(meta, lab, outdir):
    """Cluster x core composition, both directions plus enrichment.

    Row % answers "where are this cluster's cells", column % answers "what is
    this core made of". They differ a lot because cores range 1,517-13,250
    cells, so a cluster can be 15% "in" a big core while barely featuring in it.
    """
    ct = pd.crosstab(pd.Series(lab, name="cluster"), meta["core"].to_numpy())
    ct.columns.name = "core"
    row = 100 * ct.div(ct.sum(1), axis=0)             # % of cluster in each core
    col = 100 * ct.div(ct.sum(0), axis=1)             # % of core that is cluster
    enr = row.div(100 * ct.sum(0) / ct.values.sum(), axis=1)   # obs / expected

    ct.to_csv(outdir / "cluster_by_core_counts.csv")
    row.round(2).to_csv(outdir / "cluster_by_core_pct.csv")
    col.round(2).to_csv(outdir / "core_by_cluster_pct.csv")
    enr.round(3).to_csv(outdir / "cluster_by_core_enrichment.csv")

    p = (row / 100).to_numpy()
    even = -(p * np.log(p + 1e-12)).sum(1) / np.log(ct.shape[1])
    top3 = [", ".join(f"{c} {row.loc[i, c]:.1f}%" for c in row.loc[i].nlargest(3).index)
            for i in row.index]
    summary = pd.DataFrame({
        "n": ct.sum(1), "top_core": row.idxmax(1), "top_pct": row.max(1).round(1),
        "top_enrich": enr.max(1).round(2), "cores_ge5pct": (row >= 5).sum(1),
        "cores_ge1pct": (row >= 1).sum(1), "evenness": even.round(3), "top3_cores": top3})
    summary.to_csv(outdir / "cluster_core_summary.csv")
    log("cluster x core summary (evenness 1.0 = spread equally over all "
        f"{ct.shape[1]} cores, 0 = one core)\n" + summary.to_string())
