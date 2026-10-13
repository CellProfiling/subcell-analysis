"""Harmony batch correction of the HPA tissue SubCell features (Extended Data Fig. 2i).

The microtubule and vimentin SubCell embeddings of each cell (1,536-d each) are
concatenated into one 3,072-d matrix and corrected with Harmony (harmony-pytorch,
no PCA, no scaling) over six candidate batch keys that span the design hierarchy:

    none  ·  cohort (2)  ·  acquisition block (9)  ·  donor (19)
          ·  donor + acquisition block (22)  ·  core (29)

Core correction is the one carried forward (``kmeans_k13``). Cores are nested in
both acquisition blocks and donors, so block + core == core == donor + core; donor
+ block is the only informative composite because three donors have cores in two
blocks. In the metadata columns and file names, ``patient`` is the donor and ``slide``
is the acquisition block (the filename prefix G1..H7), while ``cohort`` is the G/H
letter of the annotation's own "Slide" column.

Stages (``python -m subcell.tissue.harmonization <stage> [--fresh] [--pca-dim N]``):

    validate  join the core annotation to the cells and check the join
    harm      Harmony per key, slide-effect test, kNN purity of every variant
    umap      one UMAP per variant, for ``figures.ed2i_tissue_umaps``
    all       harm, then umap

The UMAP neighbour search can run on the top ``--pca-dim`` principal components
(0 = the raw 3,072 dims); Harmony itself always runs on the full 3,072 dims. The
reported embeddings used 256 (``slurm/tissue_umap.sbatch``).

Join. The annotation's acquisition prefix does not agree with the filenames (it
lumps R27-R36 into one block "HX"; the data splits it into two), so cells join on
cohort letter + region number, which is unique. Cores in the features but absent
from the annotation, or the reverse, are listed by ``validate``.

Inputs   ``config.TISSUE_CORE_META``; ``config.TISSUE_FEATURES``/subcell_result_{mt,
         Vimentin}.csv (parsed once into a cached ``features.npy`` with its
         ``metadata.csv`` in ``config.TISSUE_FEATURE_CACHE``)
Outputs  ``config.TISSUE_HARM``: metadata.csv, core_annotation_resolved.csv,
         harmonized_<key>.npy, slide_effect_pairs.csv, metrics.csv, umaps_patient.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from subcell import config
from subcell.runlog import log

CHANNELS = ["mt", "Vimentin"]
FEATURE_CACHE = config.TISSUE_FEATURE_CACHE
OUT_DIR = config.TISSUE_HARM
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# batch key name -> metadata columns passed to Harmony (the names are on-disk names)
KEYS = {"cohort": ["cohort"], "slide": ["slide"], "patient": ["patient"],
        "patient+slide": ["patient", "slide"], "core": ["core"]}
UMAP_VARIANTS = ["none", *KEYS]


# ------------------------------------------------------------------- inputs


def load_features():
    """Concatenated mt + Vimentin features and per-cell metadata, cached."""
    fcache, mcache = FEATURE_CACHE / "features.npy", FEATURE_CACHE / "metadata.csv"
    if fcache.exists() and mcache.exists():
        return np.load(fcache), pd.read_csv(mcache)

    blocks, ids = [], None
    for ch in CHANNELS:
        path = config.TISSUE_FEATURES / f"subcell_result_{ch}.csv"
        cols = pd.read_csv(path, nrows=0).columns
        feat_cols = [c for c in cols if c.startswith("feat")]
        log(f"reading {ch}: {len(feat_cols)} feature columns")
        df = pd.read_csv(path, usecols=["id"] + feat_cols,
                         dtype={c: np.float32 for c in feat_cols})
        # ids look like G1_R1_s00_cell1_mt -> drop the channel suffix
        cell_id = df["id"].str.rsplit("_", n=1).str[0]
        feats = df[feat_cols].to_numpy(np.float32)
        if ids is None:
            ids = cell_id
        else:
            pos = pd.Index(ids).get_indexer(cell_id)
            assert len(cell_id) == len(ids) and (pos >= 0).all(), (
                f"{ch} and {CHANNELS[0]} do not contain the same cells")
            # argsort(pos) below only reorders correctly if each id occurs once
            assert len(set(pos)) == len(pos), f"{ch} has duplicated cell ids"
            feats = feats[np.argsort(pos)]
        blocks.append(feats)

    X = np.concatenate(blocks, axis=1)
    parts = ids.str.split("_", expand=True)
    meta = pd.DataFrame({"id": ids})
    meta["slide"] = parts[0]
    meta["core"] = parts[0] + "_" + parts[1]
    meta["fov"] = parts[0] + "_" + parts[1] + "_" + parts[2]

    config.ensure(FEATURE_CACHE)
    np.save(fcache, X)
    meta.to_csv(mcache, index=False)
    log(f"cached concatenated features -> {fcache}")
    return X, meta


def load_annotation():
    """The verified core table (one row per TMA core): tissue, donor, cancer status.

    The join key is cohort letter + region number, unique because region numbers
    repeat only across cohorts (G_R1 vs H_R1).
    """
    a = pd.read_csv(config.TISSUE_CORE_META, sep="\t").rename(columns={
        "Core": "annot_core", "Slide": "cohort", "Tissue": "tissue",
        "Patient_tentative": "patient", "Cancer": "cancer",
        "Imaged": "imaged", "Data_Analyzed": "analyzed"})
    a["region"] = a["annot_core"].str.split("_").str[1]
    a["core_id"] = a["cohort"] + "_" + a["region"]
    a["annot_block"] = a["annot_core"].str.split("_").str[0]
    for c in ("imaged", "analyzed"):
        a[c] = a[c].fillna("N").eq("Y")
    assert a["core_id"].is_unique, "cohort+region is not unique in the annotation"
    return a


def join_annotation(meta, a):
    """Attach donor / tissue / cancer to every cell via cohort+region."""
    cores = pd.DataFrame({"core": sorted(meta["core"].unique())})
    cores["core_id"] = cores["core"].str[0] + "_" + cores["core"].str.split("_").str[1]
    assert cores["core_id"].is_unique, "dropping the acquisition prefix merged cores"

    keep = ["core_id", "annot_core", "annot_block", "cohort", "patient", "tissue",
            "cancer", "imaged", "analyzed"]
    cores = cores.merge(a[keep], on="core_id", how="left")
    unjoined = cores[cores.patient.isna()]
    assert unjoined.empty, f"cores with no annotation row: {list(unjoined.core)}"

    out = meta.drop(columns=["organ", "condition"], errors="ignore").merge(
        cores, on="core", how="left")
    assert out["patient"].notna().all() and len(out) == len(meta)
    return out, cores


def validate(meta, cores, a):
    """Log how the annotation and the segmented data line up."""
    p = log
    p(f"annotation rows {len(a)}   data cores {len(cores)}   all joined on cohort+region")
    p(f"annotation says Imaged={a.imaged.sum()}  Data_Analyzed={a.analyzed.sum()}"
      f"  ->  features contain {len(cores)} cores")
    exp = set(a.loc[a.analyzed, "core_id"])
    got = set(cores["core_id"])
    if exp - got:
        p(f"  marked Data_Analyzed=Y but absent from the features ({len(exp - got)}): "
          + ", ".join(f"{c} ({a.set_index('core_id').loc[c, 'annot_core']})"
                      for c in sorted(exp - got)))
    if got - exp:
        p(f"  in the features but not marked Data_Analyzed=Y ({len(got - exp)}): "
          f"{sorted(got - exp)}")
    p(f"  imaged but not analyzed ({int((a.imaged & ~a.analyzed).sum())}): "
      + ", ".join(a.loc[a.imaged & ~a.analyzed, "annot_core"]))
    p(f"  never imaged ({int((~a.imaged).sum())}): " + ", ".join(a.loc[~a.imaged, "annot_core"]))

    dis = cores[cores.annot_block != cores.core.str.split("_").str[0]]
    p(f"acquisition prefix differs between annotation and filenames ({len(dis)} cores); "
      "the filenames are used:")
    for _, r in dis.iterrows():
        p(f"  {r.core_id:6s}  annotation {r.annot_core:8s}  data {r.core:8s}")

    p("cohort letter vs cancer (must be 1:1):\n" + pd.crosstab(a["cohort"], a["cancer"]).to_string())
    p("resolved design, one row per core:\n"
      + cores[["core", "core_id", "patient", "tissue", "cancer"]].to_string(index=False))

    t = cores.groupby(["tissue", "cancer"]).agg(
        cores=("core", "size"), donors=("patient", "nunique"),
        which=("patient", lambda s: ",".join(sorted(set(s)))))
    p("cores and donors per tissue x cancer:\n" + t.to_string()
      + f"\ntotal distinct donors: {cores.patient.nunique()}")
    g = cores.groupby("patient").agg(
        n=("core", "size"), blocks=("core", lambda s: sorted({x.split('_')[0] for x in s})),
        cores=("core_id", lambda s: sorted(s)))
    p("donors with more than one core:\n" + g[g.n > 1].to_string())

    p("batch-key cardinality:\n" + "\n".join(
        f"  {'+'.join(k):16s} {meta[k].drop_duplicates().shape[0]:5d}"
        for k in (["slide"], ["patient"], ["patient", "slide"], ["core"], ["fov"])))
    sp = meta.groupby("patient")["slide"].nunique()
    p(f"cores spanning more than one donor: "
      f"{int((meta.groupby('core')['patient'].nunique() > 1).sum())}   (0 => donor+core == core)\n"
      f"donors spanning more than one acquisition block: {int((sp > 1).sum())} "
      f"{sorted(sp[sp > 1].index)}   (> 0 => donor+block is informative)")


# ----------------------------------------------------------- measurements


def pca_reduce(X, dim):
    """Top-``dim`` PCs by randomized SVD on the GPU."""
    t = torch.as_tensor(X, device=DEV)
    t = t - t.mean(0, keepdim=True)
    U, S, _ = torch.pca_lowrank(t, q=dim + 10, niter=4)
    out = (U[:, :dim] * S[:dim]).cpu().numpy().astype(np.float32)
    var = float((S[:dim] ** 2).sum() / (t * t).sum())
    del t, U, S
    torch.cuda.empty_cache()
    log(f"    PCA {X.shape[1]} -> {dim} dims, {100 * var:.1f}% of variance")
    return out


def knn_purity(X, labels, k=50, n_sub=25000, seed=0):
    """Mean fraction of a cell's k neighbours sharing its label, plus the value
    expected under perfect mixing. Computed on a random subsample of ``n_sub``
    cells (full 3,072-d distances over all cells are not worth the time)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), min(n_sub, len(labels)), replace=False)
    t = torch.as_tensor(X[idx], device=DEV, dtype=torch.float32)
    codes = torch.as_tensor(pd.factorize(labels[idx])[0], device=DEV, dtype=torch.long)
    tot = 0.0
    for i in range(0, len(idx), 2048):
        d = torch.cdist(t[i:i + 2048], t)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        nn = d.topk(k, largest=False).indices
        tot += (codes[nn] == codes[i:i + d.shape[0], None]).float().mean(1).sum().item()
    del t
    torch.cuda.empty_cache()
    freq = pd.Series(labels[idx]).value_counts(normalize=True).to_numpy()
    return tot / len(idx), float((freq ** 2).sum())


def fov_means(X, codes, n):
    """Per-FOV mean of the cell matrix; reduceat on a sorted copy is ~20x faster
    than a pandas groupby over 3,072 columns."""
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n))
    return (np.add.reduceat(X[order].astype(np.float64), starts, axis=0)
            / np.bincount(codes, minlength=n)[:, None])


def slide_effect(variants, meta):
    """Does the acquisition block add technical variance beyond core?

    For every donor with two or more cores, separate two of the donor's cores with
    a logistic probe on FOV-mean features. Pairs from the same block measure
    core-to-core tissue heterogeneity; pairs from different blocks measure that
    plus the block effect. Equal AUCs mean the block contributes nothing.
    """
    fcodes, funiq = pd.factorize(meta["fov"], sort=True)
    fm = meta.groupby("fov")[["patient", "slide", "core"]].first().loc[funiq]
    pairs = []
    for p, sub in fm.groupby("patient"):
        if p.startswith("?"):
            continue
        cs = sorted(sub["core"].unique())
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                s1 = sub.loc[sub.core == cs[i], "slide"].iloc[0]
                s2 = sub.loc[sub.core == cs[j], "slide"].iloc[0]
                pairs.append((p, cs[i], cs[j], "same slide" if s1 == s2 else
                              f"across slide ({s1}/{s2})"))
    rows = []
    for name, X in variants.items():
        fX = fov_means(X, fcodes, len(funiq))
        for p, c1, c2, kind in pairs:
            m = fm["core"].isin([c1, c2]).to_numpy()
            y = (fm.loc[m, "core"] == c2).to_numpy().astype(int)
            if min(y.sum(), (1 - y).sum()) < 3:
                continue
            rng = np.random.default_rng(SEED)
            tr = np.zeros(len(y), bool)
            for c in (0, 1):
                idx = np.where(y == c)[0]
                tr[rng.choice(idx, max(1, len(idx) // 2), replace=False)] = True
            clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
                fX[m][tr], y[tr])
            auc = roc_auc_score(y[~tr], clf.predict_proba(fX[m][~tr])[:, 1])
            rows.append({"variant": name, "patient": p, "core_a": c1, "core_b": c2,
                         "contrast": kind, "auc": auc})
    df = pd.DataFrame(rows)
    df["group"] = np.where(df.contrast == "same slide", "same slide", "across slide")
    return df


# ------------------------------------------------------------------- stages


def harmonize_all(X, meta, fresh):
    """Harmonized matrix per key (cached on disk), plus the uncorrected ``none``."""
    from harmony import harmonize

    variants = {"none": X}
    for name, keys in KEYS.items():
        path = OUT_DIR / f"harmonized_{name}.npy"
        if path.exists() and not fresh:
            log(f"--- {name}: cached ({path})")
            variants[name] = np.load(path)
            continue
        log(f"--- harmonize {name} ({meta[keys].drop_duplicates().shape[0]} batches)")
        got = harmonize(X, meta, batch_key=keys, use_gpu=True,
                        random_state=SEED).astype(np.float32)
        np.save(path, got)
        variants[name] = got
    return variants


def harmonize_stage(X, meta, fresh):
    variants = harmonize_all(X, meta, fresh)

    # donor+block should NOT collapse to donor (three donors span two blocks)
    d = np.abs(variants["patient+slide"] - variants["patient"])
    log(f"donor+block vs donor: max|diff| {d.max():.3e}  mean|diff| {d.mean():.3e}  "
        f"(values average {np.abs(variants['patient']).mean():.3f})")

    se = slide_effect(variants, meta)
    se.to_csv(OUT_DIR / "slide_effect_pairs.csv", index=False)
    log("slide effect, AUC separating two cores of the same donor:\n"
        + se.pivot_table(index="variant", columns="group", values="auc",
                         aggfunc="mean").reindex(list(variants)).round(4).to_string())

    rows = []
    for n, e in variants.items():
        r = {"variant": n}
        for key in ("core", "patient", "slide"):
            p, x = knn_purity(e, meta[key].to_numpy())
            r[f"knn_{key}_purity"], r[f"knn_{key}_expected"] = p, x
        rows.append(r)
    res = pd.DataFrame(rows).set_index("variant")
    res.to_csv(OUT_DIR / "metrics.csv")
    log("kNN purity (fraction of 50 neighbours sharing the label; 'expected' = "
        "perfect mixing):\n" + res.round(4).to_string())


def umap_stage(fresh=False, pca_dim=0):
    """One UMAP per variant (cached in the npz; ``fresh`` refits all)."""
    import umap

    npz = config.TISSUE_UMAPS
    umaps = dict(np.load(npz)) if npz.exists() and not fresh else {}
    for name in UMAP_VARIANTS:
        if name in umaps:
            log(f"umap {name}: cached")
            continue
        fp = (FEATURE_CACHE / "features.npy" if name == "none"
              else OUT_DIR / f"harmonized_{name}.npy")
        if not fp.exists():
            log(f"umap {name}: {fp} missing, skipped")
            continue
        log(f"umap {name}")
        E = np.load(fp)
        if pca_dim:
            E = pca_reduce(E, pca_dim)
        # no random_state: it forces UMAP single-threaded
        umaps[name] = umap.UMAP(n_neighbors=30, min_dist=0.3).fit_transform(E)
        del E
        np.savez(npz, **umaps)
        log(f"    saved {npz}")


def main(stage="all", fresh=False, pca_dim=0):
    """stage: validate | harm | umap | all (see the module docstring)."""
    config.ensure(OUT_DIR)
    if stage != "umap":
        X, base = load_features()
        a = load_annotation()
        meta, cores = join_annotation(base, a)
        validate(meta, cores, a)
        meta.to_csv(config.TISSUE_META, index=False)
        cores.to_csv(OUT_DIR / "core_annotation_resolved.csv", index=False)
        if stage != "validate":
            harmonize_stage(X, meta, fresh)
        del X
    if stage in ("umap", "all"):
        umap_stage(fresh, pca_dim)
    log(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["validate", "harm", "umap", "all"])
    ap.add_argument("--fresh", action="store_true", help="ignore cached matrices/embeddings")
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="PCs for the UMAP neighbour search (0 = all 3,072 dims)")
    a = ap.parse_args()
    main(a.stage, a.fresh, a.pca_dim)
