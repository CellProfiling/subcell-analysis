"""Cross-plate retrieval mean average precision (Extended Data Fig. 3e).

For each model and Harmony regime, scores how well FOV embeddings retrieve FOVs
that share an annotation, from a different plate. Panel e plots the location task
(signal) and the cell_line task (nuisance) from ``retrieval_map.csv``.

Inputs: FOV-mean features from ``subcell.io.features.fov_features``.
Outputs, in ``config.HPA_UMAP_DIR / "retrieval"``: ``retrieval_map.csv`` (one row
per model, regime and task) and ``ap/`` (per-(FOV, label) AP caches and per-label
mAP tables).

Run::

    python -m subcell.feature_analysis.retrieval
    python -m subcell.feature_analysis.retrieval --check-copairs   # validate the GPU AP

Three tasks, each scored with copairs' definition of average precision (Kalinin et
al., Nat. Commun. 2025):

  gene      can a FOV retrieve OTHER FOVs of the same gene, from a different
            plate, against every FOV of a different gene?
  location  can a FOV retrieve OTHER FOVs of the same subcellular compartment,
            from a different plate and a DIFFERENT gene, against FOVs of other
            compartments? Positives share no gene with the anchor, so this is not
            gene retrieval in disguise.
  cell_line reported as a diagnostic: how strongly the embedding still encodes
            which cell line an image came from. A nuisance axis, not a result.

Positives must come from a different plate in every task: same-plate replicates
are the trivial case, and batch-aware retrieval benchmarks hold them out.

GENES ARE MULTI-LABEL. 3.3% of FOVs use an antibody targeting several genes at
once (HPA051757 -> ANKRD36 / ANKRD36B / ANKRD36C), and 172 genes appear in more
than one such group. Treating the raw ``ensembl_ids`` string as an atomic label
would score two FOVs that share a gene as a negative pair, so the label is the
SET of genes: a pair is positive when the sets intersect and negative only when
they are disjoint.

AP is computed on the GPU, reproducing copairs exactly (see
:func:`validate_against_copairs`): candidates are the union of a profile's
positive and negative pairs, ranked by descending cosine similarity with ties
resolved positives-first, then AP = sum(precision@k * rel_k) / n_pos. The GPU path
exists because the negative set is quadratic -- copairs needs ~4 min for 8k
profiles, which does not scale to 85k -- while aggregation and p-values are still
handed back to the library.

``--check-copairs`` runs validate_against_copairs(), which compares the two on a
subset small enough for copairs, instead of the full grid. The gene task matches exactly; the location task can differ in the last digits,
since its many positives per row make near-ties that matmul-versus-per-pair
accumulation order can flip. The pair sets and the AP code itself are copairs'.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from .. import config, naming
from ..io import features, metadata
from ..runlog import log

RETRIEVAL_DIR = config.HPA_UMAP_DIR / "retrieval"

# Extended Data Fig. 3e shows location and cell_line. Gene retrieval stays in the
# tables: with 12,733 labels it is on a different scale and would need its own axis.
TASKS = ["gene", "location", "cell_line"]

MAP_THRESHOLD = 0.05
N_BOOTSTRAP = 2000
# copairs builds a permutation null per distinct (n_pos, n_total) pool shape.
# Those shapes come from the metadata alone -- labels and plates -- which is
# identical across all embeddings, so a shared on-disk cache pays the cost
# once and every later embedding reads it back.
MAP_NULL_SIZE = 200
MAP_WORKERS = min(16, os.cpu_count() or 1)

# "copairs" uses copairs.compute.ap_contiguous verbatim and is exact, but moves
# every pair relevance to the host. "gpu" applies the same formula with a GPU
# reduction; the two are compared by ``--check-copairs`` (validate_against_copairs()).
# ``--ap-backend`` selects the one used for the full grid.
AP_BACKEND = "gpu"
SEED = 0


def label_incidence(sets, dev):
    """FOV x label incidence, so a shared label is a set intersection.

    Returns the matrix and the label vocabulary, so a column index can be
    turned back into a gene or compartment name.
    """
    vocab = {t: i for i, t in enumerate(sorted({t for ss in sets for t in ss}))}
    inc = torch.zeros((len(sets), max(len(vocab), 1)), dtype=torch.float64,
                      device=dev)
    if vocab:
        rows = np.concatenate([[i] * len(ss) for i, ss in enumerate(sets)
                               if ss] or [np.array([], dtype=int)])
        cols = np.array([vocab[t] for ss in sets for t in sorted(ss)], dtype=int)
        if len(rows):
            inc[torch.from_numpy(rows).long().to(dev),
                torch.from_numpy(cols).long().to(dev)] = 1
    return inc, list(vocab)


def expanded_masks(task, p, l, ctx):
    """(positive, negative) masks for a chunk of (profile, label) rows.

    Positives for (profile p, label l) are the FOVs also carrying l, on a
    different plate. Negatives are per-profile: FOVs whose label set is DISJOINT
    from p's full set -- which is why the multi-label structure matters, since an
    FOV sharing any gene with p must never count as a negative. For the location
    task positives must additionally share no gene with p, so it cannot collapse
    into gene retrieval.
    """
    plate, ginc = ctx["plate"], ctx["gene_inc"]
    inc = {"gene": ginc, "location": ctx["loc_inc"],
           "cell_line": ctx["cl_inc"]}[task]
    diff_plate = plate[p][:, None] != plate[None, :]
    shares_gene = (ginc[p] @ ginc.T) > 0

    pos = inc[:, l].T.bool() & diff_plate
    if task == "gene":
        neg = ~shares_gene
    else:
        # positives must share no gene with the anchor, so neither the location
        # nor the cell-line task can collapse into gene retrieval
        pos = pos & ~shares_gene
        neg = ~((inc[p] @ inc.T) > 0)
    return pos, neg


def copairs_average_precision(x, ctx, task, chunk=128):
    """Average precision using copairs' own ranking and AP code.

    The similarity values come from a float64 GPU matmul; everything after that
    is copairs:

      * the rank list is ordered by ``np.lexsort([1 - sim, ix])``, copairs'
        exact call in build_rank_lists -- descending similarity, and since
        positives are concatenated ahead of negatives and lexsort is stable, a
        tie resolves in favour of the positive;
      * AP itself is ``copairs.compute.ap_contiguous``, unmodified.

    copairs materialises every pair before ranking, which needs hundreds of GB at
    this scale. That is avoidable: ap_contiguous already operates on contiguous per-row
    segments, so a row's score does not depend on which other rows are in
    memory, and the work can be streamed in chunks of anchors with identical
    results.

    Scored per (profile, label), matching copairs' multilabel path: an antibody
    targeting three genes yields three APs, one per gene.
    """
    from copairs.compute import ap_contiguous

    inc = {"gene": ctx["gene_inc"], "location": ctx["loc_inc"],
           "cell_line": ctx["cl_inc"]}[task]
    prof_ix, label_ix = inc.nonzero(as_tuple=True)
    n_rows = prof_ix.numel()

    ap = np.full(n_rows, np.nan)
    n_pos_out = np.zeros(n_rows, dtype=np.int64)
    n_tot_out = np.zeros(n_rows, dtype=np.int64)

    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        p, l = prof_ix[s:e], label_ix[s:e]
        pos, neg = expanded_masks(task, p, l, ctx)
        rows = e - s
        self_ix = (torch.arange(rows, device=x.device), p)
        pos[self_ix] = False
        neg[self_ix] = False

        sim = x[p] @ x.T                       # float64
        pr, pc = pos.nonzero(as_tuple=True)
        nr, nc = neg.nonzero(as_tuple=True)
        pos_sims = sim[pr, pc].cpu().numpy()
        neg_sims = sim[nr, nc].cpu().numpy()
        pr = pr.cpu().numpy()
        nr = nr.cpu().numpy()

        # copairs' build_rank_lists, same concatenation order and same sort key
        ix = np.concatenate([pr, nr])
        if ix.size == 0:
            continue
        sim_all = np.concatenate([pos_sims, neg_sims])
        labels = np.concatenate([np.ones(pr.size, dtype=np.uint32),
                                 np.zeros(nr.size, dtype=np.uint32)])
        ix_sort = np.lexsort([1 - sim_all, ix])
        rel_k_list = labels[ix_sort]
        uniq, counts = np.unique(ix, return_counts=True)

        ap_chunk, confs = ap_contiguous(rel_k_list, counts.astype(np.uint32))
        ap[s + uniq] = ap_chunk
        n_pos_out[s + uniq] = confs[:, 0]
        n_tot_out[s + uniq] = confs[:, 1]

    return {"ap": ap, "n_pos": n_pos_out, "n_tot": n_tot_out,
            "profile": prof_ix.cpu().numpy(), "label": label_ix.cpu().numpy()}


def gpu_average_precision(x, ctx, task, chunk=256):
    """Same AP as copairs_average_precision, reduced on the GPU.

    copairs' ap_contiguous needs the full per-pair relevance list on the host:
    billions of labels per embedding-task. That volume makes native copairs
    infeasible for the full grid, so the reduction has to happen where the
    similarities already are.

    The formula is copairs': AP = sum(precision@k * rel_k) / n_pos over the
    pos-union-neg rank list, ties resolved positives-first (two stable sorts
    reproduce `np.lexsort([1 - sim, ix])`). Only the reduction differs -- a
    cumsum instead of np.add.reduceat.

    Measured against copairs in float64 by validate_against_copairs() (``--check-copairs``).
    """
    inc = {"gene": ctx["gene_inc"], "location": ctx["loc_inc"],
           "cell_line": ctx["cl_inc"]}[task]
    prof_ix, label_ix = inc.nonzero(as_tuple=True)
    n_rows, n = prof_ix.numel(), x.shape[0]

    ap = torch.full((n_rows,), float("nan"), device=x.device, dtype=torch.float64)
    n_pos_out = torch.zeros(n_rows, dtype=torch.long, device=x.device)
    n_tot_out = torch.zeros(n_rows, dtype=torch.long, device=x.device)
    ranks = torch.arange(1, n + 1, device=x.device, dtype=torch.float64)

    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        p, l = prof_ix[s:e], label_ix[s:e]
        pos, neg = expanded_masks(task, p, l, ctx)
        self_ix = (torch.arange(e - s, device=x.device), p)
        pos[self_ix] = False
        neg[self_ix] = False
        cand = pos | neg

        sim = (x[p] @ x.T).masked_fill(~cand, float("-inf"))
        o1 = torch.argsort((~pos).to(torch.uint8), dim=1, stable=True)
        o2 = torch.argsort(-sim.gather(1, o1), dim=1, stable=True)
        pos_sorted = pos.gather(1, o1).gather(1, o2).to(torch.float64)

        k = pos_sorted.sum(1)
        prec = pos_sorted.cumsum(1) / ranks[None, :]
        ap[s:e] = torch.where(k > 0, (prec * pos_sorted).sum(1) / k.clamp(min=1),
                              torch.full_like(k, float("nan")))
        n_pos_out[s:e] = pos.sum(1)
        n_tot_out[s:e] = cand.sum(1)

    return {"ap": ap.cpu().numpy(), "n_pos": n_pos_out.cpu().numpy(),
            "n_tot": n_tot_out.cpu().numpy(),
            "profile": prof_ix.cpu().numpy(), "label": label_ix.cpu().numpy()}


def build_context(meta, dev):
    gene_inc, gene_vocab = label_incidence(metadata.label_sets(meta, "ensembl_ids"), dev)
    loc_inc, loc_vocab = label_incidence(metadata.label_sets(meta, "locations"), dev)
    cl_inc, cl_vocab = label_incidence(metadata.label_sets(meta, "atlas_name"), dev)
    return {
        "plate": torch.from_numpy(
            pd.factorize(meta["if_plate_id"], sort=True)[0]).long().to(dev),
        "gene_inc": gene_inc, "gene_vocab": gene_vocab,
        "loc_inc": loc_inc, "loc_vocab": loc_vocab,
        "cl_inc": cl_inc, "cl_vocab": cl_vocab,
    }


def expected_ap_vec(n_pos, n_total):
    """AP expected from a random ranking, copairs' exact finite-sample formula.

        E[AP] = (1/L)[((M-1)/(L-1))(L - H_L) + H_L],   L = M + N

    This is the honest baseline for mAP. The naive alternative -- positives
    divided by pool size -- understates it by about a factor of ten, because AP
    is driven by E[1/rank] = H_L/L rather than 1/L.

    Vectorized: copairs sums the harmonic series in a Python loop per row. One
    cumulative sum gives every harmonic number at once; equality is checked in
    validate_normalize_ap.
    """
    M = np.asarray(n_pos, dtype=np.float64)
    L = np.asarray(n_total, dtype=np.float64)
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, int(L.max()) + 1))])
    H_L = H[L.astype(np.int64)]

    mu0 = (1.0 / np.maximum(L, 1.0)) * (
        ((M - 1.0) / np.maximum(L - 1.0, 1e-12)) * (L - H_L) + H_L)
    mu0 = np.where(L < 2, np.where(M == 1, 1.0, 0.0), mu0)
    mu0 = np.where(M == 0, 0.0, mu0)
    mu0 = np.where((M == L) & (L >= 2), 1.0, mu0)
    return mu0


def fast_normalize_ap(ap, n_pos, n_total):
    """copairs' normalized AP, vectorized.

    Identical formula to copairs.map.normalization.normalize_ap:
        E[AP] = (1/L)[((M-1)/(L-1))(L - H_L) + H_L],  norm = (AP - E)/(1 - E)
    but copairs computes H_L with a Python loop summing 1/k up to L, once per
    row. H_L depends only on L, so a single cumulative sum yields every harmonic
    number at once.

    Equality with copairs is checked by validate_normalize_ap().
    """
    mu0 = expected_ap_vec(n_pos, n_total)
    # copairs guards the denominator with max(1-mu0, eps) -- not (1-mu0)+eps --
    # and clips the result to [-1, 1]; both matter when mu0 approaches 1
    eps = 1e-10
    out = (np.asarray(ap, dtype=np.float64) - mu0) / np.maximum(1.0 - mu0, eps)
    return np.clip(out, -1.0, 1.0)


def validate_normalize_ap(n=500, seed=SEED):
    """Confirm the vectorized normalization equals copairs' row-wise version."""
    from copairs.map.normalization import normalize_ap

    rng = np.random.default_rng(seed)
    L = rng.integers(2, 5000, size=n)
    M = np.array([rng.integers(1, l) for l in L])
    ap = rng.random(n)
    ref = np.asarray(normalize_ap(ap, M, L - M), dtype=np.float64)
    return float(np.abs(ref - fast_normalize_ap(ap, M, L)).max())


def retrieval_map(key, feats, meta, task, ap_backend=AP_BACKEND):
    """Per-(profile,label) AP (cached) plus copairs' mAP with permutation p-values.

    mAP is a mean over LABELS -- genes for the gene task, compartments for the
    location task -- so a gene with many FOVs counts once rather than many times.
    """
    from copairs.map import mean_average_precision

    config.ensure(RETRIEVAL_DIR / "ap")
    cache = RETRIEVAL_DIR / "ap" / f"{key}__{task}__copairs64.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            ap, n_pos, n_tot, names = z["ap"], z["n_pos"], z["n_tot"], z["names"]
        log(f"    {task}: cached AP ({len(ap):,} rows)")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # float64: the ranking is discontinuous, so similarity precision is the
        # one place where a reimplementation can still move an AP value
        x = torch.nn.functional.normalize(
            torch.from_numpy(feats).double().to(dev), dim=1)
        ctx = build_context(meta, dev)
        t0 = time.perf_counter()
        ap_fn = (copairs_average_precision if ap_backend == "copairs"
                 else gpu_average_precision)
        r = ap_fn(x, ctx, task)
        vocab = {"gene": ctx["gene_vocab"], "location": ctx["loc_vocab"],
                 "cell_line": ctx["cl_vocab"]}[task]
        names = np.array([vocab[i] for i in r["label"]])
        ap, n_pos, n_tot = r["ap"], r["n_pos"], r["n_tot"]
        del x, ctx
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        np.savez(cache, ap=ap, n_pos=n_pos, n_tot=n_tot, names=names)
        log(f"    {task}: AP for {len(ap):,} (FOV,label) rows on GPU in "
            f"{time.perf_counter() - t0:.0f}s")

    scored = np.isfinite(ap)
    label_col = {"gene": "gene", "location": "compartment",
                 "cell_line": "cell_line"}[task]
    # copairs' own scale correction: AP minus its analytic expectation for that
    # pool composition, so rows with different pos/neg counts are comparable.
    # mean_average_precision requires this column.
    tbl = pd.DataFrame({
        "average_precision": ap, "n_pos_pairs": n_pos, "n_total_pairs": n_tot,
        "normalized_average_precision": fast_normalize_ap(
            np.nan_to_num(ap), n_pos, n_tot),
        "expected_ap": expected_ap_vec(n_pos, n_tot),
        label_col: names,
    })[scored]
    t0 = time.perf_counter()
    agg = mean_average_precision(tbl, sameby=[label_col],
                                 null_size=MAP_NULL_SIZE,
                                 threshold=MAP_THRESHOLD, seed=SEED,
                                 progress_bar=False, max_workers=MAP_WORKERS,
                                 cache_dir=str(config.COPAIRS_CACHE))
    log(f"      mAP + p-values in {time.perf_counter() - t0:.0f}s")
    sig = next((c for c in agg.columns if "below_" in c), None)
    # CI by resampling LABELS, the unit mAP averages over: the several FOVs of
    # one gene share an antibody and a handful of plates, so they are not
    # independent and resampling rows would understate the interval.
    vals = agg["mean_average_precision"].values
    rng = np.random.default_rng(SEED)
    draws = np.nanmean(vals[rng.integers(0, len(vals), (N_BOOTSTRAP, len(vals)))],
                       axis=1)
    return {
        "key": key, "task": task,
        "mAP": float(agg["mean_average_precision"].mean()),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        # random-ranking baseline, aggregated exactly like mAP: mean within a
        # label, then mean over labels
        "chance_mAP": float(tbl.groupby(label_col)["expected_ap"].mean().mean()),
        "AP_row": float(ap[scored].mean()),
        "AP_row_normalized": float(
            tbl["normalized_average_precision"].mean()),
        "n_rows_scored": int(scored.sum()),
        "n_labels": int(len(agg)),
        "frac_significant": float(agg[sig].mean()) if sig else np.nan,
        "median_pos_pairs": float(np.median(n_pos[scored])),
        "median_total_pairs": float(np.median(n_tot[scored])),
    }, agg


def validate_against_copairs(feats, meta, task, n_sample=6000, seed=SEED,
                             ap_backend=AP_BACKEND):
    """Check the GPU path reproduces the library, on a subset copairs can handle.

    copairs is the reference; the fast path is never trusted on faith. Both tasks
    use copairs' multilabel entry point, so the per-(profile,label) rule is the
    library's, and rows are matched on (FOV, label).

    One restriction for the location task: copairs takes a single
    `multilabel_col`, and `locations` needs that slot, so the different-gene
    constraint has to be written against the raw `ensembl_ids` string. The
    comparison therefore runs on single-gene FOVs (96.7% of the data), where
    raw-string inequality and gene-set disjointness coincide.
    """
    from copairs.map.multilabel import average_precision as ml_ap

    rng = np.random.default_rng(seed)
    pool = np.arange(len(feats))
    if task == "location":
        pool = pool[~meta["ensembl_ids"].astype(str).str.contains(",").values]
    idx = np.sort(rng.choice(pool, min(n_sample, len(pool)), replace=False))
    # float64 on both sides: the pipeline ranks in float64, so handing copairs
    # float32 would compare precision rather than implementation
    m, f = meta.iloc[idx].reset_index(drop=True), feats[idx].astype(np.float64)
    m = m.assign(
        genes=[sorted(g) or ["__none__"] for g in metadata.label_sets(m, "ensembl_ids")],
        locs=[sorted(l) or ["__none__"] for l in metadata.label_sets(m, "locations")])

    if task == "gene":
        ref = ml_ap(m, f, pos_sameby=["genes"], pos_diffby=["if_plate_id"],
                    neg_sameby=[], neg_diffby=["genes"],
                    multilabel_col="genes", progress_bar=False)
        ref_label = "genes"
    else:
        ref = ml_ap(m, f, pos_sameby=["locs"],
                    pos_diffby=["if_plate_id", "ensembl_ids"],
                    neg_sameby=[], neg_diffby=["locs"],
                    multilabel_col="locs", progress_bar=False)
        ref_label = "locs"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.nn.functional.normalize(torch.from_numpy(f).double().to(dev), dim=1)
    ctx = build_context(m, dev)
    r = (copairs_average_precision if ap_backend == "copairs"
         else gpu_average_precision)(x, ctx, task)
    vocab = {"gene": ctx["gene_vocab"], "location": ctx["loc_vocab"],
             "cell_line": ctx["cl_vocab"]}[task]
    del x, ctx

    gpu = pd.DataFrame({
        "fov": m["plate_position_sample"].values[r["profile"]],
        "label": [vocab[i] for i in r["label"]],
        "ap_gpu": r["ap"], "npos_gpu": r["n_pos"], "ntot_gpu": r["n_tot"]})
    refd = pd.DataFrame({
        "fov": ref["plate_position_sample"].values,
        "label": ref[ref_label].values,
        "ap_ref": ref["average_precision"].values,
        "npos_ref": ref["n_pos_pairs"].values,
        "ntot_ref": ref["n_total_pairs"].values})
    j = refd.merge(gpu, on=["fov", "label"], how="inner").dropna(
        subset=["ap_ref", "ap_gpu"])
    return {
        "task": task, "n_matched": int(len(j)),
        "n_ref_rows": int(len(refd)), "n_gpu_rows": int(len(gpu)),
        "n_pos_match": bool((j.npos_ref == j.npos_gpu).all()),
        "n_total_match": bool((j.ntot_ref == j.ntot_gpu).all()),
        "max_abs_diff": float((j.ap_ref - j.ap_gpu).abs().max()),
        "mean_ref": float(j.ap_ref.mean()), "mean_gpu": float(j.ap_gpu.mean()),
    }


def write_retrieval(rows, aggs):
    config.ensure(RETRIEVAL_DIR)
    df = pd.DataFrame(rows)
    df[["method", "harmonized"]] = df["key"].str.extract(
        r"^(.*?)_(none|microscope|atlas_name|if_plate_id.*)$")
    df.to_csv(RETRIEVAL_DIR / "retrieval_map.csv", index=False)
    for (key, task), agg in aggs.items():
        agg.to_csv(RETRIEVAL_DIR / "ap" / f"{key}__{task}__grouped.csv",
                   index=False)

    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        log("")
        log("=" * 72)
        log(f"  {task.upper()} RETRIEVAL -- copairs mAP  "
            f"(median pos/total pairs {sub['median_pos_pairs'].median():.0f}/"
            f"{sub['median_total_pairs'].median():.0f})")
        log("=" * 72)
        print(sub.sort_values("mAP", ascending=False)[
            ["method", "harmonized", "mAP", "ci_lo", "ci_hi", "chance_mAP",
             "frac_significant", "n_labels"]].to_string(index=False,
                                                        float_format="%.4f"))
    log("")
    log(f"wrote {RETRIEVAL_DIR / 'retrieval_map.csv'}")
    return df


def check_copairs(ap_backend=AP_BACKEND, tasks=("gene", "location")):
    """Compare the AP implementation with copairs on one embedding (gene, location)."""
    log(f"vectorized normalize_ap vs copairs: max|diff|={validate_normalize_ap():.2e}")
    method = naming.HARM_METHODS[0]
    harmonized = metadata.methods_table()["Harmonized"].unique()[0]
    feats, meta = features.fov_features(method, harmonized)
    results = []
    for task in tasks:
        r = validate_against_copairs(feats, meta, task, ap_backend=ap_backend)
        log(f"{method}_{harmonized} {r}")
        results.append(r)
    return results


def main(check=False, ap_backend=AP_BACKEND):
    if check:
        return check_copairs(ap_backend)
    log(f"vectorized normalize_ap vs copairs: max|diff|={validate_normalize_ap():.2e}")
    rows, aggs = [], {}
    for method in naming.HARM_METHODS:
        for harmonized in metadata.methods_table()["Harmonized"].unique():
            key = f"{method}_{harmonized}"
            log(f"=== {key}")
            feats, meta = features.fov_features(method, harmonized)
            for task in TASKS:
                r, agg = retrieval_map(key, feats, meta, task, ap_backend)
                rows.append(r)
                aggs[(key, task)] = agg
    return write_retrieval(rows, aggs)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check-copairs", action="store_true",
                   help="validate the AP implementation against copairs on a subset, then exit")
    p.add_argument("--ap-backend", choices=["gpu", "copairs"], default=AP_BACKEND)
    a = p.parse_args()
    main(a.check_copairs, a.ap_backend)
