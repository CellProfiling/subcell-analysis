"""
Compute copairs mAP and nearest-neighbor accuracy for every processed embedding.

For each parquet in {EMBED_PROCESSED_ROOT}/{model}/{cell_type}/*.parquet:
    1. Merge in MoA annotations
    2. Drop DMSO wells (not queries — DMSO has no replicates by definition)
    3. Replicate mAP: same compound, different plate
    4. MoA mAP:       same MoA, different plate AND different compound
                      (multilabel — one compound can have several MoAs)
    5. Compound NN accuracy (k=1..50): nearest cross-plate neighbor shares compound?
    6. MoA NN accuracy (k=1..50): nearest cross-plate neighbor shares MoA?
       (restricted to profiles with known MoA annotations)

Pair design (CellFlux convention, confirmed with user):
    replicate: pos_sameby=[compound], pos_diffby=[plate]
               neg_sameby=[plate],    neg_diffby=[compound]
    MoA:       pos_sameby=[moas],     pos_diffby=[plate, compound]
               neg_sameby=[plate],    neg_diffby=[moas]

Output:
    {RESULTS_ROOT}/results_summary.csv — one row per (parquet, task) with aggregate scores
    {RESULTS_ROOT}/map_per_group/{model}__{cell_type}__{stem}__{task}.csv — per-compound/per-MoA breakdown

Summary columns:
    model, cell_type, agg, fs, norm1, norm2, task,
    mAP, frac_retrievable, n_groups, n_queries, accuracy

Usage:
    python scripts/compute_map.py                      # all parquets
    python scripts/compute_map.py --model subcell_mae  # filter to one model
    python scripts/compute_map.py --null-size 10000    # more null samples (slower, better p-values)
"""

import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import pandas as pd
from copairs import map as cp_map
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)

from utils.metadata import load_moa_map  # noqa: E402


META_COLS = ["plate", "well", "cell_type", "timepoint", "compound"]


def parse_parquet_path(path: str) -> dict:
    """
    Parse a processed-embedding path into identifying keys.

    Path: {root}/{model}/{cell_type}/{agg}__fs{0|1}__{norm1}__{norm2}.parquet
    """
    parts = path.split(os.sep)
    model, cell_type = parts[-3], parts[-2]
    stem = os.path.basename(path).removesuffix(".parquet")
    agg, fs, norm1, norm2 = stem.split("__")
    return {
        "model": model,
        "cell_type": cell_type,
        "agg": agg,
        "fs": int(fs.removeprefix("fs")),
        "norm1": norm1,
        "norm2": norm2,
        "_stem": stem,
    }


def compute_replicate_map(df: pd.DataFrame, feat_cols: list[str],
                          null_size: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """
    Replicate retrieval: same compound across different plates.

    Returns (summary_dict, per_compound_df).
    """
    ap = cp_map.average_precision(
        df[["plate", "well", "compound"]],
        df[feat_cols].to_numpy(),
        pos_sameby=["compound"],
        pos_diffby=["plate"],
        neg_sameby=["plate"],
        neg_diffby=["compound"],
    )
    mAP = cp_map.mean_average_precision(
        ap, sameby=["compound"], null_size=null_size, threshold=0.05, seed=seed,
    )
    summary = {
        "mAP": mAP["mean_average_precision"].mean(),
        "frac_retrievable": mAP["below_corrected_p"].mean(),
        "n_groups": len(mAP),
        "n_queries": len(ap),
    }
    return summary, mAP


def compute_moa_map(df: pd.DataFrame, feat_cols: list[str], moa_map: pd.DataFrame,
                    null_size: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """
    MoA retrieval: same MoA across different plates AND different compounds
    (so we're testing generalization, not just re-finding the same compound).

    Multilabel — one compound can have several MoAs, so copairs expands to
    one query per (well, moa_label) pair.

    Returns (summary_dict, per_moa_df).
    """
    # Inner join drops compounds with no "usable" MoA annotation
    df_moa = df.merge(moa_map, on="compound", how="inner").reset_index(drop=True)

    if df_moa.empty:
        return {"mAP": np.nan, "frac_retrievable": np.nan,
                "n_groups": 0, "n_queries": 0}, pd.DataFrame()

    ap = cp_map.multilabel.average_precision(
        df_moa[["plate", "well", "compound", "moas"]],
        df_moa[feat_cols].to_numpy(),
        pos_sameby=["moas"],
        pos_diffby=["plate", "compound"],
        neg_sameby=["plate"],
        neg_diffby=["moas"],
        multilabel_col="moas",
    )
    mAP = cp_map.mean_average_precision(
        ap, sameby=["moas"], null_size=null_size, threshold=0.05, seed=seed,
    )
    summary = {
        "mAP": mAP["mean_average_precision"].mean(),
        "frac_retrievable": mAP["below_corrected_p"].mean(),
        "n_groups": len(mAP),
        "n_queries": len(ap),
    }
    return summary, mAP


def _cosine_knn(feats: np.ndarray, mask: np.ndarray, max_k: int) -> np.ndarray:
    """
    Compute cosine similarity, apply a boolean exclusion mask, and return
    top-max_k neighbor indices per query sorted by descending similarity.
    """
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    feats_norm = feats / norms
    sim = feats_norm @ feats_norm.T

    sim[mask] = -np.inf

    # Top-max_k neighbors per query (descending similarity)
    top_k_indices = np.argpartition(-sim, kth=max_k, axis=1)[:, :max_k]
    row_idx = np.arange(len(sim))[:, None]
    top_k_sims = sim[row_idx, top_k_indices]
    order = np.argsort(-top_k_sims, axis=1)
    top_k_indices = top_k_indices[row_idx, order]

    return top_k_indices


def _nn_accuracy(match_matrix: np.ndarray, max_k: int) -> dict:
    """
    From a (n_queries, max_k) boolean match matrix, compute hit-rate
    accuracy for k=1..max_k. Returns dict with keys nn_k1, nn_k2, ...
    """
    result = {}
    cumulative = np.zeros(match_matrix.shape[0], dtype=bool)
    for k in range(1, max_k + 1):
        cumulative |= match_matrix[:, k - 1]
        result[f"nn_k{k}"] = cumulative.mean()
    return result


def compute_nn_compound(df: pd.DataFrame, feat_cols: list[str],
                        max_k: int) -> dict:
    """
    Nearest-neighbor compound accuracy for k=1..max_k.

    For each non-DMSO query, find top-k cross-plate neighbors by cosine
    similarity and check if at least one shares the same compound.
    Returns dict with keys nn_k1..nn_k{max_k}.
    """
    feats = df[feat_cols].to_numpy().astype(np.float32)
    plates = df["plate"].to_numpy()
    compounds = df["compound"].to_numpy()

    mask = plates[:, None] == plates[None, :]
    top_k_idx = _cosine_knn(feats, mask, max_k)
    neighbor_compounds = compounds[top_k_idx]
    match_matrix = neighbor_compounds == compounds[:, None]

    return _nn_accuracy(match_matrix, max_k)


def compute_nn_moa(df: pd.DataFrame, feat_cols: list[str],
                   moa_map: pd.DataFrame, max_k: int) -> dict:
    """
    Nearest-neighbor MoA accuracy for k=1..max_k.

    Restricted to profiles with known MoA. A match means the query and
    neighbor share at least one MoA label (multilabel, pipe-separated).
    Returns dict with keys nn_k1..nn_k{max_k}.
    """
    df_moa = df.merge(moa_map, on="compound", how="inner").reset_index(drop=True)

    feats = df_moa[feat_cols].to_numpy().astype(np.float32)
    plates = df_moa["plate"].to_numpy()
    moa_sets = [set(str(m).split("|")) for m in df_moa["moas"].to_numpy()]

    compounds = df_moa["compound"].to_numpy()
    mask = (plates[:, None] == plates[None, :]) | (compounds[:, None] == compounds[None, :])
    top_k_idx = _cosine_knn(feats, mask, max_k)

    n = len(df_moa)
    match_matrix = np.zeros((n, max_k), dtype=bool)
    for i in range(n):
        for j_pos in range(max_k):
            match_matrix[i, j_pos] = bool(moa_sets[i] & moa_sets[top_k_idx[i, j_pos]])

    return _nn_accuracy(match_matrix, max_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Filter to one model (e.g. subcell_mae).")
    parser.add_argument("--cell-type", default=None, choices=[None, "A549", "U2OS"])
    parser.add_argument("--null-size", type=int, default=10000,
                        help="Null distribution size for p-value estimation.")
    parser.add_argument("--max-k", type=int, default=50,
                        help="Max k for nearest-neighbor accuracy (k=1..max_k).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-skip", action="store_true",
                        help="Recompute rows already in the summary CSV.")
    parser.add_argument("--output-suffix", default=None,
                        help="Write to results_summary_{suffix}.csv instead of "
                             "results_summary.csv. Useful for parallel jobs.")
    args = parser.parse_args()

    proc_root = os.environ["EMBED_PROCESSED_ROOT"]
    results_root = os.environ["RESULTS_ROOT"]
    per_group_dir = os.path.join(results_root, "map_per_group")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    summary_path = os.path.join(results_root, f"results_summary{suffix}.csv")
    os.makedirs(per_group_dir, exist_ok=True)

    # Discover parquets
    pattern = os.path.join(proc_root, args.model or "*", args.cell_type or "*", "*.parquet")
    parquets = sorted(glob(pattern))
    if not parquets:
        print(f"No parquets matched: {pattern}")
        return

    # Resume: skip parquets whose BOTH tasks already appear in the summary
    done = set()
    if os.path.exists(summary_path) and not args.no_skip:
        prev = pd.read_csv(summary_path)
        id_cols = ["model", "cell_type", "agg", "fs", "norm1", "norm2"]
        # A parquet is done if both replicate and moa tasks are present
        counts = prev.groupby(id_cols).size()
        done = set(counts[counts >= 2].index)
        print(f"Found {len(done)} completed parquets in existing summary.")

    moa_map = load_moa_map()  # [compound, moas]

    summary_rows = []
    for pq_path in tqdm(parquets):
        info = parse_parquet_path(pq_path)
        key = (info["model"], info["cell_type"], info["agg"],
               info["fs"], info["norm1"], info["norm2"])
        if key in done:
            continue

        df = pd.read_parquet(pq_path)
        df = df[df["compound"] != "DMSO"].reset_index(drop=True)
        feat_cols = [c for c in df.columns if c not in META_COLS]

        # --- Replicate mAP + compound NN ---
        rep_summary, rep_detail = compute_replicate_map(
            df, feat_cols, args.null_size, args.seed
        )
        nn_compound = compute_nn_compound(df, feat_cols, args.max_k)
        summary_rows.append({**info, "task": "replicate", **rep_summary, **nn_compound})
        rep_detail.to_csv(
            os.path.join(per_group_dir,
                         f"{info['model']}__{info['cell_type']}__{info['_stem']}__replicate.csv"),
            index=False,
        )

        # --- MoA mAP + MoA NN ---
        moa_summary, moa_detail = compute_moa_map(
            df, feat_cols, moa_map, args.null_size, args.seed
        )
        nn_moa = compute_nn_moa(df, feat_cols, moa_map, args.max_k)
        summary_rows.append({**info, "task": "moa", **moa_summary, **nn_moa})
        if not moa_detail.empty:
            moa_detail.to_csv(
                os.path.join(per_group_dir,
                             f"{info['model']}__{info['cell_type']}__{info['_stem']}__moa.csv"),
                index=False,
            )

        # Flush summary incrementally (append mode) so long runs are crash-safe
        flush_df = pd.DataFrame(summary_rows).drop(columns=["_stem"])
        if os.path.exists(summary_path) and not args.no_skip:
            flush_df = pd.concat([pd.read_csv(summary_path), flush_df], ignore_index=True)
            flush_df = flush_df.drop_duplicates(
                subset=["model", "cell_type", "agg", "fs", "norm1", "norm2", "task"],
                keep="last",
            )
        flush_df.to_csv(summary_path, index=False)
        summary_rows = []  # already flushed

    print(f"\nDone. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
