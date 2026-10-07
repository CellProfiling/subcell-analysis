"""
Compute 2D UMAP coordinates for a single processed embedding parquet.

Reads a parquet from embeddings/processed/, merges full metadata via
build_full_metadata(), runs UMAP with cosine distance, and saves a CSV
with all metadata columns + umap_1, umap_2.

Output mirrors the input directory structure under results/umaps/:
    embeddings/processed/{model}/{cell_type}/{stem}.parquet
    -> results/umaps/{model}/{cell_type}/{stem}.csv

Usage:
    python scripts/compute_umap.py path/to/embeddings.parquet
    python scripts/compute_umap.py path/to/embeddings.parquet --n-neighbors 30
    python scripts/compute_umap.py path/to/embeddings.parquet --seed 0
"""

import os
import sys

import pandas as pd
import umap

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)

from utils.metadata import build_full_metadata  # noqa: E402

META_COLS = ["plate", "well", "cell_type", "timepoint", "compound"]


def main(input_path: str, n_neighbors: int, min_dist: float, seed: int):
    # --- Load embeddings ---
    df = pd.read_parquet(input_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # --- Merge full metadata ---
    meta = build_full_metadata()
    # Drop columns already in the parquet to avoid duplicates on merge
    meta_extra = meta.drop(columns=[c for c in META_COLS if c in meta.columns],
                           errors="ignore")
    meta_key = meta[["plate", "well"]].join(meta_extra)
    df = df.merge(meta_key, on=["plate", "well"], how="left")

    # --- Compute UMAP ---
    features = df[feat_cols].values
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(features)
    df["umap_1"] = coords[:, 0]
    df["umap_2"] = coords[:, 1]

    # --- Build output path ---
    embed_root = os.environ["EMBED_PROCESSED_ROOT"]
    results_root = os.environ["RESULTS_ROOT"]
    rel = os.path.relpath(input_path, embed_root)
    mindist_str = f"{min_dist:.2f}".replace(".", "")
    umap_dir = f"umaps_NN{n_neighbors}_mindist{mindist_str}"
    out_path = os.path.join(results_root, umap_dir, os.path.splitext(rel)[0] + ".csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- Save: metadata + umap coords only (drop feature columns) ---
    out_cols = [c for c in df.columns if c not in feat_cols]
    df[out_cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute UMAP for processed embeddings.")
    parser.add_argument("input", help="Path to a processed embedding parquet file.")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.input, args.n_neighbors, args.min_dist, args.seed)
