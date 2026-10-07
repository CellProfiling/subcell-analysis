"""
Aggregate cell-level embeddings to well-level profiles.

Two-stage hierarchy (matches old ankit_eval/aggregation_scripts):
    1. cells -> FOV : mean OR median across all cells in each FOV
    2. FOVs  -> well: always mean across FOV vectors

(The old scripts had a bug in the "mean" branch — they called np.mean on the
list of filepaths instead of the loaded arrays. Fixed here.)

Input:  {EMBED_CELL_ROOT}/{model}/{plate}/{well}.pth
        dict(fov_ids: list[str], cell_ids: list[str], features: Tensor[n_cells, D])
Output: {EMBED_WELL_ROOT}/{model}_well_{agg}.csv
        columns: plate, well, cell_type, timepoint, feature_0, ..., feature_{D-1}

Wells with no cell embeddings are skipped (not zero-filled), so the output
row count may be slightly under n_plates * 384.

Usage:
    python scripts/aggregate_cells.py --model subcell_mae --agg mean
    python scripts/aggregate_cells.py --model dino --agg median
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)

from utils.metadata import load_plate_info  # noqa: E402


AGG_FNS = {
    "mean": lambda x: np.mean(x, axis=0),
    "median": lambda x: np.median(x, axis=0),
}


def aggregate_well(fov_ids: list[str], features: np.ndarray, agg: str) -> np.ndarray:
    """
    Aggregate one well's cell features to a single well vector.

    Args:
        fov_ids: length-n list of FOV identifiers (one per cell).
        features: (n, D) array of cell embeddings.
        agg: "mean" or "median" — used for the cells->FOV step.
             The FOVs->well step is always a mean.

    Returns:
        (D,) well-level embedding.
    """
    agg_fn = AGG_FNS[agg]
    fov_ids = np.asarray(fov_ids)

    fov_vectors = []
    for fov in np.unique(fov_ids):
        mask = fov_ids == fov
        fov_vectors.append(agg_fn(features[mask]))

    return np.mean(np.stack(fov_vectors), axis=0)


def aggregate_model(model: str, agg: str) -> pd.DataFrame:
    """
    Aggregate all plates for one model. Returns long-form DataFrame with
    one row per (plate, well).
    """
    cell_root = os.path.join(os.environ["EMBED_CELL_ROOT"], model)
    plate_info = load_plate_info().set_index("plate")

    if not os.path.isdir(cell_root):
        raise FileNotFoundError(
            f"No cell embeddings at {cell_root}. Run inference_*.py first."
        )

    plates = [p for p in plate_info.index if os.path.isdir(os.path.join(cell_root, p))]
    missing = set(plate_info.index) - set(plates)
    if missing:
        print(f"[warn] {len(missing)} plates not found on disk (skipping): {sorted(missing)}")

    rows = []
    feat_dim = None

    for plate in plates:
        plate_dir = os.path.join(cell_root, plate)
        well_files = sorted(f for f in os.listdir(plate_dir) if f.endswith(".pth"))

        for well_file in tqdm(well_files, desc=plate):
            well = well_file.removesuffix(".pth")
            data = torch.load(os.path.join(plate_dir, well_file), weights_only=False)

            features = data["features"].numpy()
            if feat_dim is None:
                feat_dim = features.shape[1]

            well_vec = aggregate_well(data["fov_ids"], features, agg)
            rows.append({"plate": plate, "well": well, "_feat": well_vec})

    if not rows:
        raise RuntimeError(f"No well embeddings produced for model={model}")

    # Build DataFrame: metadata cols + feature cols
    meta_df = pd.DataFrame([{"plate": r["plate"], "well": r["well"]} for r in rows])
    feat_arr = np.stack([r["_feat"] for r in rows])
    feat_df = pd.DataFrame(feat_arr, columns=[f"feature_{i}" for i in range(feat_dim)])

    df = pd.concat([meta_df, feat_df], axis=1)

    # Merge in cell_type + timepoint from plates.yaml
    df = df.merge(
        plate_info[["cell_type", "timepoint"]].reset_index(),
        on="plate",
        how="left",
    )

    # Reorder: metadata first, then features
    meta_cols = ["plate", "well", "cell_type", "timepoint"]
    feat_cols = [f"feature_{i}" for i in range(feat_dim)]
    return df[meta_cols + feat_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model name = subdir under EMBED_CELL_ROOT "
                             "(e.g. subcell_mae, subcell_vit, dino)")
    parser.add_argument("--agg", required=True, choices=["mean", "median"],
                        help="Aggregation method for cells->FOV. FOVs->well is always mean.")
    args = parser.parse_args()

    out_root = os.environ["EMBED_WELL_ROOT"]
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"{args.model}_well_{args.agg}.csv")

    print(f"Aggregating {args.model} ({args.agg})...")
    df = aggregate_model(args.model, args.agg)

    print(f"  {len(df)} wells, {df.shape[1] - 4} features")
    print(f"  cell types: {df['cell_type'].value_counts().to_dict()}")

    df.to_csv(out_path, index=False)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
