"""
Convert DeepProfiler per-site .npz features to per-well .pth files
matching the format expected by aggregate_cells.py.

DeepProfiler output:  {DP_FEATURES}/{plate}/{well}/{site}.npz
    keys: features (n_cells, 672), metadata, locations

Target format:        {EMBED_CELL_ROOT}/deepprofiler/{plate}/{well}.pth
    dict(fov_ids: list[str], cell_ids: list[str], features: Tensor[n_cells, D])

Usage:
    python scripts/convert_deepprofiler_to_pth.py --dp-features /path/to/features
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
from tqdm import tqdm


def convert_well(site_files: list[str]) -> dict:
    """Load all site .npz files for one well and merge into a single dict."""
    all_feats = []
    fov_ids = []
    cell_ids = []
    cell_counter = 0

    for npz_path in sorted(site_files):
        site = os.path.basename(npz_path).removesuffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        feats = data["features"]
        n_cells = feats.shape[0]

        all_feats.append(feats)
        fov_ids.extend([site] * n_cells)
        cell_ids.extend([str(cell_counter + i) for i in range(n_cells)])
        cell_counter += n_cells

    return {
        "fov_ids": fov_ids,
        "cell_ids": cell_ids,
        "features": torch.from_numpy(np.concatenate(all_feats, axis=0)),
    }


def main(dp_features, out_root):
    plates = sorted(
        d for d in os.listdir(dp_features)
        if os.path.isdir(os.path.join(dp_features, d))
    )

    for plate in plates:
        plate_dir = os.path.join(dp_features, plate)
        wells = sorted(
            d for d in os.listdir(plate_dir)
            if os.path.isdir(os.path.join(plate_dir, d))
        )

        out_plate_dir = os.path.join(out_root, plate)
        os.makedirs(out_plate_dir, exist_ok=True)

        for well in tqdm(wells, desc=plate):
            site_files = glob(os.path.join(plate_dir, well, "*.npz"))
            if not site_files:
                continue

            well_data = convert_well(site_files)
            torch.save(well_data, os.path.join(out_plate_dir, f"{well}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-features",
        required=True,
        help="Path to DeepProfiler features dir "
             "(e.g. /scratch/.../outputs/results/features/)",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output dir. Defaults to $EMBED_CELL_ROOT/deepprofiler.",
    )
    args = parser.parse_args()

    out_root = args.out_root or os.path.join(
        os.environ["EMBED_CELL_ROOT"], "deepprofiler"
    )
    main(args.dp_features, out_root)
