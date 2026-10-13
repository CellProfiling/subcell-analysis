"""Build the cell metadata CSV for a per-plate HPA crop tree.

Concatenates the per-image ``<plate>/*_cell_bbox.csv`` files under ``--crops-folder``
into one CSV with a ``cell_path`` column, keeping only cells whose
``_cell_image.png`` exists. This is the ``metadata_file`` that ``HPATestDataset2``
(``subcell.io.datasets``) reads.

    python -m scripts.build_hpa_cell_metadata --crops-folder DIR --out-csv FILE
"""

import argparse
import os
from glob import glob

import pandas as pd


def main(crops_folder, out_csv):
    df = pd.concat(
        [pd.read_csv(f) for f in sorted(glob(f"{crops_folder}/*/*_cell_bbox.csv"))],
        ignore_index=True,
    )
    df["cell_path"] = df.apply(
        lambda r: f"{crops_folder}/{r['if_plate_id']}/{r['if_plate_id']}_{r['position']}"
        f"_{r['sample']}_{r['cell_id']}_cell_image.png",
        axis=1,
    )
    on_disk = {f"{crops_folder}/{p}/{f}" for p in df["if_plate_id"].unique()
               for f in os.listdir(f"{crops_folder}/{p}")}
    df = df[df["cell_path"].isin(on_disk)].reset_index(drop=True)
    df.to_csv(out_csv)
    print(f"{len(df)} cells -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crops-folder", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()
    main(a.crops_folder, a.out_csv)
