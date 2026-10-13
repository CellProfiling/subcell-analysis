"""Build the metadata CSV for the Kaggle HPA single-cell test crops (Fig. 2a,b).

Reads ``all_crops.csv``, the per-cell table that accompanies the test crops (one row
per cell: ``if_plate_id``, ``position``, ``sample``, ``cell_id``, ``antibody``,
``annotated_label``, ...), drops cells whose antibody is in the HPA train or valid
split (``config.HPA_SPLITS``), and writes the rows in the form ``HPATestDataset3``
reads: ``id`` is the image, ``cell`` is ``<n>_crop`` so the crop files
``<image>_cell<n>_crop_<color>.png`` resolve, and ``locations`` carries the expert
cell-level ``annotated_label``.

    python -m scripts.build_kaggle_metadata

Both paths default to files in ``config.ROOTS["KAGGLE_DATA"]``: ``all_crops.csv`` in,
``kaggle_crops_metadata.csv`` out (the ``metadata_file`` of ``configs/config_kaggle.yaml``).
"""

import argparse

import pandas as pd

from subcell import config

KAGGLE = config.ROOTS["KAGGLE_DATA"]


def main(all_crops_csv, out_csv):
    df = pd.read_csv(all_crops_csv, index_col=0)
    seen = set()
    for split in ("train", "valid"):
        seen |= set(pd.read_csv(config.HPA_SPLITS / f"{split}_antibodies.txt", header=None)[0])
    keep = ~df["antibody"].isin(seen)
    df = df[keep].reset_index(drop=True)
    df["id"] = df["if_plate_id"].astype(str) + "_" + df["position"] + "_" + df["sample"].astype(str)
    df["cell"] = df["cell_id"].astype(str) + "_crop"
    df["locations"] = df["annotated_label"]
    df = df.drop(columns=["cell_id"])
    df.to_csv(out_csv, index=False)
    print(f"kept {len(df)} of {len(keep)} cells ({df['id'].nunique()} images, "
          f"{df['antibody'].nunique()} antibodies) -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all-crops", default=str(KAGGLE / "all_crops.csv"))
    ap.add_argument("--out-csv", default=str(KAGGLE / "kaggle_crops_metadata.csv"))
    a = ap.parse_args()
    main(a.all_crops, a.out_csv)
