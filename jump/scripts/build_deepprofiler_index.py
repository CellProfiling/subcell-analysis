"""
Build index.csv for DeepProfiler by scanning downloaded images and
merging JUMP metadata for treatment labels.

Scans {DP_ROOT}/inputs/images/{plate}/Images/ for TIFFs, parses
plate/well/site/channel from filenames, and joins compound labels.

Output: {DP_ROOT}/inputs/metadata/index.csv

Usage:
    python scripts/build_deepprofiler_index.py
"""

import argparse
import os
import re
from glob import glob

import pandas as pd
import yaml

# JUMP TIFF filename pattern: r{row}c{col}f{fov}p01-ch{channel}sk1fk1fl1.tiff
TIFF_RE = re.compile(r"r(\d{2})c(\d{2})f(\d{2})p01-ch(\d)sk1fk1fl1\.tiff?$")

CHANNEL_MAP = {1: "Mito", 2: "AGP", 3: "RNA", 4: "ER", 5: "DNA"}


def row_col_to_well(row_num, col_num):
    """Convert numeric row/col (1-indexed) to well name like A01."""
    return f"{chr(64 + row_num)}{col_num:02d}"


def scan_images(dp_root):
    """Scan downloaded images and return a DataFrame of (plate, well, site, channel, rel_path)."""
    images_root = os.path.join(dp_root, "inputs", "images")
    records = []

    for tiff_path in glob(os.path.join(images_root, "*", "Images", "*.tiff")) + \
                      glob(os.path.join(images_root, "*", "Images", "*.tif")):
        fname = os.path.basename(tiff_path)
        m = TIFF_RE.match(fname)
        if not m:
            continue

        row_num, col_num, fov, ch = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if ch not in CHANNEL_MAP:
            continue

        plate = tiff_path.split(os.sep)[-3]
        well = row_col_to_well(row_num, col_num)
        # Relative path from inputs/images/
        rel_path = os.path.relpath(tiff_path, images_root)

        records.append({
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Site": fov,
            "channel": CHANNEL_MAP[ch],
            "path": rel_path,
        })

    return pd.DataFrame(records)


def load_treatment_map(project_root, metadata_root):
    """Build well -> treatment label from JUMP metadata."""
    platemap = pd.read_csv(
        os.path.join(metadata_root, "JUMP-Target-1_compound_platemap.txt"),
        sep="\t",
    )
    cmpd = pd.read_csv(
        os.path.join(metadata_root, "JUMP-Target-1_compound_metadata.tsv"),
        sep="\t",
    )
    df = platemap.merge(
        cmpd[["broad_sample", "pert_iname"]],
        on="broad_sample",
        how="left",
    )
    df["Treatment"] = df["pert_iname"].fillna("DMSO")
    return df[["well_position", "Treatment"]].rename(columns={"well_position": "Metadata_Well"})


def main(dp_root, project_root, metadata_root):
    print("Scanning images...")
    df = scan_images(dp_root)
    if df.empty:
        print("No images found. Run the download script first.")
        return

    # Pivot channels into columns
    index = df.pivot_table(
        index=["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
        columns="channel",
        values="path",
        aggfunc="first",
    ).reset_index()
    index.columns.name = None

    # Only keep rows where all 5 channels are present
    required = ["DNA", "RNA", "ER", "AGP", "Mito"]
    complete = index.dropna(subset=required)
    if len(complete) < len(index):
        print(f"Dropped {len(index) - len(complete)} rows with missing channels")
    index = complete

    # Merge treatment labels
    treatments = load_treatment_map(project_root, metadata_root)
    index = index.merge(treatments, on="Metadata_Well", how="left")
    index["Treatment"] = index["Treatment"].fillna("UNKNOWN")

    # Sort for reproducibility
    index = index.sort_values(["Metadata_Plate", "Metadata_Well", "Metadata_Site"]).reset_index(drop=True)

    # Reorder columns to match config
    index = index[["Metadata_Plate", "Metadata_Well", "Metadata_Site",
                    "DNA", "RNA", "ER", "AGP", "Mito", "Treatment"]]

    out_path = os.path.join(dp_root, "inputs", "metadata", "index.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    index.to_csv(out_path, index=False)
    print(f"Wrote {len(index)} rows to {out_path}")
    print(f"Plates: {sorted(index['Metadata_Plate'].unique())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-root", default=os.environ.get("DP_ROOT"),
                        help="DeepProfiler project root")
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"),
                        help="sucell_jump_analysis project root")
    parser.add_argument("--metadata-root", default=os.environ.get("METADATA_ROOT"),
                        help="JUMP metadata directory")
    args = parser.parse_args()
    main(args.dp_root, args.project_root, args.metadata_root)
