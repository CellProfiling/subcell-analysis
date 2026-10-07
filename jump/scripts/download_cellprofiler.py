"""
Fetch pre-aggregated CellProfiler well-level profiles from the
cellpainting-gallery public S3 bucket and assemble into one CSV
matching the schema of aggregate_cells.py output.

Each plate's {plate}.csv.gz is already well-aggregated (one row per
well, 5792 named CellProfiler feature columns). We just concat the 15
compound plates and merge in cell_type/timepoint from plates.yaml.

Unlike the neural models, there is only one CP aggregation (no
mean/median variants), so the output is cellprofiler_well.csv with no
suffix.

Output: {EMBED_WELL_ROOT}/cellprofiler_well.csv
    columns: plate, well, cell_type, timepoint, Cells_AreaShape_Area, ...

Usage:
    python scripts/download_cellprofiler.py
"""

import gzip
import io
import os
import subprocess
import sys
import tempfile

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)

from utils.metadata import load_plate_info  # noqa: E402


def fetch_plate_profile(bucket: str, prefix: str, plate: str) -> pd.DataFrame:
    """Download and parse one plate's raw aggregated CellProfiler profile."""
    url = f"https://{bucket}.s3.amazonaws.com/{prefix}/{plate}/{plate}.csv.gz"
    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=True) as tmp:
        result = subprocess.run(
            ["curl", "-sS", "-f", "-o", tmp.name, url],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed for {plate}: {result.stderr.strip()}")
        with open(tmp.name, "rb") as f:
            compressed = f.read()
    df = pd.read_csv(io.BytesIO(gzip.decompress(compressed)))
    df = df.rename(columns={"Metadata_Plate": "plate", "Metadata_Well": "well"})
    return df


def main():
    bucket = os.environ["CP_S3_BUCKET"]
    prefix = os.environ["CP_S3_PREFIX"]
    out_root = os.environ["EMBED_WELL_ROOT"]
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, "cellprofiler_well_median.csv") #paper confirms, aggregation was median

    if os.path.exists(out_path):
        df = pd.read_csv(out_path, nrows=0)
        print(f"Already exists: {out_path} ({len(df.columns)} cols). "
              f"Delete to re-download.")
        return

    plate_info = load_plate_info().set_index("plate")
    plates = list(plate_info.index)
    print(f"Fetching {len(plates)} plates from s3://{bucket}/{prefix}/")

    parts = []
    for plate in tqdm(plates):
        parts.append(fetch_plate_profile(bucket, prefix, plate))

    df = pd.concat(parts, ignore_index=True)

    # Merge cell_type + timepoint
    df = df.merge(
        plate_info[["cell_type", "timepoint"]].reset_index(),
        on="plate",
        how="left",
    )

    # Reorder: metadata first, then CP features
    meta_cols = ["plate", "well", "cell_type", "timepoint"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    print(f"  {len(df)} wells, {len(feat_cols)} features")
    print(f"  cell types: {df['cell_type'].value_counts().to_dict()}")

    df.to_csv(out_path, index=False)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
