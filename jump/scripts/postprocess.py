"""
Run the full postprocessing grid over well-level embeddings.

For each input CSV in {EMBED_WELL_ROOT} and each cell_type in it:
    [optional feature_select] -> normalize_step1 -> normalize_step2

Normalization is run PER CELL TYPE because the user evaluates A549 and
U2OS separately — spherize fit on A549 DMSO shouldn't contaminate U2OS
and vice versa. Per-plate standardize is also implicitly per-cell-type
since no plate mixes cell types.

Grid: 2 feature_select (on/off) * 18 norm combos = 36 outputs per
(input CSV, cell_type) pair.

Output: {EMBED_PROCESSED_ROOT}/{model}/{cell_type}/{agg}__fs{0|1}__{norm1}__{norm2}.parquet
    columns: plate, well, cell_type, timepoint, compound, <surviving features>

Input filenames are parsed as: {model}_well_{agg}.csv or {model}_well.csv.

Usage:
    python scripts/postprocess.py                          # full grid, all inputs
    python scripts/postprocess.py --input subcell_mae_well_mean.csv
    python scripts/postprocess.py --cell-type U2OS
    python scripts/postprocess.py --dry-run                # list outputs, don't compute
"""

import argparse
import os
import re
import sys
from glob import glob

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)

from utils.metadata import load_well_compound_map  # noqa: E402
from utils.postprocess_utils import apply_pipeline, get_norm_grid  # noqa: E402


META_COLS = ["plate", "well", "cell_type", "timepoint", "compound"]


def parse_input_name(fname: str) -> tuple[str, str]:
    stem = os.path.basename(fname).removesuffix(".csv")
    m = re.fullmatch(r"(.+)_well(?:_(.+))?", stem)
    model, agg = m.group(1), m.group(2)
    return model, agg


def norm_label(method: str | None) -> str:
    """Filesystem-safe label for a normalization method."""
    return "none" if method is None else method.replace("-", "")


def load_and_prep(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a well-level CSV and merge in the compound column (needed for
    spherize's DMSO filter). Returns (df, feature_cols).
    """
    df = pd.read_csv(csv_path)

    # Merge compound on well (same 384-well platemap across all plates)
    compound_map = load_well_compound_map()[["well", "compound"]]
    df = df.merge(compound_map, on="well", how="left")

    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Drop rows where all features are NaN (wells with no detected cells).
    # These cause SVD failures in spherize via NaN propagation.
    all_nan = df[feature_cols].isna().all(axis=1)
    if all_nan.any():
        print(f"  Dropping {all_nan.sum()} rows with all-NaN features")
        df = df[~all_nan].reset_index(drop=True)

    return df[META_COLS + feature_cols], feature_cols


def build_jobs(input_csvs: list[str], cell_type_filter: str | None) -> list[dict]:
    """Enumerate all (input, cell_type, fs, norm1, norm2) jobs."""
    norm_grid = get_norm_grid()  # 18 (norm1, norm2) tuples
    jobs = []
    for csv_path in input_csvs:
        model, agg = parse_input_name(csv_path)
        # Peek at cell types (cheap — just one column)
        cell_types = pd.read_csv(csv_path, usecols=["cell_type"])["cell_type"].unique()
        for ct in sorted(cell_types):
            if cell_type_filter and ct != cell_type_filter:
                continue
            for do_fs in (False, True):
                for norm1, norm2 in norm_grid:
                    out_name = f"{agg}__fs{int(do_fs)}__{norm_label(norm1)}__{norm_label(norm2)}.parquet"
                    jobs.append({
                        "csv_path": csv_path,
                        "model": model,
                        "agg": agg,
                        "cell_type": ct,
                        "do_fs": do_fs,
                        "norm1": norm1,
                        "norm2": norm2,
                        "out_path": os.path.join(
                            os.environ["EMBED_PROCESSED_ROOT"], model, ct, out_name
                        ),
                    })
    return jobs


def run_job(job: dict, df_ct: pd.DataFrame, feature_cols: list[str]):
    """Run one postprocessing combo on a pre-filtered cell-type slice."""
    out_df, _ = apply_pipeline(
        df_ct,
        feature_cols=feature_cols,
        metadata_cols=META_COLS,
        do_feature_select=job["do_fs"],
        norm_step1=job["norm1"],
        norm_step2=job["norm2"],
    )
    os.makedirs(os.path.dirname(job["out_path"]), exist_ok=True)
    out_df.to_parquet(job["out_path"], index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="Single input filename (e.g. subcell_mae_well_mean.csv). "
                             "Default: all CSVs in EMBED_WELL_ROOT.")
    parser.add_argument("--cell-type", default=None, choices=[None, "A549", "U2OS"],
                        help="Restrict to one cell type.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without running.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Overwrite existing outputs instead of skipping.")
    args = parser.parse_args()

    well_root = os.environ["EMBED_WELL_ROOT"]
    if args.input:
        input_csvs = [os.path.join(well_root, args.input)]
    else:
        input_csvs = sorted(glob(os.path.join(well_root, "*.csv")))

    if not input_csvs:
        print(f"No input CSVs found in {well_root}")
        return

    jobs = build_jobs(input_csvs, args.cell_type)

    if not args.no_skip:
        pending = [j for j in jobs if not os.path.exists(j["out_path"])]
        n_skipped = len(jobs) - len(pending)
        if n_skipped:
            print(f"Skipping {n_skipped} existing outputs (use --no-skip to overwrite).")
        jobs = pending

    print(f"{len(jobs)} jobs across {len(input_csvs)} input file(s)")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['model']:<14} {j['cell_type']:<6} {os.path.basename(j['out_path'])}")
        return

    # Group jobs by (csv, cell_type) so we load+filter each input once
    from itertools import groupby
    key = lambda j: (j["csv_path"], j["cell_type"])
    jobs.sort(key=key)

    current_csv = None
    df_full = feature_cols = None

    for (csv_path, cell_type), group in groupby(jobs, key=key):
        group = list(group)

        if csv_path != current_csv:
            print(f"\nLoading {os.path.basename(csv_path)}...")
            df_full, feature_cols = load_and_prep(csv_path)
            current_csv = csv_path
            print(f"  {len(df_full)} wells, {len(feature_cols)} features")

        df_ct = df_full[df_full["cell_type"] == cell_type].reset_index(drop=True)
        n_dmso = (df_ct["compound"] == "DMSO").sum()
        print(f"  {cell_type}: {len(df_ct)} wells ({n_dmso} DMSO) -> {len(group)} combos")

        for job in tqdm(group, desc=f"  {cell_type}"):
            try:
                run_job(job, df_ct, feature_cols)
            except Exception as e:
                print(f"\n  [FAIL] {os.path.basename(job['out_path'])}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
