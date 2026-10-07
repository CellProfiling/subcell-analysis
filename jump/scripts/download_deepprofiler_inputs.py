"""
Download JUMP FOV images and Nuclei.csv files from the public
cellpainting-gallery S3 bucket and arrange them in the directory
layout DeepProfiler expects.

Images:  {DP_ROOT}/inputs/images/{plate}/Images/{filename}.tiff
Nuclei:  {DP_ROOT}/inputs/locations/{plate}/{well}-{site}-Nuclei.csv
         with columns Nuclei_Location_Center_X, Nuclei_Location_Center_Y

Usage:
    python scripts/download_deepprofiler_inputs.py --plate BR00117010
"""

import argparse
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "cellpainting-gallery"
IMAGE_PREFIX = "cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
NUCLEI_PREFIX = "cpg0000-jump-pilot/source_4/workspace/analysis/2020_11_04_CPJUMP1/"


def get_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def resolve_plate_prefix(s3, plate):
    """Find the full plate directory name (includes timestamp suffix).

    e.g. BR00116991 -> BR00116991__2020-11-05T19_51_35-Measurement1
    """
    r = s3.list_objects_v2(
        Bucket=BUCKET, Prefix=IMAGE_PREFIX + plate, Delimiter="/"
    )
    prefixes = [p["Prefix"] for p in r.get("CommonPrefixes", [])]
    if len(prefixes) == 0:
        raise FileNotFoundError(f"No image directory found for plate {plate}")
    if len(prefixes) > 1:
        print(f"  Warning: multiple directories for {plate}, using first: {prefixes}")
    return prefixes[0]


def _download_one_tiff(key, local_path):
    """Download a single TIFF. Each thread gets its own S3 client."""
    s3 = get_s3_client()
    s3.download_file(BUCKET, key, local_path)


def download_images(s3, plate, dp_root, n_threads):
    """Download all FOV TIFFs for a plate into inputs/images/{plate}/Images/."""
    plate_s3_prefix = resolve_plate_prefix(s3, plate)
    s3_prefix = plate_s3_prefix + "Images/"
    local_dir = os.path.join(dp_root, "inputs", "images", plate, "Images")
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    tiff_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tiff") or key.endswith(".tif"):
                tiff_keys.append(key)

    # Filter to files that don't already exist locally
    to_download = []
    skipped = 0
    for key in tiff_keys:
        fname = key.split("/")[-1]
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path):
            skipped += 1
        else:
            to_download.append((key, local_path))

    print(f"[{plate}] Images: {len(to_download)} to download, {skipped} already exist")

    downloaded = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(_download_one_tiff, key, local_path): key
            for key, local_path in to_download
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                downloaded += 1
            except Exception as e:
                fname = futures[future].split("/")[-1]
                print(f"  Error downloading {fname}: {e}")
                errors += 1
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(to_download)}")

    print(f"[{plate}] Images done: {downloaded} downloaded, {errors} errors")


def _download_one_nuclei(key, well, fov, local_path):
    """Download and reformat a single Nuclei.csv. Each thread gets its own S3 client."""
    s3 = get_s3_client()
    response = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))

    if "AreaShape_Center_X" not in df.columns:
        raise ValueError(f"Missing expected columns in {well}-{fov}")

    df = df[["AreaShape_Center_X", "AreaShape_Center_Y"]].rename(columns={
        "AreaShape_Center_X": "Nuclei_Location_Center_X",
        "AreaShape_Center_Y": "Nuclei_Location_Center_Y",
    })
    df.to_csv(local_path, index=False)


def download_nuclei(s3, plate, dp_root, n_threads):
    """Download Nuclei.csv files and reformat for DeepProfiler.

    S3:    .../{plate}/analysis/{plate}-{well}-{fov}/Nuclei.csv
    Local: inputs/locations/{plate}/{well}-{fov}-Nuclei.csv
    """
    local_dir = os.path.join(dp_root, "inputs", "locations", plate)
    os.makedirs(local_dir, exist_ok=True)

    analysis_prefix = NUCLEI_PREFIX + f"{plate}/analysis/{plate}-"
    paginator = s3.get_paginator("list_objects_v2")

    nuclei_re = re.compile(
        re.escape(NUCLEI_PREFIX) +
        re.escape(plate) +
        r"/analysis/" +
        re.escape(plate) +
        r"-([A-P]\d{2})-(\d+)/Nuclei\.csv$"
    )

    nuclei_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=analysis_prefix):
        for obj in page.get("Contents", []):
            m = nuclei_re.match(obj["Key"])
            if m:
                nuclei_keys.append((obj["Key"], m.group(1), m.group(2)))

    # Filter to files that don't already exist locally
    to_download = []
    skipped = 0
    for key, well, fov in nuclei_keys:
        local_path = os.path.join(local_dir, f"{well}-{fov}-Nuclei.csv")
        if os.path.exists(local_path):
            skipped += 1
        else:
            to_download.append((key, well, fov, local_path))

    print(f"[{plate}] Nuclei: {len(to_download)} to download, {skipped} already exist")

    downloaded = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(_download_one_nuclei, key, well, fov, local_path): (well, fov)
            for key, well, fov, local_path in to_download
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                downloaded += 1
            except Exception as e:
                well, fov = futures[future]
                print(f"  Error with {plate}-{well}-{fov}: {e}")
                errors += 1
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(to_download)}")

    print(f"[{plate}] Nuclei done: {downloaded} downloaded, {errors} errors")


def main(plate, dp_root, n_threads):
    s3 = get_s3_client()
    download_images(s3, plate, dp_root, n_threads)
    download_nuclei(s3, plate, dp_root, n_threads)
    print(f"[{plate}] All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plate", required=True, help="Plate barcode (e.g. BR00117010)")
    parser.add_argument("--dp-root", default=os.environ.get("DP_ROOT"),
                        help="DeepProfiler project root")
    parser.add_argument("--threads", type=int, default=16,
                        help="Number of download threads (default: 16)")
    args = parser.parse_args()
    main(args.plate, args.dp_root, args.threads)
