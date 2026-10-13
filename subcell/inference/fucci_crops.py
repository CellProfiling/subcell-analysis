"""Single-cell crops of the FUCCI U2OS images, the input of the FUCCI cell-cycle analysis (Fig. 2k,l).

Each field of view (four channels, ``w1``-``w4``) is cut at the cells of its cell mask: a 512 x 512 px window centred
on the cell, multiplied by the cell mask, with the nuclear mask appended as a fifth channel, saved as
``<ab_id>/<fov_id>_<label>.npy``.

    python -m subcell.inference.fucci_crops [--jobs N]

Inputs (``FUCCI_DATA``): ``single_cell_statistics_pcs.csv``, ``Files/`` (the fields), ``cell_masks/``.
Output: ``single_cell_crops/`` there (replaced if it exists), the ``crops_folder`` of the configs
``fucci_subcell_*.yaml``. ``subcell.inference.fucci`` encodes them.
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops
from tqdm import tqdm

from subcell import config
from subcell.io.preprocess import safe_crop

CROP_SIZE = (512, 512)


def process_fov_row(row, df, crops_folder, mask_folder, save_folder, crop_size=CROP_SIZE):
    """Process a single FOV row to create cell crops."""
    try:
        fov_path = os.path.join(
            crops_folder, row["ab_id"], row["cell_id"].rsplit("_", 1)[0]
        )
        img = np.stack(
            [
                cv2.imread(f"{fov_path}_{channel}.tif", -1)
                for channel in ["w1", "w2", "w3", "w4"]
            ],
            axis=-1,
        )
        fovcellmask = cv2.imread(
            f"{fov_path.replace(crops_folder, mask_folder)}_cellmask.png", -1
        )
        fovnucmask = cv2.imread(
            f"{fov_path.replace(crops_folder, mask_folder)}_nucleimask.png", -1
        )

        fovregionprops = regionprops(fovcellmask)

        cell_ids = list(
            df[(df["fov_id"] == row["fov_id"]) & (df["ab_id"] == row["ab_id"])][
                "cell_id"
            ].apply(lambda x: int(x.rsplit("_", 1)[1]))
        )
        region_labels = set([region.label for region in fovregionprops])
        if region_labels != set(cell_ids):
            print(
                f"Warning: rows and mask don't match for {row['ab_id']} and {row['fov_id']}"
            )
            return f"Error: {row['ab_id']} - {row['fov_id']}"

        for region in fovregionprops:
            cell_center = [
                (region.bbox[0] + region.bbox[2]) // 2,
                (region.bbox[1] + region.bbox[3]) // 2,
            ]
            cell_bbox = [
                cell_center[0] - crop_size[0] // 2,
                cell_center[1] - crop_size[1] // 2,
                cell_center[0] + crop_size[0] // 2,
                cell_center[1] + crop_size[1] // 2,
            ]

            cell_img, _ = safe_crop(img, cell_bbox)
            cell_mask, _ = safe_crop(fovcellmask, cell_bbox)
            cell_mask = cell_mask == region.label
            cell_img = cell_img * cell_mask[:, :, None]

            nuc_mask, _ = safe_crop(fovnucmask, cell_bbox)
            nuc_mask = (nuc_mask == region.label).astype(cell_img.dtype)

            cell_img = np.concatenate([cell_img, nuc_mask[:, :, None]], axis=-1)

            cell_save_path = os.path.join(
                save_folder, f"{row['ab_id']}/{row['fov_id']}_{region.label}.npy"
            )
            os.makedirs(os.path.dirname(cell_save_path), exist_ok=True)
            np.save(cell_save_path, cell_img)

        return f"Success: {row['ab_id']} - {row['fov_id']}"

    except Exception as e:
        return f"Error processing {row['ab_id']} - {row['fov_id']}: {str(e)}"


def main(jobs=16):
    """Cut every cell of ``single_cell_statistics_pcs.csv`` out of its field."""
    root = config.ROOTS["FUCCI_DATA"]
    df = pd.read_csv(root / "single_cell_statistics_pcs.csv")
    crops_folder = str(root / "Files")
    mask_folder = str(root / "cell_masks")
    save_folder = str(root / "single_cell_crops")
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)

    df["fov_id"] = df["cell_id"].apply(lambda x: x.rsplit("_", 1)[0])
    fov_df = df.groupby(["ab_id", "fov_id"]).first().reset_index()[["ab_id", "fov_id", "cell_id"]]

    results = Parallel(n_jobs=jobs, verbose=10)(
        delayed(process_fov_row)(row, df, crops_folder, mask_folder, save_folder)
        for _, row in tqdm(fov_df.iterrows(), total=len(fov_df), desc="Processing FOVs")
    )
    success_count = sum(1 for result in results if result.startswith("Success"))
    print(f"Processed {len(results)} FOVs: {success_count} succeeded, {len(results) - success_count} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=16, help="parallel workers")
    main(parser.parse_args().jobs)
