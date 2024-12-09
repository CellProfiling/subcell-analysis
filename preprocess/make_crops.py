import os
import shutil
import warnings
from glob import glob

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.measure import regionprops
from tqdm import tqdm

from utils import create_crops

warnings.simplefilter(action="ignore", category=FutureWarning)


def add_basename(df):
    df["basename"] = (
        df["if_plate_id"].astype(str)
        + "_"
        + df["position"].astype(str)
        + "_"
        + df["sample"].astype(str)
    )


def process_fov_row(row, images_folder, mask_folder, crop_folder, CROP_SIZE=1024):
    row_path = f"{row['if_plate_id']}/{row['basename']}"
    image_mt = cv2.imread(f"{images_folder}/{row_path}_red.png", -1)
    image_er = cv2.imread(f"{images_folder}/{row_path}_yellow.png", -1)
    image_nuc = cv2.imread(f"{images_folder}/{row_path}_blue.png", -1)
    image_pr = cv2.imread(f"{images_folder}/{row_path}_green.png", -1)
    image = np.dstack((image_mt, image_er, image_nuc, image_pr))

    mask = cv2.imread(f"{mask_folder}/{row['basename']}_cellmask.png", -1)

    crops, masks, cell_bbox_df = create_crops(
        image, mask, crop_size=CROP_SIZE, mask_cell=False
    )

    cell_bbox_df["if_plate_id"] = row["if_plate_id"]
    cell_bbox_df["position"] = row["position"]
    cell_bbox_df["sample"] = row["sample"]
    cell_bbox_df["atlas_name"] = row["atlas_name"]
    cell_bbox_df["antibody"] = row["antibody"]
    cell_bbox_df["ensembl_ids"] = row["ensembl_ids"]
    cell_bbox_df["gene_names"] = row["gene_names"]
    cell_bbox_df["locations"] = row["locations"]

    plate_crop_folder = f"{crop_folder}/{row['if_plate_id']}"
    os.makedirs(plate_crop_folder, exist_ok=True)
    for (cell_id, crop), (_, mask) in zip(crops.items(), masks.items()):
        cv2.imwrite(
            f"{plate_crop_folder}/{row['basename']}_{cell_id}_cell_image.png", crop
        )
        cv2.imwrite(
            f"{plate_crop_folder}/{row['basename']}_{cell_id}_cell_mask.png",
            mask.astype(np.uint8) * 255,
        )

    cell_bbox_df.to_csv(
        f"{plate_crop_folder}/{row['basename']}_cell_bbox.csv", index=False
    )


def get_bbox_df(row):
    mask = cv2.imread(f"{mask_folder}/{row['basename']}_cellmask.png", -1)
    regions = regionprops(mask)
    all_bbox = []
    for region in regions:
        all_bbox.append(
            {
                "cell_id": region.label,
                "x1": region.bbox[1],
                "y1": region.bbox[0],
                "x2": region.bbox[3],
                "y2": region.bbox[2],
            }
        )
    bbox_df = pd.DataFrame(all_bbox)
    row_dict = row.to_dict()
    bbox_df[[k for k in row_dict.keys() if k not in bbox_df.columns]] = [
        v for v in row_dict.values()
    ]
    return bbox_df


CROP_SIZE = 1024

images_folder = "/proj/hpa_subcell/hpa_data/HPA-IF-images"
mask_folder = "/proj/hpa_subcell/hpa_data/ankit/processed_masks"

df = pd.read_csv("annotations/IF-image-v23-filtered.csv")
add_basename(df)

crop_folder = "/proj/hpa_subcell/hpa_data/orig_crops"
shutil.rmtree(crop_folder, ignore_errors=True)
os.makedirs(crop_folder, exist_ok=True)

Parallel(n_jobs=-1)(
    delayed(process_fov_row)(row, images_folder, mask_folder, crop_folder)
    for i, row in tqdm(df.iterrows(), total=len(df))
)

dfs = sorted(glob(f"{crop_folder}/*/*_cell_bbox.csv"))
all_bbox_df = pd.concat([pd.read_csv(f) for f in tqdm(dfs)], ignore_index=True)
all_bbox_df.to_csv(f"{crop_folder}/IF-image-cell_bboxes.csv")

