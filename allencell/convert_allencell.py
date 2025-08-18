import os
import cv2
import numpy as np
import tifffile
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def save_resized_crop(row, output_folder, max_proj, res_max_proj):
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint16)
    # Getting matching center coordinates of the resized image and the cropped image
    center_y = min(crop_size_hf, int(len(max_proj) / 2))
    center_x = min(crop_size_hf, int(len(max_proj[0]) / 2))
    # Copying the sliced crop from the resized image
    crop[
        (crop_size_hf - center_y) : (crop_size_hf + center_y),
        (crop_size_hf - center_x) : (crop_size_hf + center_x),
    ] = res_max_proj[
        (int(len(res_max_proj) / 2) - center_y) : (
            int(len(res_max_proj) / 2) + center_y
        ),
        (int(len(res_max_proj[0]) / 2) - center_x) : (
            int(len(res_max_proj[0]) / 2) + center_x
        ),
    ]

    cv2.imwrite(
        os.path.join(
            output_folder,
            row["crop_raw"]
            .replace(".ome.tif", "_resized_proj.png")
            .replace("crop_raw/", "crop_subcell/"),
        ),
        crop,
    )


def process_row(row, output_folder):
    """Process a single row from the metadata dataframe"""
    crop = tifffile.imread(os.path.join(input_folder, row["crop_raw"]))
    segmentation = tifffile.imread(os.path.join(input_folder, row["crop_seg"]))

    name_dict = eval(row["name_dict"])

    channel_dict = {
        "dna": "dna_segmentation",
        "membrane": "membrane_segmentation_roof",
        "structure": "struct_segmentation_roof",
    }

    raw_dict = {
        x: name_dict["crop_raw"].index(x) for i, x in enumerate(channel_dict.keys())
    }
    seg_dict = {k: name_dict["crop_seg"].index(v) for k, v in channel_dict.items()}

    max_proj = np.uint16(
        np.max(
            crop[:, list(raw_dict.values())]
            * (segmentation[:, list(seg_dict.values())] / 255),
            axis=0,
        ).transpose(1, 2, 0)
    )

    cv2.imwrite(
        os.path.join(
            output_folder, row["crop_raw"].replace(".ome.tif", "_proj.png")
        ).replace("crop_raw/", "crop_subcell/"),
        max_proj,
    )

    res_max_proj = cv2.resize(
        max_proj,
        None,
        fx=pixel_micron_AllenCell / pixel_micron_HPA,
        fy=pixel_micron_AllenCell / pixel_micron_HPA,
    )
    save_resized_crop(row, output_folder, max_proj, res_max_proj)


input_folder = "/scratch/groups/emmalu/AllenCell/loaddata/"
output_folder = "/scratch/groups/emmalu/AllenCell/loaddata"
metadata_file = "/scratch/groups/emmalu/AllenCell/manifest.csv"

pixel_micron_HPA = 0.0800885
pixel_micron_AllenCell = 0.10833

crop_size = 640
crop_size_hf = int(crop_size / 2)


os.makedirs(output_folder, exist_ok=True)
mf = pd.read_csv(metadata_file)


n_jobs = -1  
Parallel(n_jobs=n_jobs, verbose=1)(
    delayed(process_row)(row, output_folder)
    for _, row in tqdm(mf.iterrows(), total=len(mf))
)
