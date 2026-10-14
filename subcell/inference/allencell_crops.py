"""Allen Cell hiPSC crops at the HPA pixel size, the input of ``subcell.inference.allencell``.

For every cell of the Allen Cell manifest the DNA, membrane and structure channels are masked with their
segmentations and max-projected along z, written as ``<name>_proj.png``; the projection is then rescaled from the
Allen Cell pixel size (0.10833 um) to the HPA pixel size (0.0800885 um) and centred in a 640 x 640 px canvas,
written as ``<name>_resized_proj.png`` (the file the Allen Cell Dataset reads).

    python -m subcell.inference.allencell_crops [--jobs N]

Inputs (``ALLENCELL_DATA``): ``manifest.csv`` and ``loaddata/{crop_raw,crop_seg}``.
Output: ``config.ALLENCELL_DIR/crop_subcell``, the ``crops_folder`` of the configs ``allencell_subcell_*.yaml``.
"""
import argparse
import ast
import os

import cv2
import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from tqdm import tqdm

from subcell import config
from subcell.io.datasets import pixel_micron_HPA

PIXEL_MICRON_ALLENCELL = 0.10833
CROP_SIZE = 640                         # side of the final canvas, in pixels
CHANNEL_SEGMENTATION = {"dna": "dna_segmentation", "membrane": "membrane_segmentation_roof",
                        "structure": "struct_segmentation_roof"}


def save_resized_crop(row, output_folder, max_proj, res_max_proj):
    """Centre the rescaled projection in a CROP_SIZE canvas (cropping it if larger) and write it."""
    half = CROP_SIZE // 2
    crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint16)
    # half-extents of the copied region, taken from the projection before rescaling
    center_y = min(half, len(max_proj) // 2)
    center_x = min(half, len(max_proj[0]) // 2)
    crop[(half - center_y):(half + center_y), (half - center_x):(half + center_x)] = res_max_proj[
        (len(res_max_proj) // 2 - center_y):(len(res_max_proj) // 2 + center_y),
        (len(res_max_proj[0]) // 2 - center_x):(len(res_max_proj[0]) // 2 + center_x),
    ]
    name = row["crop_raw"].replace(".ome.tif", "_resized_proj.png").replace("crop_raw/", "crop_subcell/")
    cv2.imwrite(os.path.join(output_folder, name), crop)


def process_row(row, input_folder, output_folder):
    """Project and rescale one cell of the manifest."""
    crop = tifffile.imread(os.path.join(input_folder, row["crop_raw"]))
    segmentation = tifffile.imread(os.path.join(input_folder, row["crop_seg"]))
    name_dict = ast.literal_eval(row["name_dict"])

    raw_idx = [name_dict["crop_raw"].index(channel) for channel in CHANNEL_SEGMENTATION]
    seg_idx = [name_dict["crop_seg"].index(seg) for seg in CHANNEL_SEGMENTATION.values()]
    max_proj = np.uint16(np.max(crop[:, raw_idx] * (segmentation[:, seg_idx] / 255), axis=0).transpose(1, 2, 0))
    name = row["crop_raw"].replace(".ome.tif", "_proj.png").replace("crop_raw/", "crop_subcell/")
    cv2.imwrite(os.path.join(output_folder, name), max_proj)

    scale = PIXEL_MICRON_ALLENCELL / pixel_micron_HPA
    res_max_proj = cv2.resize(max_proj, None, fx=scale, fy=scale)
    save_resized_crop(row, output_folder, max_proj, res_max_proj)


def main(jobs=1):
    """Write the projections of every cell of the manifest to ``config.ALLENCELL_DIR/crop_subcell``."""
    data_root = config.ROOTS["ALLENCELL_DATA"]
    input_folder = data_root / "loaddata"
    output_folder = config.ensure(config.ALLENCELL_DIR)
    config.ensure(output_folder / "crop_subcell")
    manifest = pd.read_csv(data_root / "manifest.csv")
    Parallel(n_jobs=jobs, verbose=1)(
        delayed(process_row)(row, input_folder, output_folder) for _, row in tqdm(manifest.iterrows(), total=len(manifest))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers")
    main(parser.parse_args().jobs)
