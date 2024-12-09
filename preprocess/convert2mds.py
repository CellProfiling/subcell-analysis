import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Tuple

import cv2

import numpy as np
import pandas as pd
from scipy.ndimage import grey_dilation
from streaming.base.util import merge_index
from streaming import MDSWriter
from tqdm import tqdm


def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f"\nInitialize Worker PID: {pid}", flush=True, end="")


def add_basename(df):
    df["basename"] = (
        # df["atlas_name"]
        # + "_" +
        df["if_plate_id"].astype(str)
        + "_"
        + df["position"].astype(str)
        + "_"
        + df["sample"].astype(str)
    )
    return df


def add_loc_labels(df, unique_locations):
    df.loc[df["locations"].isna(), "locations"] = "Negative"
    locations = df["locations"].str.split(",").tolist()
    one_hot = np.array(
        [
            [1 if loc in location else 0 for loc in unique_locations]
            for location in locations
        ]
    )
    df[unique_locations] = one_hot
    df = df[df[unique_locations].sum(axis=1) > 0].reset_index(drop=True)
    return df


def safe_crop(image, bbox):
    x1, y1, x2, y2 = bbox
    img_w, img_h = image.shape[:2]
    is_single_channel = len(image.shape) == 2
    if x1 < 0:
        pad_x1 = 0 - x1
        new_x1 = 0
    else:
        pad_x1 = 0
        new_x1 = x1
    if y1 < 0:
        pad_y1 = 0 - y1
        new_y1 = 0
    else:
        pad_y1 = 0
        new_y1 = y1
    if x2 > img_w - 1:
        pad_x2 = x2 - (img_w - 1)
        new_x2 = img_w - 1
    else:
        pad_x2 = 0
        new_x2 = x2
    if y2 > img_h - 1:
        pad_y2 = y2 - (img_h - 1)
        new_y2 = img_h - 1
    else:
        pad_y2 = 0
        new_y2 = y2

    patch = image[new_x1:new_x2, new_y1:new_y2]
    patch = (
        np.pad(
            patch,
            ((pad_x1, pad_x2), (pad_y1, pad_y2)),
            mode="constant",
            constant_values=0,
        )
        if is_single_channel
        else np.pad(
            patch,
            ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    )
    return patch


def dilate_mask(mask, kernel_size=7):
    mask = (mask * 255).astype(np.uint8)
    mask = grey_dilation(mask, size=(kernel_size, kernel_size))
    mask = mask / 255
    return mask


def get_cell_bbox_within_crop(crop_size, w, h):
    center = crop_size[0] // 2, crop_size[1] // 2
    x1 = center[0] - w // 2
    y1 = center[1] - h // 2
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def get_antibody_crops(df, bbox_df, image_path, mask_path, crop_size, cell_size):
    antibody_cell_images = []
    antibody_cell_masks = []
    save_df_antibody = pd.DataFrame()

    for image_idx, row in df.iterrows():
        plate_id = row["if_plate_id"]
        position = row["position"]
        sample = row["sample"]
        name_str = str(plate_id) + "_" + str(position) + "_" + str(sample)

        image_mt = cv2.imread(f"{image_path}/{plate_id}/{name_str}_red.png", -1)
        image_er = cv2.imread(f"{image_path}/{plate_id}/{name_str}_yellow.png", -1)
        image_nuc = cv2.imread(f"{image_path}/{plate_id}/{name_str}_blue.png", -1)
        image_pr = cv2.imread(f"{image_path}/{plate_id}/{name_str}_green.png", -1)

        image = np.dstack((image_mt, image_er, image_nuc, image_pr))
        image_mask = cv2.imread(f"{mask_path}/{name_str}_cellmask.png", -1)

        bbox_df_image = bbox_df.iloc[
            np.where(
                (bbox_df["if_plate_id"] == row["if_plate_id"])
                & (bbox_df["position"] == row["position"])
                & (bbox_df["sample"] == row["sample"])
            )
        ]
        bbox_df_image = bbox_df_image.assign(crop_x1=0, crop_y1=0, crop_x2=0, crop_y2=0)
        for bbox_idx, bbox_row in bbox_df_image.iterrows():
            x1 = int(bbox_row["top"] + (bbox_row["width"] / 2) - (crop_size[0] / 2))
            y1 = int(bbox_row["left"] + (bbox_row["height"] / 2) - (crop_size[1] / 2))
            x2 = int(x1 + crop_size[0])
            y2 = int(y1 + crop_size[1])
            bbox_label = bbox_row["cell_id"]

            cell_image = safe_crop(image, [x1, y1, x2, y2])

            cell_mask = safe_crop(image_mask, [x1, y1, x2, y2])
            cell_mask = cell_mask == bbox_label
            cell_mask = dilate_mask(cell_mask, kernel_size=7)
            cell_mask = cell_mask.astype(cell_image.dtype)

            if cell_size:
                cell_image = cv2.resize(cell_image, cell_size, cv2.INTER_LINEAR)
                cell_mask = cv2.resize(cell_mask, cell_size, cv2.INTER_NEAREST)

            antibody_cell_images.append(cell_image)
            antibody_cell_masks.append(cell_mask)

            crop_bbox = get_cell_bbox_within_crop(
                crop_size, bbox_row["width"], bbox_row["height"]
            )
            bbox_df_image.loc[bbox_idx, "crop_x1"] = crop_bbox[0]
            bbox_df_image.loc[bbox_idx, "crop_y1"] = crop_bbox[1]
            bbox_df_image.loc[bbox_idx, "crop_x2"] = crop_bbox[2]
            bbox_df_image.loc[bbox_idx, "crop_y2"] = crop_bbox[3]

        save_df_antibody = pd.concat([save_df_antibody, bbox_df_image]).reset_index(
            drop=True
        )

    antibody_cell_images = np.array(antibody_cell_images)
    antibody_cell_masks = np.array(antibody_cell_masks)

    antibody_images = np.concatenate(
        [antibody_cell_images, antibody_cell_masks[..., np.newaxis]], axis=-1
    )

    save_df_antibody = save_df_antibody.rename(
        columns={"top": "x1", "left": "y1", "width": "w", "height": "h"}
    )
    return antibody_images, save_df_antibody


def get_split_idx_df(df, n_splits=4):
    df = add_basename(df)
    index_per_category = (
        df.groupby("basename")
        .apply(lambda x: x.index.tolist(), include_groups=False)
        .to_dict()
    )
    n_samples_per_split = [len(df) // n_splits] * n_splits
    n_samples_per_split[-1] += len(df) % n_splits
    all_split_idx = []
    for n_sample_split in n_samples_per_split:
        split_idx = []
        while len(split_idx) < n_sample_split:
            for category, indexlist in index_per_category.items():
                if len(indexlist) > 0:
                    random.shuffle(indexlist)
                    split_idx.append(indexlist.pop())
                if len(split_idx) == n_sample_split:
                    break
        all_split_idx.append(split_idx)
    assert sum(len(split_idx) for split_idx in all_split_idx) == len(
        df
    ), "Error in split index generation"
    assert set(sum(all_split_idx, [])) == set(
        df.index
    ), "Error in split index generation"
    if min(len(split_idx) for split_idx in all_split_idx) < 8:
        print(f"Spilts smaller than 8 samples for {df['antibody'].iloc[0]}")

    return all_split_idx


def get_mds_sample(unique_id, bbox_df_antibody, antibody_crops):
    cell_lines = bbox_df_antibody["atlas_name"].values
    cell_lines_str = ";".join(cell_lines)

    plate_positions = (
        bbox_df_antibody["if_plate_id"].astype(str)
        + "_"
        + bbox_df_antibody["position"].astype(str)
        + "_"
        + bbox_df_antibody["sample"].astype(str)
    ).values
    plate_positions_str = ";".join(plate_positions)

    ensembl_ids = bbox_df_antibody["ensembl_ids"].values
    ensembl_ids_str = ";".join(ensembl_ids)

    locations = bbox_df_antibody["locations"].values
    locations_str = ";".join(locations)

    targets = bbox_df_antibody[unique_locations].values
    bboxes = bbox_df_antibody[["crop_x1", "crop_y1", "crop_x2", "crop_y2"]].values

    sample = {
        "antibody": unique_id,
        "cell_line": cell_lines_str,
        "plate_position": plate_positions_str,
        "ensembl_ids": ensembl_ids_str,
        "locations": locations_str,
        "img": antibody_crops.astype(np.uint16),
        "targets": targets.astype(np.uint8),
        "bboxes": bboxes.astype(np.uint16),
    }

    return sample


def get_samples(unique_id):
    df_antibody = df[df["antibody"] == unique_id].sort_values(
        by=["if_plate_id", "position", "sample"]
    )
    bbox_df_antibody = bbox_df[bbox_df["antibody"] == unique_id].sort_values(
        by=["if_plate_id", "position", "sample", "cell_id"]
    )

    if "mult" in df_antibody.columns:
        n_mult = df_antibody.loc[df_antibody.index[0], "mult"]
        assert np.equal(
            df_antibody["mult"].values, n_mult
        ).all(), (
            f"Error in mult value for {unique_id}, got {df_antibody['mult'].values}"
        )
    else:
        n_mult = 1

    antibody_crops, bbox_df_antibody = get_antibody_crops(
        df_antibody, bbox_df_antibody, image_path, mask_path, crop_size, cell_size
    )

    assert len(antibody_crops) == len(
        bbox_df_antibody
    ), f"Error in crop generation, {unique_id}, got {len(antibody_crops)} crops and {len(bbox_df_antibody)} bboxes"

    all_split_idx = get_split_idx_df(bbox_df_antibody, 4)

    all_samples = []
    for split_idx in all_split_idx:
        split_df = bbox_df_antibody.loc[split_idx]
        split_crops = antibody_crops[split_idx]
        sample = get_mds_sample(unique_id, split_df, split_crops)
        all_samples.append(sample)

    mult_all_samples = []
    for _ in range(n_mult):
        mult_all_samples.extend(all_samples)

    return mult_all_samples


def each_task(out_root: str, len_groups: int) -> Iterator[Tuple[str, int, int]]:
    total_samples = len(unique_antibodies)
    num_groups = (total_samples // len_groups) + 1

    for data_group in range(num_groups):
        sub_out_root = os.path.join(out_root, str(data_group))
        start_sample_idx = data_group * len_groups
        end_sample_idx = min(start_sample_idx + len_groups - 1, total_samples - 1)
        yield sub_out_root, start_sample_idx, end_sample_idx


def convert_to_mds(args: Iterator[Tuple[str, int, int]]) -> None:
    sub_out_root, start_sample_idx, end_sample_idx = args

    def get_data(start: int, end: int):
        for i in range(start, end + 1):
            yield get_samples(unique_antibodies[i])

    with MDSWriter(
        out=sub_out_root,
        columns=columns,
        compression=compression,
        hashes=hashes,
        progress_bar=True,
        exist_ok=True,
        size_limit="3GB",
    ) as out:
        for sample in get_data(start_sample_idx, end_sample_idx):
            for sample_part in sample:
                out.write(sample_part)


def main():
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)

    arg_tuples = each_task(out_path, len_groups=len_groups)

    with Pool(initializer=init_worker, processes=num_process) as pool:
        for count in pool.imap(convert_to_mds, arg_tuples):
            pass


if __name__ == "__main__":
    image_path = "/proj/hpa_subcell/hpa_data/HPA-IF-images"
    mask_path = "/proj/hpa_subcell/hpa_data/ankit/processed_masks"

    out_path = "/proj/hpa_subcell/hpa_data/antibody_mds_data" 

    crop_size = (896, 896)  # (512, 512)
    cell_size = (448, 448)  # (256, 256)

    len_groups = 200
    num_process = 64

    compression = "zstd"
    hashes = ("sha1", "xxh64")

    LOCATION_MAP = pd.read_csv("annotations/location_group_mapping.csv")
    unique_locations = LOCATION_MAP["Original annotation"].tolist()
    unique_locations.append("Negative")

    columns = {
        "antibody": "str",
        "cell_line": "str",
        "plate_position": "str",
        "ensembl_ids": "str",
        "locations": "str",
        "img": "ndarray:uint16",
        "targets": "ndarray:uint8",
        "bboxes": "ndarray:uint16",
    }

    tag = "train-balanced"
    df = pd.read_csv(
        f"annotations/splits/IF-image-v23-{tag}.csv",
        low_memory=False,
        index_col=0,
    )
    df = add_loc_labels(df, unique_locations)

    bbox_df = pd.read_csv(
        f"annotations/splits/IF-image-orig-bboxes-{tag}.csv",
        low_memory=False,
        index_col=0,
    )
    bbox_df = add_loc_labels(bbox_df, unique_locations)
    bbox_df = add_basename(bbox_df)

    unique_antibodies = df["antibody"].unique().tolist()

    print(
        f"Saving {tag} samples to {out_path}. Found {np.sum(df['locations'].isna())} samples with no location label"
    )
    out_path = f"{out_path}/{tag}"

    main()
    merge_index(out_path, keep_local=True)
