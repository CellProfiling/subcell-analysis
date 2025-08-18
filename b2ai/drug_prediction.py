import argparse
import copy
import glob
import itertools
import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    label_ranking_average_precision_score,
)
from skimage.measure import regionprops
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.mlp import MLPClassifier
from utils.preprocess import safe_crop
from utils.eval_mlp import cross_val_mlp

HPA_COLORS = ["red", "yellow", "blue", "green"]


def set_random_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_single_mask_pair(
    cell_mask_path, nuc_mask_path, mask_save_folder, crop_size=1024
):
    cell_mask = cv2.imread(cell_mask_path, -1)
    nuclei_mask = cv2.imread(nuc_mask_path, -1)
    mask_name = os.path.basename(cell_mask_path).replace("cellmask.png", "")
    mask_region_props = regionprops(cell_mask)

    for i, region_prop in enumerate(mask_region_props):
        cell_label = region_prop.label
        center = (
            (region_prop.bbox[0] + region_prop.bbox[2]) // 2,
            (region_prop.bbox[1] + region_prop.bbox[3]) // 2,
        )
        crop_bbox = (
            int(center[0] - crop_size // 2),
            int(center[1] - crop_size // 2),
            int(center[0] + crop_size // 2),
            int(center[1] + crop_size // 2),
        )

        cell_bin_mask = cell_mask == cell_label
        cell_bin_mask, _ = safe_crop(cell_bin_mask, crop_bbox)
        assert (cell_bin_mask.shape[0] == crop_size) & (
            cell_bin_mask.shape[1] == crop_size
        ), "Size mismatch"
        cv2.imwrite(
            f"{mask_save_folder}/{mask_name}cell{cell_label}_crop_cell_mask.png",
            (cell_bin_mask * 255).astype(np.uint8),
        )
        nuclei_bin_mask = nuclei_mask == cell_label
        nuclei_bin_mask, _ = safe_crop(nuclei_bin_mask, crop_bbox)
        cv2.imwrite(
            f"{mask_save_folder}/{mask_name}cell{cell_label}_crop_nuc_mask.png",
            (nuclei_bin_mask * 255).astype(np.uint8),
        )


def generate_cell_masks():
    mask_folder = "/scratch/groups/emmalu/subcell_ankit/bridge2ai_segmentations"
    mask_save_folder = (
        "/scratch/groups/emmalu/subcell_ankit/bridge2ai_cell_segmentations"
    )
    os.makedirs(mask_save_folder, exist_ok=True)

    cell_masks_paths = glob.glob(f"{mask_folder}/*cellmask.png")
    nuc_masks_paths = [
        mask_path.replace("cellmask.png", "nucleimask.png")
        for mask_path in cell_masks_paths
    ]

    crop_size = 1024
    # Process masks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_mask_pair)(
            cell_path, nuc_path, mask_save_folder, crop_size
        )
        for cell_path, nuc_path in tqdm(
            zip(cell_masks_paths, nuc_masks_paths), total=len(cell_masks_paths)
        )
    )


def get_cross_val_splits(df, n_splits=10):
    if not os.path.exists("annotations/b2ai_protein_split.csv"):
        proteins = df["protein"].unique().tolist()

        random.seed(42)
        random.shuffle(proteins)

        val_split_proteins = np.array_split(proteins, n_splits)
        val_split_idxs = [
            [i] * len(val_split) for i, val_split in enumerate(val_split_proteins)
        ]

        split_df = pd.DataFrame(
            {
                "protein": list(itertools.chain.from_iterable(val_split_proteins)),
                "fold_idx": list(itertools.chain.from_iterable(val_split_idxs)),
            }
        )
        split_df.to_csv("annotations/b2ai_protein_split.csv", index=False)
    else:
        split_df = pd.read_csv("annotations/b2ai_protein_split.csv")
    return split_df


def drug_prediction(all_method_data, save_folder, num_splits):
    save_folder = f"{save_folder}/drug_prediction"
    os.makedirs(save_folder, exist_ok=True)

    parameters = {
        "n_layers": [1, 2, 3],
        "hidden_dim": [256, 512, 1024],
        "dropout": np.round(np.linspace(0.0, 0.5, 8), 1),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_result_dfs = []
    for method, data in all_method_data.items():
        print(f"Evaluating {method}", flush=True)

        if os.path.isfile(f"{save_folder}/{method}_results.csv"):
            print(f"Skipping {method} as data already exists")
            method_result_df = pd.read_csv(f"{save_folder}/{method}_results.csv")
        else:
            method_result_df = []
            for fold_idx in range(num_splits):
                val_idx = data["fold_idx"] == fold_idx
                train_x = data[~val_idx][[x for x in data.columns if "fv" in x]].values
                train_x = torch.from_numpy(train_x).float().to(device)
                val_x = data[val_idx][[x for x in data.columns if "fv" in x]].values
                val_x = torch.from_numpy(val_x).float().to(device)

                label_encoder = LabelBinarizer()
                train_y = label_encoder.fit_transform(
                    data[~val_idx]["treatment"].values
                )
                train_y = torch.from_numpy(train_y).float().to(device)
                val_y = label_encoder.transform(data[val_idx]["treatment"].values)
                val_y = torch.from_numpy(val_y).float().to(device)

                fold_results_df = cross_val_mlp(
                    train_x, train_y, val_x, val_y, parameters
                )
                fold_results_df["Fold"] = fold_idx
                fold_results_df["Method"] = method
                method_result_df.append(fold_results_df)
            method_result_df = pd.concat(method_result_df, ignore_index=True)
            method_result_df.to_csv(f"{save_folder}/{method}_results.csv", index=False)

        best_result_fold_idx = method_result_df.groupby(["Method", "Fold"])[
            "macro_ap"
        ].idxmax()
        best_result_fold = method_result_df.loc[best_result_fold_idx]
        all_result_dfs.append(best_result_fold)

        all_result_df = pd.concat(all_result_dfs, ignore_index=True)
        all_result_df.to_csv(f"{save_folder}/all_results.csv", index=False)


if __name__ == "__main__":
    # generate_cell_masks()

    save_folder = "/scratch/groups/emmalu/subcell_ankit/b2ai_analysis"

    n_splits = 10

    exp_name_dir_dict = {
        "bestfitting": "/scratch/groups/emmalu/subcell_ankit/b2ai/bestfitting/all_features.pth",
        "DINO4Cells-HPA": "/scratch/groups/emmalu/subcell_ankit/b2ai/dino/all_features.pth",
        "MAE-CellS-ProtS-Pool": "/scratch/groups/emmalu/subcell_ankit/b2ai/subcell_rybg_mae/all_features.pth",
        "ViT-ProtS-Pool": "/scratch/groups/emmalu/subcell_ankit/b2ai/subcell_rybg_vit/all_features.pth",
    }

    image_path = "/scratch/groups/emmalu/subcell_ankit/bridge2ai_crops"
    mask_path = "/scratch/groups/emmalu/subcell_ankit/bridge2ai_cell_segmentations"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_method_data = {}
    for method, result_path in exp_name_dir_dict.items():
        df, features = torch.load(result_path, map_location="cpu", weights_only=False)

        non_neg_idx = df["protein"] != "NEGATIVE"
        df = df[non_neg_idx].reset_index(drop=True)
        features = features[non_neg_idx]

        df.loc[:, [f"fv_{i}" for i in range(features.shape[1])]] = features

        split_df = get_cross_val_splits(df, n_splits)
        split_dict = dict(zip(split_df["protein"], split_df["fold_idx"]))
        df = df.assign(fold_idx=df["protein"].map(split_dict))

        all_method_data[method] = df

    drug_prediction(all_method_data, save_folder, n_splits)
