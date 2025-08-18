import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from harmony import harmonize
from joblib import Parallel, delayed
from skimage.measure import regionprops
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.vit_pool import ViTPoolModelOutput
from utils.dataset import FUCCIDataset, collate_fn
from utils.display_image import process_attn, save_attention_images
from utils.load_model import get_dino_model, get_subcell_model
from utils.preprocess import (
    preprocess_input_bestfitting,
    preprocess_input_dino,
    preprocess_input_subcell,
    safe_crop,
)
from utils.train_mlp import eval_model, get_metrics, train_mlp


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def process_fov_row(row, df, crops_folder, mask_folder, save_folder, CROP_SIZE):
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
                cell_center[0] - CROP_SIZE[0] // 2,
                cell_center[1] - CROP_SIZE[1] // 2,
                cell_center[0] + CROP_SIZE[0] // 2,
                cell_center[1] + CROP_SIZE[1] // 2,
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


def make_single_cell_crops():
    CROP_SIZE = (512, 512)
    df = pd.read_csv(
        "/scratch/groups/emmalu/cell_cycle_callisto/single_cell_statistics.csv"
    )
    crops_folder = "/scratch/groups/emmalu/cell_cycle_callisto/Files"
    mask_folder = "/scratch/groups/emmalu/cell_cycle_callisto/cell_masks"
    save_folder = "/scratch/groups/emmalu/cell_cycle_callisto/single_cell_crops"
    shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    df["fov_id"] = df["cell_id"].apply(lambda x: x.rsplit("_", 1)[0])
    fov_df = (
        df.groupby(["ab_id", "fov_id"])
        .first()
        .reset_index()[["ab_id", "fov_id", "cell_id"]]
    )

    n_jobs = 16
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_fov_row)(
            row, df, crops_folder, mask_folder, save_folder, CROP_SIZE
        )
        for i, row in tqdm(fov_df.iterrows(), total=len(fov_df), desc="Processing FOVs")
    )

    success_count = sum(1 for result in results if result.startswith("Success"))
    print(
        f"Processed {len(results)} FOVs: {success_count} succeeded, {len(results) - success_count} failed"
    )


if __name__ == "__main__":
    # make_single_cell_crops()

    argparser = argparse.ArgumentParser(description="config file path")
    argparser.add_argument("-c", "--config", help="path to configuration file")

    args = argparser.parse_args()
    # args = argparser.parse_args(["-c", "configs/config_subcell_mae_fucci.yaml"])

    config_path = args.config
    with open(config_path, "r") as config_buffer:
        configs = yaml.safe_load(config_buffer)

    for config in configs:
        crops_folder = config.get("crops_folder")
        inference_folder = config.get("inference_folder")
        os.makedirs(inference_folder, exist_ok=True)

        inference_folder = f"{inference_folder}/{config['name']}_channel_comb_{config['channel_combination']}_resize_{config['resize']}"
        os.makedirs(inference_folder, exist_ok=True)
        print(f"Saving inference results to {inference_folder}")

        attn_inference_folder = f"{inference_folder}/attention_images"
        shutil.rmtree(attn_inference_folder, ignore_errors=True)
        os.makedirs(attn_inference_folder, exist_ok=True)

        inference_temp_folder = f"{inference_folder}/temp"
        shutil.rmtree(inference_temp_folder, ignore_errors=True)
        os.makedirs(inference_temp_folder, exist_ok=True)

        print(f"Starting inference for {config['name']}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = FUCCIDataset(
            config["crops_folder"],
            config["metadata_file"],
            config["channel_combination"],
            config["resize"],
        )

        batch_size = 16
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
        )

        model = get_subcell_model(config["model"])

        with torch.no_grad():
            model.to(device)
            model.eval()
            for idx, (batch_images, batch_images_orig, batch_df) in tqdm(
                enumerate(dataloader), total=len(dataloader)
            ):
                batch_images = batch_images.to(device)

                if len(batch_images.shape) == 5:
                    b, _, _, h, w = batch_images.shape
                    all_features = []
                    all_attentions = []
                    all_pool_attentions = []
                    for i in range(batch_images.shape[1]):
                        batch_op = model(batch_images[:, i, :, :, :])
                        batch_features = batch_op.feature_vector
                        all_features.append(batch_features)
                        all_attentions.append(batch_op.attentions[-1])
                        all_pool_attentions.append(batch_op.pool_attn)
                    features = torch.cat(all_features, dim=1)
                    all_attentions = (torch.mean(torch.stack(all_attentions), dim=0),)
                    all_pool_attentions = torch.mean(
                        torch.stack(all_pool_attentions), dim=0
                    )
                    op = ViTPoolModelOutput(
                        attentions=all_attentions, pool_attn=all_pool_attentions
                    )
                else:
                    b, _, h, w = batch_images.shape
                    op = model(batch_images)
                    features = op.feature_vector

                if idx % 25 == 0 and config.get("save_attention", False):
                    h_feat = h // model.vit_config.patch_size
                    w_feat = w // model.vit_config.patch_size
                    attention_maps = process_attn(op, b, h_feat, w_feat, h, w)

                    save_attention_images(
                        images=batch_images_orig,
                        attention_maps=attention_maps,
                        color_channels=["red", "light_blue", "orange", "green"],
                        normalization="fucci",
                        metadata=batch_df,
                        save_folder=attn_inference_folder,
                        batch_idx=idx,
                    )

                batch_df.to_csv(
                    f"{inference_temp_folder}/{idx}_metadata.csv", index=False
                )
                torch.save(features, f"{inference_temp_folder}/{idx}_features.pt")

        for i in tqdm(range(len(dataloader))):
            metadata = pd.read_csv(f"{inference_temp_folder}/{i}_metadata.csv")
            features = torch.load(
                f"{inference_temp_folder}/{i}_features.pt",
                map_location="cpu",
                weights_only=False,
            )
            if i == 0:
                all_features = features
                all_metadata = metadata
            else:
                all_features = torch.cat([all_features, features], dim=0)
                all_metadata = pd.concat([all_metadata, metadata], ignore_index=True)

        torch.save((all_metadata, all_features), f"{inference_folder}/all_features.pth")
        shutil.rmtree(inference_temp_folder, ignore_errors=True)
