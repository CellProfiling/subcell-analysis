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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.vit_pool import ViTPoolModelOutput
from utils.dataset import OpenCellDataset, collate_fn
from utils.display_image import process_attn, save_attention_images
from utils.load_model import get_subcell_model


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="config file path")
    argparser.add_argument("-c", "--config", help="path to configuration file")

    args = argparser.parse_args()
    # args = argparser.parse_args(["-c", "configs/config_subcell_mae_opencell.yaml"])

    config_path = args.config
    with open(config_path, "r") as config_buffer:
        configs = yaml.safe_load(config_buffer)

    for config in configs:
        crops_folder = config.get("crops_folder")
        inference_folder = config.get("inference_folder")

        os.makedirs(inference_folder, exist_ok=True)
        inference_folder = f"{inference_folder}/{config['name']}_resized_{config['resize']}_norm_{config['normalization']}"
        os.makedirs(inference_folder, exist_ok=True)
        print(f"Saving inference results to {inference_folder}")

        attn_inference_folder = f"{inference_folder}/attention_images"
        shutil.rmtree(attn_inference_folder, ignore_errors=True)
        os.makedirs(attn_inference_folder, exist_ok=True)

        inference_temp_folder = f"{inference_folder}/temp"
        shutil.rmtree(inference_temp_folder, ignore_errors=True)
        os.makedirs(inference_temp_folder, exist_ok=True)

        print(
            f"Starting inference for {config['name']}_resized_{config['resize']}_norm_{config['normalization']}"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = OpenCellDataset(
            config["crops_folder"],
            config["metadata_file"],
            config["resize"],
            config["normalization"],
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
                        color_channels=["blue", "white"],
                        normalization="perchannel",
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
