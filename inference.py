import argparse
import os
import shutil
import sys

im
port cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.dataset import HPATestDataset, collate_fn
from utils.display_image import process_attn, save_attention_images
from utils.load_model import get_bestfitting_model, get_dino_model, get_subcell_model
from utils.preprocess import (
    preprocess_input_bestfitting,
    preprocess_input_dino,
    preprocess_input_subcell,
)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="config file path")
    argparser.add_argument("-c", "--config", help="path to configuration file")

    args = argparser.parse_args()
    # args = argparser.parse_args(["-c", "configs/config_subcell_mae.yaml"])

    config_path = args.config
    with open(config_path, "r") as config_buffer:
        config = yaml.safe_load(config_buffer)

    if config["name"] == "bestfitting":
        model = get_bestfitting_model(config["model"])
        preprocess_algo = preprocess_input_bestfitting
    elif "dino" in config["name"]:
        model = get_dino_model(config["model"])
        preprocess_algo = preprocess_input_dino
    elif "subcell" in config["name"]:
        model = get_subcell_model(config["model"])
        preprocess_algo = preprocess_input_subcell
    else:
        raise ValueError(f"Model {config['name']} not supported")

    crops_folder = config.get("crops_folder")
    inference_folder = config.get("inference_folder")

    print(f"Saving inference results to {inference_folder}")

    os.makedirs(inference_folder, exist_ok=True)
    inference_folder = f"{inference_folder}/{config['name']}"
    os.makedirs(inference_folder, exist_ok=True)

    attn_inference_folder = f"{inference_folder}/attention_images"
    shutil.rmtree(attn_inference_folder, ignore_errors=True)
    os.makedirs(attn_inference_folder, exist_ok=True)

    print(
        f"Starting inference for {config['name']} with crop params {config['crop_params']}"
    )

    dataset = HPATestDataset(
        crops_folder, config["metadata_file"], config["crop_params"], preprocess_algo
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_temp_folder = f"{inference_folder}/temp"
    shutil.rmtree(inference_temp_folder, ignore_errors=True)
    os.makedirs(inference_temp_folder, exist_ok=True)

    with torch.no_grad():
        model.to(device)
        model.eval()
        for idx, (batch_images, batch_images_orig, batch_df) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            b, _, h, w = batch_images.shape
            batch_images = batch_images.to(device)
            op = model(batch_images)
            features = model(batch_images).feature_vector

            if (
                idx % 25 == 0
                and hasattr(op, "attentions")
                and config.get("save_attention", False)
            ):
                h_feat = h // model.vit_config.patch_size
                w_feat = w // model.vit_config.patch_size
                attention_maps = process_attn(op, b, h_feat, w_feat, h, w)
                save_attention_images(
                    images=batch_images_orig,
                    attention_maps=attention_maps,
                    color_channels=["red", "yellow", "blue", "green"],
                    normalization="perchannel",
                    metadata=batch_df,
                    save_folder=attn_inference_folder,
                    batch_idx=idx,
                )

            batch_df.to_csv(f"{inference_temp_folder}/{idx}_metadata.csv", index=False)
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
