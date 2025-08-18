import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from typing import List
from torchvision.utils import make_grid


def get_display_image(input, color_channels):
    color_ip = []
    for i, color in enumerate(color_channels):
        lut = np.array(
            pd.read_csv(f"utils/colormaps/{color}.lut", sep="\t", index_col="Index")
        )[None, ...].astype(np.uint8)
        ch_img = input[..., i][..., None].repeat(3, axis=-1)
        image_lut = cv2.LUT(ch_img, lut)
        color_ip.append(image_lut)
    color_composite = np.sum(color_ip, axis=0).clip(0, 255).astype(np.uint8)
    return color_composite, color_ip


def process_attn(model_output, b, h_feat, w_feat, h, w):
    attention_maps = model_output.attentions[-1][:, :, 0, 1:].reshape(
        b, -1, h_feat, w_feat
    )
    if model_output.pool_attn is not None:
        pool_attention_maps = model_output.pool_attn[:, :, 1:].reshape(
            b, -1, h_feat, w_feat
        )
        attention_maps = torch.cat([attention_maps, pool_attention_maps], dim=1)
    attention_maps = F.interpolate(
        attention_maps, size=(h, w), mode="bilinear", align_corners=False
    )

    return attention_maps


def save_attention_images(
    images,
    attention_maps,
    color_channels,
    normalization,
    metadata,
    save_folder,
    batch_idx,
):
    combined_grid = []
    for i, img in enumerate(images):
        img = img.detach().cpu().numpy()

        if normalization == "perchannel":
            img_min = np.min(img, axis=(1, 2), keepdims=True)
            img_max = np.max(img, axis=(1, 2), keepdims=True)
        elif normalization == "perimage":
            img_min = np.min(img, axis=(0, 1, 2), keepdims=True)
            img_max = np.max(img, axis=(0, 1, 2), keepdims=True)
        elif normalization == "allencell":
            min_percentiles = [5, 1, 30]
            img_min = np.array(
                [
                    (
                        np.percentile(x[x != 0], q=min_percentiles[i])
                        if len(x[x != 0]) > 0
                        else 0
                    )
                    for i, x in enumerate(img)
                ]
            ).reshape(-1, 1, 1)
            img_max = np.array(
                [
                    np.percentile(x[x != 0], q=100) if len(x[x != 0]) > 0 else 0
                    for x in img.reshape(img.shape[0], -1)
                ]
            ).reshape(-1, 1, 1)
        elif normalization == "fucci":
            img_min = np.min(img, axis=(1, 2), keepdims=True)
            img_max = np.max(img, axis=(1, 2), keepdims=True)
            img_max[[1, 2]] = np.max(img_max[[1, 2]])
        else:
            raise ValueError("Unknown normalization method")

        img_row = (
            (((img - img_min) / (img_max - img_min + 1e-8)).clip(0, 1) * 255)
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )

        img_disp, channels = get_display_image(img_row, color_channels=color_channels)
        img_disp = np.concatenate([img_disp] + channels, axis=1)

        attn_row = make_grid(
            attention_maps[i].unsqueeze(1).repeat(1, 3, 1, 1),
            normalize=True,
            nrow=attention_maps[i].shape[0],
            padding=0,
        )
        attn_row = (
            (attn_row.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
        )
        attn_row = 255 - attn_row
        combined_grid.append(np.concatenate([img_disp, attn_row], axis=1))  #

    combined_grid = np.concatenate(combined_grid, axis=0)
    cv2.imwrite(
        f"{save_folder}/batch_{batch_idx}.png",
        cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR),
    )
    metadata.to_csv(f"{save_folder}/batch_{batch_idx}_metadata.csv", index=False)
