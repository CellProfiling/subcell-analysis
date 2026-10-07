"""
Cell-DINO (ViT-S/8, Cell Painting) inference on JUMP Pilot crops.

Single forward pass with all 5 Cell Painting channels at 128x128 (native
training resolution; center-cropped from our 192px crops, no rescaling).

Preprocessing (matching dinov2/data/cell_dino/transforms.py eval transform and
notebooks/cell_dino/inference.ipynb):
    1. Select + reorder channels (see configs/cell_dino.yaml — order UNVERIFIED)
    2. x / 255, center crop to 128
    3. Per-channel z-score (self-normalize, unbiased=False, eps 1e-7)

Feature modes (--feature-mode, default from config):
    cls          last-block CLS token, 384D           -> {model_name}
    cls_avgpool  CLS + mean of patch tokens, 768D     -> {model_name}_avgpool
    both         write both from one forward pass

Output: one .pth per well at {EMBED_CELL_ROOT}/{out_name}/{plate}/{well}.pth
containing (fov_ids: list[str], cell_ids: list[str], features: Tensor[n_cells, D]).
With --apply-mask, "_masked" is appended to each out_name.

Usage:
    python scripts/inference_cell_dino.py --config configs/cell_dino.yaml --plate BR00117010
    python scripts/inference_cell_dino.py --config configs/cell_dino.yaml --plate BR00117010 \
        --feature-mode both --apply-mask
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.environ["DINOV2_ROOT"])

from dinov2.hub.cell_dino.backbones import cell_dino_cp_vits8
from utils.crop_loader import get_mask, load_crop, select_channels


FEATURE_MODES = ["cls", "cls_avgpool"]
MODE_SUFFIX = {"cls": "", "cls_avgpool": "_avgpool"}


def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Per-channel z-score. x: (B, C, H, W) float32 already in [0, 1].

    Matches Cell-DINO SelfNormalizeNoDiv (applied after Div255 + CenterCrop):
    mean/std per-sample per-channel over spatial dims, unbiased=False.
    """
    mean = torch.mean(x, dim=(-2, -1), keepdim=True)
    std = torch.std(x, dim=(-2, -1), keepdim=True, unbiased=False)
    return (x - mean) / (std + 1e-7)


# --- Main loop ---------------------------------------------------------------

@torch.no_grad()
def run_plate(
    model: torch.nn.Module,
    plate: str,
    config: dict,
    modes: list[str],
    out_names: dict[str, str],
    device: str,
    cell_batch_size: int = 256,
):
    crop_root = os.environ["CROP_ROOT"]
    out_roots = {
        m: os.path.join(os.environ["EMBED_CELL_ROOT"], out_names[m], plate) for m in modes
    }
    for d in out_roots.values():
        os.makedirs(d, exist_ok=True)

    crop_size = config["crop_size"]
    channels = config["channels"]
    apply_mask = config.get("apply_mask", False)

    wells = sorted(os.listdir(os.path.join(crop_root, plate)))

    for well in tqdm(wells, desc=f"{plate}"):
        out_paths = {m: os.path.join(out_roots[m], f"{well}.pth") for m in modes}
        if all(os.path.exists(p) for p in out_paths.values()):
            continue

        # Gather all crop paths for this well (across all FOVs)
        crop_paths = sorted(glob(os.path.join(crop_root, plate, well, "*", "crops", "*.png")))
        if not crop_paths:
            continue

        # Path: .../{plate}/{well}/{fov}/crops/crop_{X}_x_{Y}.png
        fov_ids = [p.split("/")[-3] for p in crop_paths]
        cell_ids = [os.path.basename(p).replace(".png", "") for p in crop_paths]

        all_feats = {m: [] for m in modes}
        for i in range(0, len(crop_paths), cell_batch_size):
            stacks = [load_crop(p, crop_size) for p in crop_paths[i : i + cell_batch_size]]
            batch = torch.from_numpy(np.stack([select_channels(s, channels) for s in stacks]))
            batch = preprocess_batch(batch)
            if apply_mask:
                masks = torch.from_numpy(np.stack([get_mask(s) for s in stacks]))  # (B, 1, H, W)
                batch = batch * masks

            out = model.forward_features(batch.to(device))
            cls = out["x_norm_clstoken"].cpu()                        # (B, 384)
            patch_mean = out["x_norm_patchtokens"].mean(dim=1).cpu()  # (B, 384)
            feats = {"cls": cls, "cls_avgpool": torch.cat([cls, patch_mean], dim=-1)}  # 768D
            for m in modes:
                all_feats[m].append(feats[m])

        for m in modes:
            torch.save(
                {"fov_ids": fov_ids, "cell_ids": cell_ids,
                 "features": torch.cat(all_feats[m], dim=0)},
                out_paths[m],
            )


def main(config_path: str, plate: str, feature_mode: str | None,
         apply_mask: bool, cell_batch_size: int, device: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    feature_mode = feature_mode or config.get("feature_mode", "cls")
    modes = FEATURE_MODES if feature_mode == "both" else [feature_mode]

    config["apply_mask"] = apply_mask
    mask_suffix = "_masked" if apply_mask else ""
    out_names = {m: config["model_name"] + MODE_SUFFIX[m] + mask_suffix for m in modes}

    ckpt_path = os.environ[config["ckpt_env_key"]]
    print(f"Loading Cell-DINO from {ckpt_path}")
    print(f"Channels: {config['channels']}")
    print(f"Outputs: {out_names}")
    model = cell_dino_cp_vits8(pretrained_path=ckpt_path).eval().to(device)

    print(f"Processing plate {plate} on {device}")
    run_plate(model, plate, config, modes, out_names, device, cell_batch_size)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to cell_dino config yaml")
    parser.add_argument("--plate", required=True, help="Plate barcode (e.g. BR00117010)")
    parser.add_argument("--feature-mode", default=None, choices=FEATURE_MODES + ["both"],
                        help="Override config feature_mode. 'both' writes cls and "
                             "cls_avgpool outputs from a single forward pass.")
    parser.add_argument("--cell-batch-size", type=int, default=256)
    parser.add_argument("--apply-mask", action="store_true",
                        help="Multiply normalized image channels by the cell segmentation "
                             "mask before inference. Appends '_masked' to output names.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args.config, args.plate, args.feature_mode, args.apply_mask,
         args.cell_batch_size, args.device)
