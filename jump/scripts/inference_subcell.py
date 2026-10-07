"""
SubCell inference on JUMP Pilot crops.

Strategy: the ybg (3-channel) SubCell model is run THREE TIMES per cell,
with ER and DNA in the y/b slots each time and one of {Mito, AGP, RNA}
in the g (protein) slot. All three passes are batched into a single forward
pass as (3, 3, H, W) -> (3, 1536) -> reshaped to (1, 4608).

For efficiency, multiple cells are processed together: with cell_batch_size=B,
the actual forward is (B*3, 3, H, W) -> (B*3, 1536) -> (B, 4608).

Output: one .pth per well at {EMBED_CELL_ROOT}/{model_name}/{plate}/{well}.pth
containing (fov_ids: list[str], cell_ids: list[str], features: Tensor[n_cells, 4608]).

Usage:
    python scripts/inference_subcell.py --config configs/subcell_mae.yaml --plate BR00117010
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Add project root and SubCellPortable to path
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "SubCellPortable"))

from vit_model import ViTPoolClassifier
from utils.crop_loader import load_crop, select_channels, get_mask


def build_subcell_input(
    crop: np.ndarray,
    context_channels: list[str],
    protein_channels: list[str],
) -> torch.Tensor:
    """
    Build the 3-pass input batch for one cell.

    Args:
        crop: (9, H, W) float32 array from load_crop.
        context_channels: e.g. ["ER", "DNA"] — fixed channels for all passes.
        protein_channels: e.g. ["Mito", "AGP", "RNA"] — one per pass.

    Returns:
        (n_passes, 3, H, W) tensor. Each pass = [context[0], context[1], protein[i]].
    """
    passes = []
    for protein in protein_channels:
        channels = context_channels + [protein]  # [ER, DNA, protein]
        passes.append(select_channels(crop, channels))
    return torch.from_numpy(np.stack(passes, axis=0))


def per_channel_minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each channel independently. x: (..., C, H, W)."""
    min_val = torch.amin(x, dim=(-2, -1), keepdim=True)
    max_val = torch.amax(x, dim=(-2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-6)


def global_minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize across all channels jointly. x: (B, C, H, W)."""
    min_val = torch.amin(x, dim=(-3, -2, -1), keepdim=True)
    max_val = torch.amax(x, dim=(-3, -2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-6)


def preprocess_batch(
    batch: torch.Tensor,
    rescale_ratio: float,
    normalize: str,
    masks: torch.Tensor = None,
) -> torch.Tensor:
    """
    Rescale, normalize, and optionally mask a batch of images.

    Masking is the LAST step: normalization stats are computed on the full
    (unmasked) crop so cell pixels keep their natural values relative to the
    surrounding FOV context, matching what the model saw during training. The
    mask then zeroes background as a clean intervention on the normalized image.

    Args:
        batch: (B, C, H, W) float32 tensor in [0, 1].
        rescale_ratio: spatial upscale factor.
        normalize: "per_chan" | "all_chan" | "none".
        masks: (B, 1, H, W) binary mask, or None. If given, rescaled with
               nearest-neighbor (keeps binary) and multiplied in last.

    Returns:
        (B, C, H', W') tensor where H' = int(H * rescale_ratio).
    """
    new_size = int(batch.shape[-1] * rescale_ratio)
    batch = F.interpolate(batch, size=new_size, mode="bilinear", align_corners=False)

    if normalize == "per_chan":
        batch = per_channel_minmax(batch)
    elif normalize == "all_chan":
        batch = global_minmax(batch)
    # else "none": leave in [0, 1]

    if masks is not None:
        masks = F.interpolate(masks, size=new_size, mode="nearest")
        batch = batch * masks

    return batch


@torch.no_grad()
def run_plate(
    model: ViTPoolClassifier,
    plate: str,
    config: dict,
    device: str,
    cell_batch_size: int = 16,
):
    crop_root = os.environ["CROP_ROOT"]
    out_root = os.path.join(os.environ["EMBED_CELL_ROOT"], config["model_name"], plate)
    os.makedirs(out_root, exist_ok=True)

    crop_size = config["crop_size"]
    rescale_ratio = config["rescale_ratio"]
    normalize = config["normalize"]
    context_channels = config["context_channels"]
    protein_channels = config["protein_channels"]
    n_passes = len(protein_channels)
    apply_mask = config.get("apply_mask", False)

    wells = sorted(os.listdir(os.path.join(crop_root, plate)))

    for well in tqdm(wells, desc=f"{plate}"):
        out_path = os.path.join(out_root, f"{well}.pth")
        if os.path.exists(out_path):
            continue

        # Gather all crop paths for this well (across all FOVs)
        crop_paths = sorted(glob(os.path.join(crop_root, plate, well, "*", "crops", "*.png")))
        if not crop_paths:
            continue

        # Parse FOV and cell_id from each path
        # Path: .../{plate}/{well}/{fov}/crops/crop_{X}_x_{Y}.png
        fov_ids = [p.split("/")[-3] for p in crop_paths]
        cell_ids = [os.path.basename(p).replace(".png", "") for p in crop_paths]

        # Process in batches
        all_feats = []
        for i in range(0, len(crop_paths), cell_batch_size):
            batch_paths = crop_paths[i : i + cell_batch_size]

            # Load and build 3-pass inputs: (B, n_passes, 3, H, W)
            crops = [load_crop(p, crop_size=crop_size) for p in batch_paths]
            inputs = torch.stack([
                build_subcell_input(c, context_channels, protein_channels)
                for c in crops
            ])
            b = inputs.shape[0]

            # Extract masks (B, 1, H, W) before flattening, then tile along pass
            # dim so each of the n_passes copies of a cell gets the same mask.
            masks = None
            if apply_mask:
                masks = torch.from_numpy(np.stack([get_mask(c) for c in crops]))
                masks = masks.repeat_interleave(n_passes, dim=0)  # (B*n_passes, 1, H, W)

            # Flatten cell and pass dims: (B * n_passes, 3, H, W)
            inputs = inputs.reshape(b * n_passes, 3, crop_size, crop_size)

            # Preprocess: rescale + normalize, then mask last
            inputs = preprocess_batch(inputs, rescale_ratio, normalize, masks)

            # Forward
            output = model(inputs.to(device))
            feats = output.pool_op.cpu()  # (B * n_passes, 1536)

            # Reshape: (B, n_passes * 1536) = (B, 4608)
            feats = feats.reshape(b, -1)
            all_feats.append(feats)

        all_feats = torch.cat(all_feats, dim=0)  # (n_cells, 4608)

        torch.save(
            {"fov_ids": fov_ids, "cell_ids": cell_ids, "features": all_feats},
            out_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to subcell config yaml")
    parser.add_argument("--plate", required=True, help="Plate barcode (e.g. BR00117010)")
    parser.add_argument("--cell-batch-size", type=int, default=16,
                        help="Number of cells per forward pass (actual batch = this * 3)")
    parser.add_argument("--apply-mask", action="store_true",
                        help="Multiply image channels by the cell segmentation mask before "
                             "inference. Appends '_masked' to model_name so outputs land in "
                             "a separate directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.apply_mask:
        config["apply_mask"] = True
        config["model_name"] = config["model_name"] + "_masked"

    encoder_path = os.environ[config["encoder_env_key"]]
    classifier_path = os.environ[config["classifier_env_key"]]

    print(f"Loading {config['model_name']} from {encoder_path}")
    model = ViTPoolClassifier(config["model_config"])
    model.load_model_dict(encoder_path, classifier_path)
    model.to(args.device)
    model.eval()

    print(f"Processing plate {args.plate} on {args.device}")
    run_plate(model, args.plate, config, args.device, args.cell_batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
