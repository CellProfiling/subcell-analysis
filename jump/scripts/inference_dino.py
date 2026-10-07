"""
DINO4Cells inference on JUMP Pilot crops.

Single forward pass with all 5 Cell Painting channels. Output is the
768D CLS token from vit_base/16. No rescaling — the checkpoint was
trained at 128x128 (pos_embed has 64+1 patches), so 128px crops are
the native resolution.

Preprocessing (matching ankit_eval/scripts/main_jump.py:load_image_dino):
    1. Select + reorder channels to [DNA, RNA, ER, AGP, Mito]
    2. Per-channel min-max to [0, 1]
    3. Per-channel z-score (self-normalize)

Output: one .pth per well at {EMBED_CELL_ROOT}/{model_name}/{plate}/{well}.pth
containing (fov_ids: list[str], cell_ids: list[str], features: Tensor[n_cells, 768]).

Usage:
    python scripts/inference_dino.py --config configs/dino.yaml --plate BR00117010
"""

import argparse
import os
import sys
import types
from glob import glob

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
DINO_ROOT = os.path.join(PROJECT_ROOT, "models", "DINO4Cells_code")

# ──────────────────────────────────────────────────────────────────────────────
# Import ordering matters here (same pattern as jump_attention_map_dino.py).
#
# 1. Import project utils FIRST. This caches `utils` and `utils.crop_loader`
#    in sys.modules, so later sys.path changes can't shadow them.
# 2. THEN stub sys.modules["utils.utils"]. DINO4Cells' archs/vision_transformer.py
#    does `from utils.utils import trunc_normal_` for weight init, but that module
#    (a) would be shadowed by our own utils/ package, and (b) imports kornia at
#    top-level which we don't need. Since we load pretrained weights anyway, we
#    stub it with torch's built-in trunc_normal_ (identical signature).
# 3. THEN insert DINO_ROOT and import DINO's ViT.
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, PROJECT_ROOT)
from utils.crop_loader import get_mask, load_crop, select_channels  # noqa: E402

_stub = types.ModuleType("utils.utils")
_stub.trunc_normal_ = torch.nn.init.trunc_normal_
sys.modules["utils.utils"] = _stub

sys.path.insert(0, DINO_ROOT)
from archs.vision_transformer import vit_base  # noqa: E402


# --- Preprocessing -----------------------------------------------------------

def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Per-channel min-max -> per-channel z-score. x: (B, C, H, W) float32.

    Matches ankit_eval pc_min_max_standardize + self_normalize exactly
    (min/max/mean/std computed per-sample per-channel over spatial dims,
    unbiased=False for std).
    """
    mn = torch.amin(x, dim=(-2, -1), keepdim=True)
    mx = torch.amax(x, dim=(-2, -1), keepdim=True)
    x = (x - mn) / (mx - mn + 1e-6)

    mean = torch.mean(x, dim=(-2, -1), keepdim=True)
    std = torch.std(x, dim=(-2, -1), keepdim=True, unbiased=False)
    x = (x - mean) / (std + 1e-7)

    return x


# --- Model loading -----------------------------------------------------------

def load_dino_model(ckpt_path: str, config: dict) -> torch.nn.Module:
    """
    Build vit_base and load the teacher backbone from a DINO checkpoint.

    The checkpoint stores teacher weights under `backbone.*` and `head.*`.
    We strip the `backbone.` prefix and discard the head (num_classes=0
    makes the model's own head an Identity). With img_size matching the
    training resolution, pos_embed shapes line up and we can load strict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("backbone."): v
        for k, v in ckpt["teacher"].items()
        if k.startswith("backbone.")
    }

    model = vit_base(
        patch_size=config["patch_size"],
        in_chans=config["in_chans"],
        num_classes=0,
        img_size=[config["img_size"]],
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# --- Main loop ---------------------------------------------------------------

@torch.no_grad()
def run_plate(
    model: torch.nn.Module,
    plate: str,
    config: dict,
    device: str,
    cell_batch_size: int = 64,
):
    crop_root = os.environ["CROP_ROOT"]
    out_root = os.path.join(os.environ["EMBED_CELL_ROOT"], config["model_name"], plate)
    os.makedirs(out_root, exist_ok=True)

    crop_size = config["crop_size"]
    channels = config["channels"]
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

        # Path: .../{plate}/{well}/{fov}/crops/crop_{X}_x_{Y}.png
        fov_ids = [p.split("/")[-3] for p in crop_paths]
        cell_ids = [os.path.basename(p).replace(".png", "") for p in crop_paths]

        all_feats = []
        for i in range(0, len(crop_paths), cell_batch_size):
            stacks = [load_crop(p, crop_size) for p in crop_paths[i : i + cell_batch_size]]
            batch = torch.from_numpy(np.stack([select_channels(s, channels) for s in stacks]))
            batch = preprocess_batch(batch)
            if apply_mask:
                # Mask LAST, after norm: normalization stats come from the full crop
                # (matching training) and masking is a clean intervention on top.
                masks = torch.from_numpy(np.stack([get_mask(s) for s in stacks]))  # (B, 1, H, W)
                batch = batch * masks

            feats = model(batch.to(device)).cpu()  # (B, 768)
            all_feats.append(feats)

        all_feats = torch.cat(all_feats, dim=0)  # (n_cells, 768)

        torch.save(
            {"fov_ids": fov_ids, "cell_ids": cell_ids, "features": all_feats},
            out_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dino config yaml")
    parser.add_argument("--plate", required=True, help="Plate barcode (e.g. BR00117010)")
    parser.add_argument("--cell-batch-size", type=int, default=64)
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

    ckpt_path = os.environ[config["ckpt_env_key"]]
    print(f"Loading {config['model_name']} from {ckpt_path}")
    model = load_dino_model(ckpt_path, config)
    model.to(args.device)

    print(f"Processing plate {args.plate} on {args.device}")
    run_plate(model, args.plate, config, args.device, args.cell_batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
