"""
Generate attention-map figures for a handful of randomly chosen JUMP cells
(one per positive-control compound + one DMSO) without re-running full inference.

Layout: rows = compounds, cols = [ RGB | 12 ViT heads | 2 pool heads ]

Because SubCell runs THREE passes per cell (protein channel ∈ {Mito, AGP, RNA}),
each cell produces three independent attention maps — so this script emits one
figure per protein channel.

Run from the repo root (after `set -a; source .env; set +a`):
    python scripts/jump_attention_map.py --cell-type A549 --model mae --seed 0
    python scripts/jump_attention_map.py --cell-type U2OS --model vit --seed 3
    python scripts/jump_attention_map.py --cell-type A549 --model mae --protein Mito
"""
import argparse
import os
import random
import sys
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "SubCellPortable"))

from vit_model import ViTPoolClassifier
from utils.crop_loader import load_crop, select_channels
from utils.metadata import load_plate_info, load_well_compound_map


# ──────────────────────────────────────────────────────────────────────────────
# Standard Cell Painting channel → pseudocolor LUT mapping
# ──────────────────────────────────────────────────────────────────────────────
CHANNEL_COLORS = {
    "DNA":  "blue",
    "RNA":  "yellow",
    "ER":   "green",
    "AGP":  "orange",
    "Mito": "red",
}


# ──────────────────────────────────────────────────────────────────────────────
# Model configs — mirror configs/subcell_{mae,vit}.yaml
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "mae": "configs/subcell_mae.yaml",
    "vit": "configs/subcell_vit.yaml",
}


# ──────────────────────────────────────────────────────────────────────────────
def load_model(config: dict, device: torch.device):
    """Load ViTPoolClassifier using env-var checkpoint paths from the config."""
    encoder_path = os.environ[config["encoder_env_key"]]
    classifier_path = os.environ[config["classifier_env_key"]]
    model = ViTPoolClassifier(config["model_config"])
    model.load_model_dict(encoder_path, classifier_path)
    model.to(device).eval()
    return model


def pick_compounds(well_map, n_poscon: int, rng, override: list[str] | None):
    """
    Pick compounds to show: n_poscon random positive controls + DMSO.
    Returns list of (compound_name, control_type) tuples.
    """
    poscon = well_map[well_map["control_type"].str.startswith("poscon", na=False)]
    poscon = poscon.drop_duplicates("compound")

    if override:
        chosen = poscon[poscon["compound"].isin(override)]
        missing = set(override) - set(chosen["compound"])
        if missing:
            raise ValueError(f"Requested compounds not found among poscons: {missing}")
        chosen = chosen.set_index("compound").loc[override].reset_index()  # preserve order
    else:
        chosen = poscon.sample(n=min(n_poscon, len(poscon)), random_state=rng)

    rows = list(zip(chosen["compound"], chosen["control_type"]))
    rows.append(("DMSO", "negcon"))
    return rows


def find_crop_for_compound(
    compound: str, well_map, plates: list[str], crop_root: str, rng,
) -> tuple[str, str, str]:
    """
    Pick one random crop PNG for the given compound.

    Searches all wells containing this compound across all plates of the given
    cell type, globs all FOVs, and returns one random path.
    Returns (crop_path, plate, well).
    """
    wells = well_map.loc[well_map["compound"] == compound, "well"].tolist()
    rng.shuffle(wells)

    for well in wells:
        plate_order = list(plates)
        rng.shuffle(plate_order)
        for plate in plate_order:
            paths = glob(os.path.join(crop_root, plate, well, "*", "crops", "*.png"))
            if paths:
                return str(rng.choice(paths)), plate, well

    raise RuntimeError(f"No crops found for compound '{compound}' on plates {plates}")


def preprocess(crop: np.ndarray, channels: list[str], rescale_ratio: float,
               normalize: str) -> torch.Tensor:
    """
    Select channels, rescale, and normalize one crop → (1, 3, H', W').
    Mirrors inference_subcell.py::preprocess_batch.
    """
    x = torch.from_numpy(select_channels(crop, channels)).unsqueeze(0)  # (1, 3, H, W)
    new_size = int(x.shape[-1] * rescale_ratio)
    x = F.interpolate(x, size=new_size, mode="bilinear", align_corners=False)

    if normalize == "per_chan":
        mn = torch.amin(x, dim=(-2, -1), keepdim=True)
        mx = torch.amax(x, dim=(-2, -1), keepdim=True)
        x = (x - mn) / (mx - mn + 1e-6)
    elif normalize == "all_chan":
        mn = torch.amin(x, dim=(-3, -2, -1), keepdim=True)
        mx = torch.amax(x, dim=(-3, -2, -1), keepdim=True)
        x = (x - mn) / (mx - mn + 1e-6)

    return x


def get_display_image(img_uint8: np.ndarray, color_names: list[str]) -> np.ndarray:
    """
    Build a pseudocolor composite from a (H, W, C) uint8 image.

    Each channel is colored via its LUT file in utils/colormaps/ and the
    results are summed into one RGB image (clipped to [0, 255]).
    """
    lut_dir = os.path.join(PROJECT_ROOT, "utils", "colormaps")
    colored = []
    for i, color in enumerate(color_names):
        lut = np.array(
            pd.read_csv(os.path.join(lut_dir, f"{color}.lut"),
                        sep="\t", index_col="Index")
        )[None, ...].astype(np.uint8)
        ch_rgb = img_uint8[..., i][..., None].repeat(3, axis=-1)
        colored.append(cv2.LUT(ch_rgb, lut))
    return np.max(colored, axis=0).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── Load config ─────────────────────────────────────────────────────────
    with open(os.path.join(PROJECT_ROOT, MODEL_CONFIGS[args.model])) as f:
        config = yaml.safe_load(f)

    crop_size        = config["crop_size"]
    rescale_ratio    = config["rescale_ratio"]
    normalize        = config["normalize"]
    context_channels = config["context_channels"]          # [ER, DNA]
    protein_channels = config["protein_channels"]          # [Mito, AGP, RNA]

    if args.protein != "all":
        if args.protein not in protein_channels:
            raise ValueError(f"--protein must be one of {protein_channels} or 'all'")
        protein_channels = [args.protein]

    # ─── Figure out which plates belong to this cell type ────────────────────
    plate_info = load_plate_info()
    plates = plate_info.loc[plate_info["cell_type"] == args.cell_type, "plate"].tolist()
    if not plates:
        raise ValueError(f"No plates found for cell_type={args.cell_type}")
    print(f"Cell type {args.cell_type}: searching {len(plates)} plates")

    # ─── Pick compounds (8 poscon + 1 DMSO) & find one crop each ─────────────
    well_map = load_well_compound_map()
    override = [c.strip() for c in args.compounds.split(",")] if args.compounds else None
    compound_rows = pick_compounds(well_map, args.n_poscon, rng, override)

    crop_root = os.environ["CROP_ROOT"]

    # Optional fixed crops: (cell_type, compound) -> crop path relative to CROP_ROOT.
    # Compounds not in the CSV fall back to random selection.
    fixed_crops = {}
    if args.crops_csv:
        crops_df = pd.read_csv(args.crops_csv)
        crops_df = crops_df[crops_df["cell_type"] == args.cell_type]
        fixed_crops = dict(zip(crops_df["compound"], crops_df["crop"]))
        print(f"Using {len(fixed_crops)} fixed crops from {args.crops_csv}")

    samples = []
    for compound, ctrl_type in compound_rows:
        if compound in fixed_crops:
            rel = fixed_crops[compound]
            path = os.path.join(crop_root, rel)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fixed crop for {compound} not found: {path}")
            plate, well = rel.split("/")[:2]
            source = "csv"
        else:
            path, plate, well = find_crop_for_compound(compound, well_map, plates, crop_root, rng)
            source = "random"
        samples.append(dict(compound=compound, ctrl_type=ctrl_type,
                            path=path, plate=plate, well=well))
        print(f"  {compound:<20s} [{ctrl_type:<15s}] → {plate}/{well}  {os.path.basename(path)}  ({source})")

    # ─── Load model ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(config, device)

    # ─── Load all crops once (9-channel stacks) ──────────────────────────────
    raw_crops = [load_crop(s["path"], crop_size=crop_size) for s in samples]  # (9, H, W) each

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ─── One figure per protein channel ──────────────────────────────────────
    all_vit_attns = []   # collect for averaged figure
    all_pool_attns = []

    for protein in protein_channels:
        channels = context_channels + [protein]   # [ER, DNA, <protein>]
        print(f"\nRunning forward pass for protein={protein}  (channels={channels})")

        # Build batch: (n_samples, 3, H', W')
        batch = torch.cat([
            preprocess(crop, channels, rescale_ratio, normalize)
            for crop in raw_crops
        ], dim=0).to(device)

        with torch.no_grad():
            out = model(batch)

        pool_attn = out.pool_attn.cpu().numpy()    # [n, 2,  h, w]
        vit_attn  = out.attentions.cpu().numpy()   # [n, 12, h, w]

        all_vit_attns.append(vit_attn)
        all_pool_attns.append(pool_attn)

        # ─── LUT composite from raw crop (native resolution) ─────────────
        color_names = [CHANNEL_COLORS[ch] for ch in channels]

        # ─── Plot ────────────────────────────────────────────────────────────
        n_rows = len(samples)
        n_vit, n_pool = vit_attn.shape[1], pool_attn.shape[1]
        n_cols = 1 + n_vit + n_pool

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
        if n_rows == 1:
            axes = axes[None, :]

        for r, s in enumerate(samples):
            # Build LUT composite from the 3 model-input channels at native crop resolution
            img_sel = select_channels(raw_crops[r], channels).transpose(1, 2, 0)  # (H, W, 3)
            mn = img_sel.min(axis=(0, 1), keepdims=True)
            mx = img_sel.max(axis=(0, 1), keepdims=True)
            img_u8 = ((img_sel - mn) / (mx - mn + 1e-8) * 255).clip(0, 255).astype(np.uint8)
            composite = get_display_image(img_u8, color_names)

            axes[r, 0].imshow(composite)
            label = f"{s['compound']}\n({s['ctrl_type']})"
            axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            if r == 0:
                ch_lbl = " ".join(f"{ch}={col}" for ch, col in zip(channels, color_names))
                axes[r, 0].set_title(f"LUT\n({ch_lbl})", fontsize=7)

            for h in range(n_vit):
                ax = axes[r, 1 + h]
                a = vit_attn[r, h]
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                ax.imshow(a, cmap="gray_r")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"ViT h{h}", fontsize=7)

            for h in range(n_pool):
                ax = axes[r, 1 + n_vit + h]
                a = pool_attn[r, h]
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                ax.imshow(a, cmap="gray_r")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"Pool h{h}", fontsize=7)

        fig.suptitle(
            f"{args.cell_type}  |  {config['model_name']}  |  protein={protein}",
            fontsize=11, y=1.005,
        )
        plt.tight_layout()

        stem = f"jump_attention_{args.cell_type}_{args.model}_{protein}_seed{args.seed}"
        fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(args.output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved → {args.output_dir / stem}.{{pdf,png}}")

    # ─── 4th figure: averaged attention across all protein passes ─────────
    if args.protein == "all":
        avg_vit  = np.mean(all_vit_attns, axis=0)   # (n_samples, 12, h, w)
        avg_pool = np.mean(all_pool_attns, axis=0)   # (n_samples, 2,  h, w)

        all_channels = context_channels + protein_channels  # [ER, DNA, Mito, AGP, RNA]
        all_color_names = [CHANNEL_COLORS[ch] for ch in all_channels]

        n_rows = len(samples)
        n_vit, n_pool = avg_vit.shape[1], avg_pool.shape[1]
        n_cols = 1 + n_vit + n_pool

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
        if n_rows == 1:
            axes = axes[None, :]

        for r, s in enumerate(samples):
            # 5-channel LUT composite at native crop resolution
            img_sel = select_channels(raw_crops[r], all_channels).transpose(1, 2, 0)  # (H, W, 5)
            mn = img_sel.min(axis=(0, 1), keepdims=True)
            mx = img_sel.max(axis=(0, 1), keepdims=True)
            img_u8 = ((img_sel - mn) / (mx - mn + 1e-8) * 255).clip(0, 255).astype(np.uint8)
            composite = get_display_image(img_u8, all_color_names)

            axes[r, 0].imshow(composite)
            label = f"{s['compound']}\n({s['ctrl_type']})"
            axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            if r == 0:
                ch_lbl = " ".join(f"{ch}={col}" for ch, col in zip(all_channels, all_color_names))
                axes[r, 0].set_title(f"LUT\n({ch_lbl})", fontsize=6)

            for h in range(n_vit):
                ax = axes[r, 1 + h]
                a = avg_vit[r, h]
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                ax.imshow(a, cmap="gray_r")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"ViT h{h}", fontsize=7)

            for h in range(n_pool):
                ax = axes[r, 1 + n_vit + h]
                a = avg_pool[r, h]
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                ax.imshow(a, cmap="gray_r")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"Pool h{h}", fontsize=7)

        fig.suptitle(
            f"{args.cell_type}  |  {config['model_name']}  |  averaged over {protein_channels}",
            fontsize=11, y=1.005,
        )
        plt.tight_layout()

        stem = f"jump_attention_{args.cell_type}_{args.model}_averaged_seed{args.seed}"
        fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(args.output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved → {args.output_dir / stem}.{{pdf,png}}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize SubCell attention maps for sampled JUMP cells "
                    "(one per positive-control compound + one DMSO)")
    p.add_argument("--cell-type", choices=["A549", "U2OS"], required=True)
    p.add_argument("--model", choices=["mae", "vit"], default="mae",
                   help="mae → subcell_mae config; vit → subcell_vit config")
    p.add_argument("--protein", default="all",
                   help="Which protein channel to show (Mito | AGP | RNA | all). "
                        "'all' generates one figure per channel.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-poscon", type=int, default=8,
                   help="Number of positive-control compounds to sample "
                        "(there are 46 on the plate, not 8). DMSO is always appended.")
    p.add_argument("--compounds", type=str, default=None,
                   help="Comma-separated list of specific poscon compounds to show "
                        "(overrides random sampling). DMSO is always appended.")
    p.add_argument("--crops-csv", type=Path, default=None,
                   help="CSV with columns cell_type, compound, crop (path relative to "
                        "CROP_ROOT). Listed compounds use that exact crop instead of a "
                        "random one; unlisted compounds are sampled as usual.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(os.environ.get("RESULTS_ROOT", "results")) / "attention_maps")
    main(p.parse_args())
