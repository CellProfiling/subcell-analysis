"""
Generate DINO4Cells attention-map figures for the same JUMP cells sampled by
scripts/jump_attention_map.py (SubCell version).

Cell selection is identical: pick_compounds() and find_crop_for_compound()
below are byte-for-byte copies of the SubCell script, and the rng is
consumed in the same order. Given the same --seed, --cell-type, and
--n-poscon, the two scripts will pick the same crops.

DINO differs from SubCell in three ways that affect the figure:
  - Single forward pass with all 5 CP channels (vs 3 passes) → 1 figure, not 3.
  - No gated pooler → no pool-attention columns.
  - 128px input, patch 16 → 8×8 attention grid (vs SubCell's 12×12 at 192px).

Layout: rows = compounds, cols = [ RGB | 12 ViT heads ]

Run from the repo root (after `set -a; source .env; set +a`):
    python scripts/jump_attention_map_dino.py --cell-type A549 --seed 0
    python scripts/jump_attention_map_dino.py --cell-type U2OS --seed 0
"""
import argparse
import os
import random
import sys
import types
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
DINO_ROOT = os.path.join(PROJECT_ROOT, "models", "DINO4Cells_code")

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
# Import ordering matters here.
#
# 1. Import project utils FIRST. This caches `utils`, `utils.metadata`,
#    `utils.crop_loader` in sys.modules. Later sys.path changes can't shadow them.
# 2. THEN stub sys.modules["utils.utils"] with torch's trunc_normal_.
# 3. THEN insert DINO_ROOT and import DINO's ViT. When vision_transformer.py
#    does `from utils.utils import trunc_normal_`, Python checks sys.modules
#    first, finds our stub, and never touches the filesystem.
#
# This avoids the conflict that forced inference_dino.py to inline its loaders.
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, PROJECT_ROOT)
from utils.crop_loader import load_crop, select_channels  # noqa: E402
from utils.metadata import load_plate_info, load_well_compound_map  # noqa: E402

_stub = types.ModuleType("utils.utils")
_stub.trunc_normal_ = torch.nn.init.trunc_normal_
sys.modules["utils.utils"] = _stub

sys.path.insert(0, DINO_ROOT)
from archs.vision_transformer import vit_base  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Cell-selection helpers — VERBATIM from jump_attention_map.py.
# Keep these byte-identical so the same seed yields the same cells.
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
def load_dino_model(config: dict, device: torch.device):
    """Build vit_base and load the teacher backbone. Mirrors inference_dino.py."""
    ckpt_path = os.environ[config["ckpt_env_key"]]
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
    model.to(device).eval()
    return model


def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """Per-channel min-max → per-channel z-score. Mirrors inference_dino.py."""
    mn = torch.amin(x, dim=(-2, -1), keepdim=True)
    mx = torch.amax(x, dim=(-2, -1), keepdim=True)
    x = (x - mn) / (mx - mn + 1e-6)

    mean = torch.mean(x, dim=(-2, -1), keepdim=True)
    std = torch.std(x, dim=(-2, -1), keepdim=True, unbiased=False)
    x = (x - mean) / (std + 1e-7)
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
    with open(os.path.join(PROJECT_ROOT, "configs", "dino.yaml")) as f:
        config = yaml.safe_load(f)

    crop_size  = config["crop_size"]       # 128
    channels   = config["channels"]        # [DNA, RNA, ER, AGP, Mito]
    patch_size = config["patch_size"]      # 16

    # ─── Figure out which plates belong to this cell type ────────────────────
    plate_info = load_plate_info()
    plates = plate_info.loc[plate_info["cell_type"] == args.cell_type, "plate"].tolist()
    if not plates:
        raise ValueError(f"No plates found for cell_type={args.cell_type}")
    print(f"Cell type {args.cell_type}: searching {len(plates)} plates")

    # ─── Pick compounds (8 poscon + 1 DMSO) & find one crop each ─────────────
    # This block is identical to jump_attention_map.py — same rng, same call
    # order, same inputs → same cells.
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
    model = load_dino_model(config, device)

    # ─── Load crops + build one batch (DINO is single-pass) ──────────────────
    raw_crops = [load_crop(s["path"], crop_size=crop_size) for s in samples]  # (9, 128, 128) each
    batch = torch.from_numpy(np.stack([
        select_channels(c, channels) for c in raw_crops
    ]))  # (n, 5, 128, 128)
    batch = preprocess_batch(batch)

    # ─── Optional upscaling ─────────────────────────────────────────────────
    input_size = int(crop_size * args.upscale_factor)
    if args.upscale_factor > 1.0:
        batch = F.interpolate(batch, size=input_size, mode="bilinear",
                              align_corners=False)
        print(f"Upscaled {crop_size}→{input_size}px (factor {args.upscale_factor})")

    batch = batch.to(device)

    # ─── Extract last-layer self-attention ──────────────────────────────────
    # get_last_selfattention returns (B, num_heads, N+1, N+1) where N = h*w.
    # Slice [:, :, 0, 1:] = CLS-token attention to each patch, same convention
    # SubCellPortable uses (vit_model.py:391).
    # Positional embeddings are bicubic-interpolated inside the model when the
    # input size differs from the training size.
    with torch.no_grad():
        attn = model.get_last_selfattention(batch)

    h_feat = w_feat = input_size // patch_size
    vit_attn = attn[:, :, 0, 1:].reshape(len(samples), -1, h_feat, w_feat).cpu().numpy()

    # ─── Plot ───────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(samples)
    n_vit  = vit_attn.shape[1]
    n_cols = 1 + n_vit

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
    if n_rows == 1:
        axes = axes[None, :]

    color_names = [CHANNEL_COLORS[ch] for ch in channels]

    for r, s in enumerate(samples):
        # 5-channel LUT composite at native crop resolution
        img_sel = select_channels(raw_crops[r], channels).transpose(1, 2, 0)  # (H, W, 5)
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
            axes[r, 0].set_title(f"LUT\n({ch_lbl})", fontsize=6)

        for h in range(n_vit):
            ax = axes[r, 1 + h]
            a = vit_attn[r, h]
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            ax.imshow(a, cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"ViT h{h}", fontsize=7)

    res_lbl = f"{input_size}px" if args.upscale_factor > 1 else f"{crop_size}px"
    fig.suptitle(
        f"{args.cell_type}  |  {config['model_name']}  |  {res_lbl}  |  5-channel single pass",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()

    up_tag = f"_up{args.upscale_factor}" if args.upscale_factor > 1.0 else ""
    stem = f"jump_attention_{args.cell_type}_dino{up_tag}_seed{args.seed}"
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(args.output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {args.output_dir / stem}.{{pdf,png}}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize DINO4Cells attention maps for the same JUMP cells "
                    "that jump_attention_map.py samples (given the same seed).")
    p.add_argument("--cell-type", choices=["A549", "U2OS"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-poscon", type=int, default=8,
                   help="Number of positive-control compounds to sample. "
                        "Must match the SubCell run to get the same cells.")
    p.add_argument("--compounds", type=str, default=None,
                   help="Comma-separated list of specific poscon compounds to show "
                        "(overrides random sampling). DMSO is always appended.")
    p.add_argument("--upscale-factor", type=float, default=1.0,
                   help="Upscale crops by this factor before feeding to DINO. "
                        "E.g. 3.74 → 128→478px. Pos embeddings are "
                        "bicubic-interpolated inside the model.")
    p.add_argument("--crops-csv", type=Path, default=None,
                   help="CSV with columns cell_type, compound, crop (path relative to "
                        "CROP_ROOT). Listed compounds use that exact crop instead of a "
                        "random one; unlisted compounds are sampled as usual.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(os.environ.get("RESULTS_ROOT", "results")) / "attention_maps")
    main(p.parse_args())
