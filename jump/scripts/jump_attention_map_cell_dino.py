"""
Generate Cell-DINO attention-map figures for the same JUMP cells sampled by
scripts/jump_attention_map_subcell.py and scripts/jump_attention_map_dino.py.

Cell selection is identical: pick_compounds() and find_crop_for_compound()
below are byte-for-byte copies of the SubCell script, and the rng is
consumed in the same order. Given the same --seed, --cell-type, and
--n-poscon, the scripts will pick the same crops.

Cell-DINO differs from DINO4Cells in ways that affect the figure:
  - ViT-S/8 at 128px → 16×16 attention grid, 6 heads (vs 8×8, 12 heads).
  - Preprocessing is x/255 → per-channel z-score (no min-max step).
  - Optional --apply-mask, matching the cell_dino_masked embeddings.
  - dinov2's attention uses a fused kernel that never materializes the
    attention matrix, so get_last_selfattention() below recomputes it from
    the last block's q/k and asserts the recomputation matches the model.

Layout: rows = compounds, cols = [ RGB | 6 ViT heads ]

Run from the repo root (after `set -a; source .env; set +a`):
    python scripts/jump_attention_map_cell_dino.py --cell-type A549 --seed 0 --apply-mask
    python scripts/jump_attention_map_cell_dino.py --cell-type U2OS --seed 0 --apply-mask
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

# Same import setup as inference_cell_dino.py
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.environ["DINOV2_ROOT"])
from dinov2.hub.cell_dino.backbones import cell_dino_cp_vits8  # noqa: E402
from utils.crop_loader import get_mask, load_crop, select_channels  # noqa: E402
from utils.metadata import load_plate_info, load_well_compound_map  # noqa: E402


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
def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Per-channel z-score. x: (B, C, H, W) float32 already in [0, 1].
    Mirrors inference_cell_dino.py (Cell-DINO SelfNormalizeNoDiv).
    """
    mean = torch.mean(x, dim=(-2, -1), keepdim=True)
    std = torch.std(x, dim=(-2, -1), keepdim=True, unbiased=False)
    return (x - mean) / (std + 1e-7)


@torch.no_grad()
def get_last_selfattention(model, x: torch.Tensor) -> torch.Tensor:
    """
    Last-block self-attention weights (B, heads, N, N) for a dinov2 ViT.

    dinov2 computes attention with a fused kernel (layers/attention.py:74 / :94)
    that never materializes softmax(q·kᵀ). We hook the last block's qkv Linear
    (attention.py:52) to capture q, k, v from a normal forward pass, recompute
    the weights with the same formula, then assert that weights @ v -> proj
    reproduces the attention module's real output.
    """
    # block_chunks=4 → blocks are BlockChunks padded with Identity (vision_transformer.py:156-163)
    last_blk = model.blocks[-1][-1] if model.chunked_blocks else model.blocks[-1]
    attn_mod = last_blk.attn

    captured = {}
    h_qkv = attn_mod.qkv.register_forward_hook(lambda m, i, o: captured.__setitem__("qkv", o))
    h_out = attn_mod.register_forward_hook(lambda m, i, o: captured.__setitem__("out", o))
    try:
        model.forward_features(x)
    finally:
        h_qkv.remove()
        h_out.remove()

    qkv = captured["qkv"]                                    # (B, N, 3*C)
    B, N, C3 = qkv.shape
    n_heads = attn_mod.num_heads
    qkv = qkv.reshape(B, N, 3, n_heads, C3 // 3 // n_heads)  # same as attention.py:90
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))     # (B, heads, N, head_dim)

    attn = torch.softmax((q @ k.transpose(-2, -1)) * attn_mod.scale, dim=-1)

    # Check: rebuild the module output from our weights and compare to the real one
    recon = attn_mod.proj((attn @ v).transpose(1, 2).reshape(B, N, -1))
    max_err = (recon - captured["out"]).abs().max().item()
    assert torch.allclose(recon, captured["out"], atol=1e-4, rtol=1e-3), (
        f"Recomputed attention does not reproduce the model's attention output "
        f"(max abs err {max_err:.2e})"
    )
    print(f"Attention check passed (max abs err {max_err:.2e})")
    return attn


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
    with open(os.path.join(PROJECT_ROOT, "configs", "cell_dino.yaml")) as f:
        config = yaml.safe_load(f)

    crop_size  = config["crop_size"]       # 128
    channels   = config["channels"]        # [DNA, RNA, ER, AGP, Mito]
    model_name = config["model_name"] + ("_masked" if args.apply_mask else "")

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
    ckpt_path = os.environ[config["ckpt_env_key"]]
    model = cell_dino_cp_vits8(pretrained_path=ckpt_path).eval().to(device)
    patch_size = model.patch_size          # 8

    # ─── Load crops + build one batch (single pass) ──────────────────────────
    raw_crops = [load_crop(s["path"], crop_size=crop_size) for s in samples]  # (9, 128, 128) each
    batch = torch.from_numpy(np.stack([
        select_channels(c, channels) for c in raw_crops
    ]))  # (n, 5, 128, 128)
    batch = preprocess_batch(batch)
    if args.apply_mask:
        # Same order as inference_cell_dino.py: normalize, then mask
        masks = torch.from_numpy(np.stack([get_mask(c) for c in raw_crops]))  # (n, 1, H, W)
        batch = batch * masks

    # ─── Optional upscaling ─────────────────────────────────────────────────
    # PatchEmbed requires a multiple of patch_size (dinov2 layers/patch_embed.py:72),
    # so round to the nearest multiple (e.g. 3.74 → 480px, not 478).
    input_size = int(round(crop_size * args.upscale_factor / patch_size)) * patch_size
    if args.upscale_factor > 1.0:
        batch = F.interpolate(batch, size=input_size, mode="bilinear",
                              align_corners=False)
        print(f"Upscaled {crop_size}→{input_size}px (factor {args.upscale_factor})")

    batch = batch.to(device)

    # ─── Extract last-layer self-attention ──────────────────────────────────
    # Slice [:, :, 0, 1:] = CLS-token attention to each patch, same convention
    # as the DINO/SubCell scripts. Positional embeddings are bicubic-interpolated
    # inside the model when the input size differs from the training size.
    attn = get_last_selfattention(model, batch)

    h_feat = w_feat = input_size // patch_size
    assert attn.shape[-1] == 1 + h_feat * w_feat  # CLS + patches, no registers
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
        f"{args.cell_type}  |  {model_name}  |  {res_lbl}  |  5-channel single pass",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()

    up_tag = f"_up{args.upscale_factor}" if args.upscale_factor > 1.0 else ""
    stem = f"jump_attention_{args.cell_type}_{model_name}{up_tag}_seed{args.seed}"
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(args.output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {args.output_dir / stem}.{{pdf,png}}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visualize Cell-DINO attention maps for the same JUMP cells "
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
                   help="Upscale crops by this factor before feeding to Cell-DINO, "
                        "rounded to a multiple of the patch size (8). E.g. 3.74 → "
                        "128→480px. Pos embeddings are bicubic-interpolated inside the model.")
    p.add_argument("--apply-mask", action="store_true",
                   help="Multiply normalized channels by the cell segmentation mask "
                        "before the forward pass (matches cell_dino_masked embeddings).")
    p.add_argument("--crops-csv", type=Path, default=None,
                   help="CSV with columns cell_type, compound, crop (path relative to "
                        "CROP_ROOT). Listed compounds use that exact crop instead of a "
                        "random one; unlisted compounds are sampled as usual.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(os.environ.get("RESULTS_ROOT", "results")) / "attention_maps")
    main(p.parse_args())
