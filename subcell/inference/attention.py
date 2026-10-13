"""Attention maps of the HPA cells, scored against the input channels (Fig. 3f,g).

For every cell, the CLS attention of the last ViT layer (heads AH-1, AH-2, ...) and the
attention-pooling heads (PAH-1, PAH-2), where the encoder has them, are upsampled
to the crop and compared with each input channel by Pearson and Spearman
correlation and by normalized mutual information. The result is
``{inference_folder}/{name}/all_relations.pth``: a dict with the cell table, the
head names and one [cells, channels, heads] tensor per measure, with the channels
in ``CHANNEL_ORDER`` (MT, ER, Nuc, Prot) whatever the order the encoder reads.

    python -m subcell.inference.attention -c configs/config_attention_hpa.yaml --name subcell_rybg_vit

The config is a list of runs, as for ``subcell.inference.hpa``. Batches are written
as they finish and skipped on a rerun, so an interrupted run resumes where it
stopped.
"""

import argparse
import os
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from subcell import config
from subcell.io.datasets import HPATestDataset2, collate_fn
from subcell.io.display import process_attn
from subcell.inference import hpa
from subcell.utils import inference


def fast_per_image_nmi(tensor1, tensor2, bins=32):
    """Normalized mutual information between every channel pair of each image.

    Shapes: [B, N1, H, W], [B, N2, H, W] -> [B, N1, N2]. Each channel is min-max
    scaled to *bins* levels; all N1 x N2 joint histograms of an image come from one
    bincount.
    """
    B, N1 = tensor1.shape[:2]
    N2 = tensor2.shape[1]

    def levels(t):
        t = t.reshape(t.shape[0], t.shape[1], -1)
        lo, hi = t.min(dim=2, keepdim=True)[0], t.max(dim=2, keepdim=True)[0]
        return ((t - lo) / (hi - lo + 1e-6) * (bins - 1)).long()

    t1, t2 = levels(tensor1), levels(tensor2)
    pair = torch.arange(N1 * N2, device=t1.device).view(N1, N2, 1) * bins**2
    results = torch.zeros((B, N1, N2), device=tensor1.device)
    for b in range(B):
        indices = pair + t1[b][:, None, :] * bins + t2[b][None, :, :]
        joint = torch.bincount(indices.flatten(), minlength=N1 * N2 * bins**2).float()
        joint = joint.view(N1, N2, bins, bins)
        joint /= joint.sum(dim=(2, 3), keepdim=True)
        p_i, p_j = joint.sum(dim=3), joint.sum(dim=2)
        H_i = -torch.sum(p_i * torch.log(p_i + 1e-9), dim=2)
        H_j = -torch.sum(p_j * torch.log(p_j + 1e-9), dim=2)
        H_ij = -torch.sum(joint * torch.log(joint + 1e-9), dim=(2, 3))
        results[b] = 2 * (H_i + H_j - H_ij) / (H_i + H_j + 1e-9)
    return results


def fast_channel_correlation(tensor1, tensor2):
    """Pearson correlation between every channel pair of each image.

    Shapes: [B, N1, H, W], [B, N2, H, W] -> [B, N1, N2]. In float64: some heads
    attend almost uniformly, and centring a near-constant map in float32 cancels
    away most of its digits.
    """
    B, N1 = tensor1.shape[:2]
    N2 = tensor2.shape[1]
    t1 = tensor1.reshape(B, N1, -1).double()
    t2 = tensor2.reshape(B, N2, -1).double()
    t1 = torch.nn.functional.normalize(t1 - t1.mean(dim=2, keepdim=True), p=2, dim=2)
    t2 = torch.nn.functional.normalize(t2 - t2.mean(dim=2, keepdim=True), p=2, dim=2)
    return torch.bmm(t1, t2.transpose(1, 2)).float()


def average_ranks(t):
    """Ranks along the pixels of each channel, ties given their average rank."""
    flat = t.reshape(t.shape[0], t.shape[1], -1).contiguous()
    ordered = flat.sort(dim=2).values
    left = torch.searchsorted(ordered, flat, right=False)
    right = torch.searchsorted(ordered, flat, right=True)
    return ((left + right).float() / 2).view_as(t)


def fast_channel_spearman(tensor1, tensor2):
    """Spearman correlation between every channel pair of each image."""
    return fast_channel_correlation(average_ranks(tensor1), average_ranks(tensor2))


def load_model(run_config):
    """The encoder of a run and its default input preprocessing."""
    name = run_config["name"]
    return inference.get_model(name, run_config["model"]), hpa.defaults(name)[0]


MEASURES = ("pearson", "spearman", "nmi")
CHANNEL_ORDER = ["red", "yellow", "blue", "green"]


def head_names(model_output):
    """AH-1.. for the heads of the last layer, then PAH-1.. for the pooling heads."""
    names = [f"AH-{i + 1}" for i in range(model_output.attentions[-1].shape[1])]
    if model_output.pool_attn is not None:
        names += [f"PAH-{i + 1}" for i in range(model_output.pool_attn.shape[1])]
    return names


def combine(parts):
    out = {"metadata": pd.concat([p["metadata"] for p in parts], ignore_index=True),
           "heads": parts[0]["heads"]}
    for measure in MEASURES:
        out[measure] = torch.cat([p[measure] for p in parts], dim=0)
    return out


def run(run_config, device):
    model, preprocess_algo = load_model(run_config)
    out_folder = f"{run_config['inference_folder']}/{run_config['name']}"
    temp_folder = f"{out_folder}/temp"
    os.makedirs(temp_folder, exist_ok=True)

    dataset = HPATestDataset2(
        run_config["crops_folder"], run_config["metadata_file"], run_config["crop_params"], preprocess_algo
    )
    order = [run_config["crop_params"]["channels"].index(c) for c in CHANNEL_ORDER]
    batch_size = run_config.get("batch_size", 24)
    # results are written per chunk of batches, which keeps the run resumable
    # without leaving tens of thousands of small files on scratch
    chunk = run_config.get("chunk_batches", 100)
    n_batches = -(-len(dataset) // batch_size)
    n_chunks = -(-n_batches // chunk)
    todo_chunks = [c for c in range(n_chunks) if not os.path.exists(f"{temp_folder}/{c}.pt")]
    todo = [i for c in todo_chunks for i in range(c * chunk, min((c + 1) * chunk, n_batches))]
    print(f"{run_config['name']}: {len(dataset):,} cells, {n_chunks - len(todo_chunks)} of "
          f"{n_chunks} chunks already done", flush=True)
    indices = [j for i in todo for j in range(i * batch_size, min((i + 1) * batch_size, len(dataset)))]
    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=run_config.get("num_workers", 8),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    parts = []
    with torch.no_grad():
        model.to(device)
        model.eval()
        for idx, (batch_images, _, batch_df) in tqdm(
            zip(todo, dataloader), total=len(todo), mininterval=60
        ):
            b, _, h, w = batch_images.shape
            batch_images = batch_images.to(device, non_blocking=True)
            op = model(batch_images)

            if hasattr(model, "vit_config"):
                patch_size = model.vit_config.patch_size
            elif hasattr(model, "patch_embed"):
                patch_size = model.patch_embed.patch_size
            else:
                patch_size = model.patch_size
            attention_maps = process_attn(op, b, h // patch_size, w // patch_size, h, w)
            channels = batch_images[:, order]

            parts.append({
                "metadata": batch_df,
                "heads": head_names(op),
                "pearson": fast_channel_correlation(channels, attention_maps).cpu(),
                "spearman": fast_channel_spearman(channels, attention_maps).cpu(),
                "nmi": fast_per_image_nmi(channels, attention_maps).cpu(),
            })
            if (idx + 1) % chunk == 0 or idx == n_batches - 1:
                path = f"{temp_folder}/{idx // chunk}.pt"
                torch.save(combine(parts), f"{path}.tmp")
                os.replace(f"{path}.tmp", path)
                parts = []

    relations = combine([torch.load(f"{temp_folder}/{c}.pt", map_location="cpu", weights_only=False)
                         for c in range(n_chunks)])
    assert len(relations["metadata"]) == len(dataset)
    torch.save(relations, f"{out_folder}/all_relations.pth")
    shutil.rmtree(temp_folder, ignore_errors=True)
    print(f"wrote {out_folder}/all_relations.pth", flush=True)


def main(config_path, names=None):
    device = inference.get_device()
    for run_config in config.load_yaml(config_path):
        if not names or run_config["name"] in names:
            run(run_config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to configuration file")
    parser.add_argument("--name", nargs="*", help="runs of the config to do (default: all)")
    args = parser.parse_args()
    main(args.config, args.name)
