"""Shared steps of the feature-extraction stages in ``subcell.inference``.

The ``hpa``, ``allencell``, ``opencell`` and ``yeast`` modules differ in their Dataset,
channel set and output folder name and share everything else: loading the encoder,
iterating over batches, writing attention panels for every 25th batch, and collecting
the pooled feature vectors with their metadata into ``(metadata, features)``.
"""

import os
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from subcell.io.datasets import collate_fn
from subcell.io.display import process_attn, save_attention_images
from subcell.models.load import get_bestfitting_model, get_cell_dino_model, get_dino_model, get_subcell_model

ATTENTION_EVERY = 25    # an attention panel is written for every 25th batch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(name, model_config):
    """The encoder a run's ``name`` refers to (bestfitting, cell_dino*, *dino*, *subcell*)."""
    if name == "bestfitting":
        return get_bestfitting_model(model_config)
    if name.startswith("cell_dino"):
        return get_cell_dino_model(model_config)
    if "dino" in name:
        return get_dino_model(model_config)
    if "subcell" in name:
        return get_subcell_model(model_config)
    raise ValueError(f"Model {name} not supported")


def reset_dir(path):
    """Empty *path*, creating it if needed."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def make_loader(dataset, batch_size, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)


def encode(model, batch_images):
    """Default forward pass: one image batch -> (feature vectors, model output)."""
    op = model(batch_images)
    return op.feature_vector, op


def extract_features(model, dataloader, forward=encode, attention=None, temp_dir=None, save_patch=False):
    """Run *model* over *dataloader* and return ``(metadata, features)`` for all cells, in order.

    ``forward(model, batch_images)`` returns ``(features, model_output)``.
    ``attention`` is ``dict(folder, colors, normalization)`` for ``save_attention_images``, or
    ``None`` for no attention panels.
    With ``temp_dir``, each batch is first written there (``<i>_metadata.csv``, ``<i>_features_pooled.pt``
    and, with ``save_patch``, the patch tokens ``<i>_features_patch.pt``) and the metadata make a
    CSV round trip when collected; otherwise batches are kept in memory.
    """
    metas, feats = [], []
    device = get_device()
    with torch.no_grad():
        model.to(device)
        model.eval()
        for idx, (batch_images, batch_images_orig, batch_df) in tqdm(enumerate(dataloader), total=len(dataloader)):
            b, h, w = batch_images.shape[0], batch_images.shape[-2], batch_images.shape[-1]
            features, op = forward(model, batch_images.to(device))
            if attention and idx % ATTENTION_EVERY == 0 and hasattr(op, "attentions"):
                patch = model.vit_config.patch_size
                attention_maps = process_attn(op, b, h // patch, w // patch, h, w)
                save_attention_images(
                    images=batch_images_orig,
                    attention_maps=attention_maps,
                    color_channels=attention["colors"],
                    normalization=attention["normalization"],
                    metadata=batch_df,
                    save_folder=attention["folder"],
                    batch_idx=idx,
                )
            if temp_dir:
                batch_df.to_csv(f"{temp_dir}/{idx}_metadata.csv", index=False)
                torch.save(features.clone(), f"{temp_dir}/{idx}_features_pooled.pt")
                if save_patch:
                    torch.save(op.last_hidden_state, f"{temp_dir}/{idx}_features_patch.pt")
            else:
                metas.append(batch_df)
                feats.append(features.cpu())
    if temp_dir:
        for i in tqdm(range(len(dataloader))):
            metas.append(pd.read_csv(f"{temp_dir}/{i}_metadata.csv"))
            feats.append(torch.load(f"{temp_dir}/{i}_features_pooled.pt", map_location="cpu", weights_only=False))
    return pd.concat(metas, ignore_index=True), torch.cat(feats, dim=0)
