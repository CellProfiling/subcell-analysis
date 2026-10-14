"""Features of every compared model on the widefield / confocal cell crops of Supplementary Note 1 (Extended Data Fig. 1).

The crops are those of ``subcell.io.microscope_crops`` (``MICROSCOPE_DATA``); all models are run in one pass over them.
Each cell is run once per protein in crop_params["proteins"] (CDT1, then GMNN), the protein in the model's protein
channel; the per-protein vectors are concatenated in that order (P x D). Models (config "models": name, model,
channels, normalizations) are loaded as in ``subcell.inference.hpa`` (bestfitting, cell_dino*, *dino*, *subcell*), each
with its normalization: perimage / perchannel = preprocess_input_subcell per_channel False / True, dino =
preprocess_input_dino, selfnorm = preprocess_input_self_norm, bestfitting = preprocess_input_bestfitting. With
common_only, only cells present at every magnification in ``mags`` are used; rows are sorted by id, so the files of
the two microscopes align row by row.
-> <inference_folder>/<model name>_norm_<normalization>/<mag>_features_pooled.pth
   = (metadata DataFrame, features tensor (N, P x D)), features_layout.json

    python -m subcell.inference.microscopes -c configs/microscopes.yaml [--shard K --nshards N]

Shards split the cells (e.g. one per GPU); ``--merge`` joins the per-shard files. Optional config key ``ids_file``:
restrict to the ids of that csv. The benchmark itself is ``python -m subcell.microscope_robustness``.
"""
import argparse
import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from subcell import config
from subcell.io.microscope_crops import MicroscopeCropDataset, collate_dict
from subcell.io.preprocess import (preprocess_input_bestfitting, preprocess_input_dino,
                                   preprocess_input_self_norm, preprocess_input_subcell)
from subcell.utils import inference


def preprocess_input_bestfitting_uint16(images):
    """preprocess_input_bestfitting divides by the dtype maximum, so the float32 (aligned / resized) crops are rounded back to uint16"""
    return preprocess_input_bestfitting([np.round(image).astype(np.uint16) for image in images])


NORMALIZATIONS = {
    "perchannel": partial(preprocess_input_subcell, per_channel=True),
    "perimage": partial(preprocess_input_subcell, per_channel=False),
    "dino": preprocess_input_dino,
    "selfnorm": preprocess_input_self_norm,
    "bestfitting": preprocess_input_bestfitting_uint16,
}


def model_list(run_config):
    """run_config["models"], or the single-model form"""
    if "models" in run_config:
        return run_config["models"]
    cp = run_config["crop_params"]
    return [dict(name=run_config["name"], model=run_config["model"], channels=cp.get("channels", ["red", "blue", "protein"]),
                 normalizations=cp.get("normalizations", ["perimage"]))]


def input_key(m, n):
    return f"{'-'.join(m['channels'])}|{n}"


def common_ids(index_file, mags, wtag_prefix=None):
    """ids present at every magnification in mags"""
    df = pd.read_parquet(index_file, columns=["id", "mag", "wtag"])
    df = df[df["mag"].isin(mags)]
    if wtag_prefix:
        df = df[df["wtag"].str.startswith(wtag_prefix)]
    n = df.groupby("id")["mag"].nunique()
    return set(n.index[n == len(mags)])


def merge_shards(folders, mags, nshards):
    """Join the per-shard files of every model and magnification; rows sorted by id, as unsharded."""
    for (name, n), folder in folders.items():
        ids0 = None
        for mag in mags:
            parts = [torch.load(f"{folder}/{mag}_features_pooled_shard{k}of{nshards}.pth", weights_only=False) for k in range(nshards)]
            meta = pd.concat([m for m, _ in parts], ignore_index=True)
            feat = torch.cat([f for _, f in parts], dim=0)
            order = meta["id"].argsort(kind="stable").values
            meta = meta.iloc[order].reset_index(drop=True)
            feat = feat[order]
            ids0 = meta["id"].values if ids0 is None else ids0
            assert (meta["id"].values == ids0).all(), f"{name} {n} {mag}: ids differ between magnifications"
            torch.save((meta, feat), f"{folder}/{mag}_features_pooled.pth")
            print(f"merged {name} {n} {mag}: {tuple(feat.shape)}")


def run(run_config, shard=0, nshards=1):
    crop_params = dict(run_config["crop_params"])
    models = model_list(run_config)
    crop_params.setdefault("channels", models[0]["channels"])
    folders = {(m["name"], n): f"{run_config['inference_folder']}/{m['name']}_norm_{n}" for m in models for n in m["normalizations"]}

    ids = common_ids(run_config["index_file"], run_config["mags"], run_config.get("wtag_prefix")) if run_config.get("common_only", True) else None
    if run_config.get("ids_file"):                                               # optional: only these cells (csv with an id column)
        keep = set(pd.read_csv(run_config["ids_file"])["id"])
        ids = (ids & keep) if ids is not None else keep
    if ids is not None and nshards > 1:
        ids = set(sorted(ids)[shard::nshards])
    suffix = f"_shard{shard}of{nshards}" if nshards > 1 else ""
    if run_config.get("tf32", False):                                             # TF32 matmuls (cuDNN convolutions default to TF32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # model inputs: one per (channels, normalisation), shared by the models that use it
    inputs = {input_key(m, n): ([crop_params["channels"].index(c) for c in m["channels"]], [NORMALIZATIONS[n]])
              for m in models for n in m["normalizations"]}
    device = inference.get_device()
    nets = {m["name"]: inference.get_model(m["name"], m["model"]).to(device).eval() for m in models}
    print(f"{run_config['name']}: {len(ids) if ids is not None else 'all'} cells, proteins {crop_params['proteins']}, "
          f"models {[(m['name'], m['normalizations']) for m in models]}")

    for mag in run_config["mags"]:
        dataset = MicroscopeCropDataset(run_config["crops_folder"], run_config["index_file"], mag, crop_params, inputs, ids)
        dataloader = DataLoader(dataset, batch_size=run_config.get("batch_size", 16), shuffle=False,
                                num_workers=run_config.get("num_workers", 8), collate_fn=collate_dict)
        metas, feats = [], {k: [] for k in folders}
        with torch.no_grad():
            for batch, batch_df in tqdm(dataloader, desc=mag):
                batch = {k: v.to(device) for k, v in batch.items()}             # per cell: len(proteins) consecutive images
                for m in models:
                    for n in m["normalizations"]:
                        fv = nets[m["name"]](batch[input_key(m, n)]).feature_vector
                        feats[(m["name"], n)].append(fv.reshape(len(batch_df), -1).cpu())   # (B, P x D): [protein 0 | protein 1]
                metas.append(batch_df)
        all_metadata = pd.concat(metas, ignore_index=True).infer_objects()
        for (name, n), folder in folders.items():
            os.makedirs(folder, exist_ok=True)
            all_features = torch.cat(feats[(name, n)], dim=0)
            torch.save((all_metadata, all_features), f"{folder}/{mag}_features_pooled{suffix}.pth")
            m = next(m for m in models if m["name"] == name)
            with open(f"{folder}/features_layout.json", "w") as f:
                json.dump(dict(model=name, proteins=crop_params["proteins"], dim_per_protein=all_features.shape[1] // len(crop_params["proteins"]),
                               channels=m["channels"], normalization=n,
                               order="features[:, k * dim_per_protein:(k + 1) * dim_per_protein] = proteins[k]",
                               config={**{k: v for k, v in run_config.items() if k != "models"}, "model": m}), f, indent=1)
            print(f"{mag} {name} {n}: {tuple(all_features.shape)} -> {folder}/{mag}_features_pooled{suffix}.pth")


def main(config_path, shard=0, nshards=1, merge=False):
    for run_config in config.load_yaml(config_path):
        if merge:
            folders = {(m["name"], n): f"{run_config['inference_folder']}/{m['name']}_norm_{n}"
                       for m in model_list(run_config) for n in m["normalizations"]}
            merge_shards(folders, run_config["mags"], nshards)
        else:
            run(run_config, shard, nshards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to the configuration file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="join the shard files of every magnification")
    args = parser.parse_args()
    main(args.config, args.shard, args.nshards, args.merge)
