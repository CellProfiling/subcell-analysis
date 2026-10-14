"""Cell-level features of an encoder on single-cell crops of HPA-style images.

Runs each encoder of the compared set (SubCell MAE/ViT, DINO4Cells, Cell-DINO, bestfitting) over
single-cell crops of the HPA (``HPA_DATA``), the HPA Kaggle test set (``KAGGLE_DATA``) or the CM4AI
perturbation images (``CM4AI_DATA``). The features are the input of the localization benchmarks
(Fig. 2a-c), the feature analyses (Fig. 3, Extended Data Figs. 3 and 8) and the CM4AI analysis (Fig. 4).

    python -m subcell.inference.hpa -c configs/hpa_subcell_mae.yaml

The config is a list of runs. Per run the encoder, its default preprocessing and the Dataset follow
from ``name`` (``bestfitting``, ``cell_dino*``, ``*dino*``, ``*subcell*``); the key ``dataset``
overrides the Dataset. Other keys: ``crops_folder``, ``metadata_file``,
``inference_folder``, ``crop_params``, ``model``, ``save_attention`` (attention panels for every 25th
batch), ``save_patch_features`` (default true: the patch tokens of each batch are kept in
``<inference_folder>/<name>/temp``), ``num_workers``.

Output: with ``output_path``, ``(metadata[feature_meta_cols], features)`` saved there (the
cell-level feature tensor of ``config.features``; ``feature_meta_cols`` defaults to ``FEATURE_META_COLS``);
otherwise ``<inference_folder>/<name>/all_features_pooled.pth`` with the metadata of the batches.
"""
import argparse
import os
import shutil

import pandas as pd
import torch

from subcell import config
from subcell.io.datasets import HPATestDataset, HPATestDataset2, HPATestDataset3
from subcell.io.preprocess import (preprocess_input_bestfitting, preprocess_input_dino,
                                   preprocess_input_self_norm, preprocess_input_subcell)
from subcell.utils import inference

FEATURE_META_COLS = ["cell_id", "if_plate_id", "position", "sample", "atlas_name",
                     "antibody", "ensembl_ids", "gene_names", "locations", "cell_path"]
DATASETS = {"HPATestDataset": HPATestDataset, "HPATestDataset2": HPATestDataset2, "HPATestDataset3": HPATestDataset3}


def defaults(name):
    """(preprocessing, Dataset class) of the encoder ``name``."""
    if name == "bestfitting":
        return preprocess_input_bestfitting, HPATestDataset3
    if name.startswith("cell_dino"):
        return preprocess_input_self_norm, HPATestDataset2
    if "dino" in name:
        return preprocess_input_dino, HPATestDataset3
    if "subcell" in name:
        return preprocess_input_subcell, HPATestDataset3
    raise ValueError(f"Model {name} not supported")


def run(run_config):
    name = run_config["name"]
    model = inference.get_model(name, run_config["model"])
    preprocess, dataset_cls = defaults(name)
    dataset_cls = DATASETS[run_config["dataset"]] if "dataset" in run_config else dataset_cls

    inference_folder = f"{run_config['inference_folder']}/{name}"
    print(f"Saving inference results to {run_config['inference_folder']}")
    attn_folder = f"{inference_folder}/attention_images"
    temp_folder = f"{inference_folder}/temp"
    inference.reset_dir(attn_folder)
    inference.reset_dir(temp_folder)

    print(f"Starting inference for {name} with crop params {run_config['crop_params']}")
    dataset = dataset_cls(run_config["crops_folder"], run_config["metadata_file"], run_config["crop_params"], preprocess)
    loader = inference.make_loader(dataset, batch_size=24, num_workers=run_config.get("num_workers", 8))
    attention = (dict(folder=attn_folder, colors=["red", "yellow", "blue", "green"], normalization="perchannel")
                 if run_config.get("save_attention", False) else None)
    save_patch = run_config.get("save_patch_features", True)
    all_metadata, all_features = inference.extract_features(model, loader, attention=attention,
                                                            temp_dir=temp_folder, save_patch=save_patch)
    if not save_patch:
        shutil.rmtree(temp_folder)

    if run_config.get("output_path"):
        meta = pd.read_csv(run_config["metadata_file"], index_col=0)
        os.makedirs(os.path.dirname(run_config["output_path"]) or ".", exist_ok=True)
        torch.save((meta[run_config.get("feature_meta_cols", FEATURE_META_COLS)], all_features), run_config["output_path"])
    else:
        torch.save((all_metadata, all_features), f"{inference_folder}/all_features_pooled.pth")


def main(config_path):
    for run_config in config.load_yaml(config_path):
        run(run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to the configuration file (a list of runs)")
    main(parser.parse_args().config)
