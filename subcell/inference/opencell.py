"""Features of the SubCell encoders on OpenCell crops, for localization (Fig. 2e).

Input: OpenCell single-cell crops and their metadata (``OPENCELL_DATA``).

    python -m subcell.inference.opencell -c configs/opencell_subcell_mae.yaml

The config is a list of runs with ``name``, ``crops_folder``, ``metadata_file``, ``inference_folder``,
``resize``, ``normalization``, ``save_attention`` and ``model``. The features are written to
``<inference_folder>/<name>_resized_<resize>_norm_<normalization>/all_features.pth`` as ``(metadata, features)``.
"""
import argparse
import shutil

import torch

from subcell import config
from subcell.io.datasets import OpenCellDataset
from subcell.utils import inference


def run(run_config):
    run_name = f"{run_config['name']}_resized_{run_config['resize']}_norm_{run_config['normalization']}"
    inference_folder = f"{run_config['inference_folder']}/{run_name}"
    print(f"Saving inference results to {inference_folder}")
    attn_folder = f"{inference_folder}/attention_images"
    temp_folder = f"{inference_folder}/temp"
    inference.reset_dir(attn_folder)
    inference.reset_dir(temp_folder)

    print(f"Starting inference for {run_name}")
    dataset = OpenCellDataset(run_config["crops_folder"], run_config["metadata_file"],
                              run_config["resize"], run_config["normalization"])
    loader = inference.make_loader(dataset, batch_size=16)
    model = inference.get_model(run_config["name"], run_config["model"])
    attention = (dict(folder=attn_folder, colors=["blue", "white"], normalization="perchannel")
                 if run_config.get("save_attention", False) else None)
    all_metadata, all_features = inference.extract_features(model, loader, attention=attention, temp_dir=temp_folder)
    torch.save((all_metadata, all_features), f"{inference_folder}/all_features.pth")
    shutil.rmtree(temp_folder, ignore_errors=True)


def main(config_path):
    for run_config in config.load_yaml(config_path):
        run(run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to the configuration file (a list of runs)")
    main(parser.parse_args().config)
