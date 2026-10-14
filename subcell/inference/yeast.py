"""Features of the SubCell encoders on yeast crops, for localization and cell cycle (Fig. 2f,h,j).

Input: the HDF5 files of yeast cells in ``crops_folder`` (``YEAST_DATA``), one per task.

    python -m subcell.inference.yeast -c configs/yeast_subcell_mae.yaml

The config is a list of runs with ``name``, ``crops_folder``, ``inference_folder``, ``resize``,
``save_attention`` and ``model``. For every HDF5 file the features are written to
``<inference_folder>/<name>_resize_<resize>/<file stem>.pth`` as ``(metadata, features)``.
"""
import argparse
import glob
import os

import torch

from subcell import config
from subcell.io.datasets import YeastDataset
from subcell.utils import inference


def run(run_config):
    model = inference.get_model(run_config["name"], run_config["model"])
    inference_folder = f"{run_config['inference_folder']}/{run_config['name']}_resize_{run_config['resize']}"
    print(f"Saving inference results to {inference_folder}")
    attn_folder = f"{inference_folder}/attention_images"
    inference.reset_dir(attn_folder)

    print(f"Starting inference for {run_config['name']}")
    attention = (dict(folder=attn_folder, colors=["red", "blue", "green"], normalization="perchannel")
                 if run_config.get("save_attention", False) else None)
    for crop_file in sorted(glob.glob(f"{run_config['crops_folder']}/*.hdf5")):
        dataset_name = os.path.basename(crop_file).split(".")[0]
        loader = inference.make_loader(YeastDataset(crop_file, run_config["resize"]), batch_size=16)
        all_metadata, all_features = inference.extract_features(model, loader, attention=attention)
        torch.save((all_metadata, all_features), f"{inference_folder}/{dataset_name}.pth")


def main(config_path):
    for run_config in config.load_yaml(config_path):
        run(run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to the configuration file (a list of runs)")
    main(parser.parse_args().config)
