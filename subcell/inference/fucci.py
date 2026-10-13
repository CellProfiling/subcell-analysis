"""Features of the SubCell encoders on FUCCI U2OS crops, for cell-cycle stage and pseudotime (Fig. 2k,l).

Input: the single-cell crops built by ``subcell.inference.fucci_crops`` (``FUCCI_DATA``). The images have
no nuclear stain; ``channel_combination`` selects the substitute fed to the DNA channel: ``nuc_zero`` (empty),
``nuc_const`` (the nuclear mask at constant intensity) or ``nuc_cdt_gmmn`` (mean of the CDT1 and GMNN channels
inside the nucleus). With ``resize`` the crops are rescaled to the HPA pixel size.

    python -m subcell.inference.fucci -c configs/config_subcell_mae_fucci.yaml

The config is a list of runs with ``name``, ``crops_folder``, ``metadata_file``, ``inference_folder``,
``channel_combination``, ``resize``, ``save_attention`` and ``model``. The features are written to
``<inference_folder>/<name>_channel_comb_<channel_combination>_resize_<resize>/all_features.pth`` as
``(metadata, features)``; ``subcell.classification.cellcycle_fucci`` reads them.
"""
import argparse
import shutil

import torch

from subcell import config
from subcell.io.datasets import FUCCIDataset
from subcell.utils import inference


def run(run_config):
    run_name = f"{run_config['name']}_channel_comb_{run_config['channel_combination']}_resize_{run_config['resize']}"
    inference_folder = f"{run_config['inference_folder']}/{run_name}"
    print(f"Saving inference results to {inference_folder}")
    attn_folder = f"{inference_folder}/attention_images"
    temp_folder = f"{inference_folder}/temp"
    inference.reset_dir(attn_folder)
    inference.reset_dir(temp_folder)

    print(f"Starting inference for {run_name}")
    dataset = FUCCIDataset(run_config["crops_folder"], run_config["metadata_file"],
                           run_config["channel_combination"], run_config["resize"])
    loader = inference.make_loader(dataset, batch_size=16)
    model = inference.get_model(run_config["name"], run_config["model"])
    attention = (dict(folder=attn_folder, colors=["red", "light_blue", "orange", "green"], normalization="fucci")
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
