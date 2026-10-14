"""Features of the SubCell encoders on Allen Cell hiPSC crops, for cell-cycle stage (Fig. 2g,i).

Input: the HPA-resolution projections built by ``subcell.inference.allencell_crops`` (``ALLENCELL_DATA``).
The channel combination (``concat``, ``nuc_prot_only``, ``plasma_nuc_prot``) and the normalization
(``perchannel``, ``perimage``) of each run give the channel/normalization sweep of Supplementary Table 10.

    python -m subcell.inference.allencell -c configs/allencell_subcell_mae.yaml

The config is a list of runs with ``name``, ``crops_folder``, ``metadata_file``, ``inference_folder``,
``channel_combination``, ``normalization``, ``save_attention`` and ``model``. The features are written to
``<inference_folder>/<name>_channel_comb_<channel_combination>_norm_<normalization>/all_features.pth`` as
``(metadata, features)``; with ``concat`` the two DNA/structure views of a cell are encoded separately and
their vectors concatenated.
"""
import argparse
import shutil

import torch

from subcell import config
from subcell.io.datasets import AllenCellDataset
from subcell.models.vit_pool import ViTPoolModelOutput
from subcell.utils import inference


def encode_views(model, batch_images):
    """Forward pass for ``[B, views, C, H, W]`` batches (``concat``): the views are encoded one by one, their
    features concatenated and their attention averaged."""
    if batch_images.dim() == 4:
        return inference.encode(model, batch_images)
    ops = [model(batch_images[:, i]) for i in range(batch_images.shape[1])]
    features = torch.cat([op.feature_vector for op in ops], dim=1)
    attentions = (torch.mean(torch.stack([op.attentions[-1] for op in ops]), dim=0),)
    pool_attn = torch.mean(torch.stack([op.pool_attn for op in ops]), dim=0)
    return features, ViTPoolModelOutput(attentions=attentions, pool_attn=pool_attn)


def run(run_config):
    run_name = f"{run_config['name']}_channel_comb_{run_config['channel_combination']}_norm_{run_config['normalization']}"
    inference_folder = f"{run_config['inference_folder']}/{run_name}"
    print(f"Saving inference results to {inference_folder}")
    attn_folder = f"{inference_folder}/attention_images"
    temp_folder = f"{inference_folder}/temp"
    inference.reset_dir(attn_folder)
    inference.reset_dir(temp_folder)

    print(f"Starting inference for {run_name}")
    dataset = AllenCellDataset(run_config["crops_folder"], run_config["metadata_file"],
                               run_config["channel_combination"], run_config["normalization"])
    loader = inference.make_loader(dataset, batch_size=16)
    model = inference.get_model(run_config["name"], run_config["model"])
    attention = (dict(folder=attn_folder, colors=["cyan", "pink", "white"], normalization="allencell")
                 if run_config.get("save_attention", False) else None)
    all_metadata, all_features = inference.extract_features(model, loader, forward=encode_views,
                                                            attention=attention, temp_dir=temp_folder)
    torch.save((all_metadata, all_features), f"{inference_folder}/all_features.pth")
    shutil.rmtree(temp_folder, ignore_errors=True)


def main(config_path):
    for run_config in config.load_yaml(config_path):
        run(run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="path to the configuration file (a list of runs)")
    main(parser.parse_args().config)
