"""Train MIRAGE co-embeddings of image and sequence embeddings (Extended Data Fig. 7b).

For every row of ``methods_info.csv`` and every ``Mirage_<A>_<B>_Embed`` column
(here ``Mirage_Image_Seq_Embed``), one MIRAGE model is trained on the embeddings in
``<A>_Embed`` and ``<B>_Embed`` (``Image_Embed``, ``Seq_Embed``). The latent
embedding of each modality is written next to the path in the ``Mirage_*`` column
as ``latent_<modality>.txt``; rows whose latent files already exist are skipped.
``subcell.multimodal.interaction_enrichment`` evaluates them.

    python -m subcell.multimodal.mirage [--wandb]

Runs on a GPU if one is available. A model that fails to train is logged and
reported at the end with a non-zero exit status. ``--wandb`` logs the losses to the
``MIRAGE-GAN`` Weights & Biases project.
"""

from __future__ import annotations

import argparse
import os
import traceback
from types import SimpleNamespace

import numpy as np

from subcell.io.metadata import read_methods_csv
from subcell.models import mirage
from subcell.runlog import log

DEFAULT_CONFIG = {
    # model
    "batch_size": 32,
    "latent_dim": 128,  # output dim of embedding
    "hidden_dim": 512,
    # training
    "verbose": True,
    "n_epochs": 100,  # number of epochs with the initial learning rate
    "epoch_count": 1,
    "n_epochs_decay": 100,  # number of epochs to linearly decay learning rate to zero
    "gpu_ids": [0],
    "num_threads": 4,
    # initializers
    "init_type": "xavier",  # normal, xavier, kaiming, orthogonal
    # loss weights
    "lambda_latent_cycle": 10.0,
    "lambda_short_cycle": 10.0,
    # gradient penalty
    "lambda_gp": 0.1,  # gradient penalty lambda https://arxiv.org/pdf/1704.00028
    "gp_constant": 0.0,  # gradient penalty constant https://arxiv.org/pdf/1704.00028
    "gp_type": "real",  # [real | fake | mixed] where the penalty is evaluated
    "pool_size": 50,  # size of the buffer that stores previously generated samples
    "gan_mode": "lsgan",  # [vanilla | lsgan | wgangp]
    # Adam optimizer
    "beta1": 0.5,  # momentum term of adam
    "lr": 0.0002,  # initial learning rate for adam
    "lr_policy": "linear",  # [linear | step | plateau | cosine]
}


def train_mirage_models(row, wandb=None):
    """Train a model for each ``Mirage_*`` combination of one ``methods_info.csv`` row.

    Returns the names of the combinations whose training failed.
    """
    failed = []
    print(f"Training MIRAGE model for {row['Method_Name']}...")
    combinations = [
        col.replace("Mirage_", "").replace("_Embed", "")
        for col in row.index
        if "Mirage" in col
    ]
    for combination in combinations:
        print(f"Processing combination: {combination}")
        modalities = combination.split("_")
        run_config = {
            **DEFAULT_CONFIG,
            "input_files": [f"{row[f'{x}_Embed']}" for x in modalities],
            "workdir": os.path.dirname(row[f"Mirage_{combination}_Embed"]),
            "modalities": [modality.lower() for modality in modalities],
        }

        if np.all(
            [
                os.path.exists(f"{run_config['workdir']}/latent_{modality}.txt")
                for modality in run_config["modalities"]
            ]
        ):
            print(f"Data already exists for {combination}, skipping...")
            continue

        try:
            if wandb is not None:
                wandb.init(
                    project="MIRAGE-GAN",
                    name=f"{row['Method_Name']}_{combination}",
                    config=run_config,
                )
                mirage.mirage_fit_predict(config=wandb.config, wandb=wandb)
            else:
                mirage.mirage_fit_predict(config=SimpleNamespace(**run_config))
        except Exception:
            log(f"training failed for {row['Method_Name']}_{combination}:\n{traceback.format_exc()}")
            failed.append(f"{row['Method_Name']}_{combination}")
        finally:
            if wandb is not None:
                wandb.finish()
    return failed


def main(use_wandb=False):
    wandb = None
    if use_wandb:
        import wandb

        wandb.login()
        wandb.finish()

    failed = []
    for _, row in read_methods_csv().iterrows():
        failed += train_mirage_models(row, wandb)
    if failed:
        raise SystemExit(f"MIRAGE training failed for {len(failed)} model(s): {', '.join(failed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wandb", action="store_true", help="log training to Weights & Biases")
    main(ap.parse_args().wandb)
