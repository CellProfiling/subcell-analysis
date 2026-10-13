"""MIRAGE (Nasser et al., Cell Syst. 16, 101444 (2025)): adversarial co-embedding of
protein modalities, used as the alternative to MuSIC in Extended Data Fig. 7b.

``mirage_fit_predict`` trains one model on the files listed in ``config.input_files``
and writes the latent embedding of each modality to ``<workdir>/latent_<modality>.txt``.
Training settings are in ``subcell.multimodal.mirage``.
"""

# Adapted from pytorch-CycleGAN-and-pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park, BSD 2-Clause licence.

import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .mirage_model import AlignedDataset, MirageModel


def mirage_fit_predict(config, wandb=None):
    """Train MIRAGE and write the latent embeddings.

    Parameters:
        config -- settings with attribute access (a ``wandb.config`` or a
                  ``types.SimpleNamespace``): see ``subcell.multimodal.mirage``
        wandb  -- optional ``wandb`` module or run; losses are logged to it every 100 iterations
    """
    resultsdir = f"{config.workdir}"
    os.makedirs(resultsdir, exist_ok=True)

    config.checkpoints_dir = f"{resultsdir}/.checkpoints"
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    dataset = AlignedDataset(config)

    print("Dataset size: ", len(dataset))
    config.dims = dataset.dims

    model = MirageModel(config)
    model.setup(config)
    print("Training model on:", model.device)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=int(config.num_threads),
    )

    # Plot the losses against the iteration count
    # https://docs.wandb.ai/guides/track/log/customize-logging-axes
    if wandb is not None:
        wandb.define_metric("total_iters")
        wandb.define_metric("Loss", step_metric="total_iters")
        wandb.define_metric("lr", step_metric="total_iters")
        for modality in config.modalities:
            wandb.define_metric(f"short_cycle_{modality}", step_metric="total_iters")
            wandb.define_metric(f"latent_cycle_{modality}", step_metric="total_iters")
            wandb.define_metric(f"G_{modality}", step_metric="total_iters")
            wandb.define_metric(f"D_{modality}", step_metric="total_iters")
    total_iters = 0
    for epoch in range(config.n_epochs + config.n_epochs_decay):
        epoch_start_time = time.time()
        # update learning rates in the beginning of every epoch.
        lr = model.update_learning_rate()

        for data in train_loader:
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()
            losses = model.get_current_losses()
            losses["Loss"] = sum(losses.values())
            elapsed = time.time() - epoch_start_time
            if total_iters % 100 == 0:
                print("-" * 89)
                print(
                    f"""| epoch {1+epoch:3d} | time: {elapsed:5.2f}s | loss {losses["Loss"]:5.2f}"""
                )
                print("-" * 89)
                losses["total_iters"] = total_iters
                losses["lr"] = lr
                if wandb is not None:
                    wandb.log(losses)
    model.save_networks(config.n_epochs)

    # latent embedding of each modality
    model.eval()
    with torch.no_grad():
        for i, modality in enumerate(config.modalities):
            enc = getattr(model, f"enc_{modality}")
            d_modality = dataset.dataframes[i].values
            index_modality = list(dataset.dataframes[i].index.values)
            input_modality = torch.from_numpy(d_modality).float().to(model.device)
            latent_modality = enc(input_modality)
            df_modality = pd.DataFrame(
                latent_modality.detach().cpu().numpy(), index=index_modality
            )
            df_modality.to_csv(f"{resultsdir}/latent_{modality}.txt")

    return model
