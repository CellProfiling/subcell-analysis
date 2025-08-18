import time
import torch
import itertools
import os

# import random
import numpy as np
import pandas as pd

# import math
import torch.nn as nn
import torch.optim as optim

# from torchvision import transforms
from torch.utils.data import DataLoader
from .mirage_model import *

# from .model import *
from .loss import *

# import wandb

# globals
device = torch.device("cpu")
config = None


def mirage_fit_predict(
    config,
    wandb=None,
    test_subset=[],
    index_names=[],
):
    # get device
    device = torch.device(
        f"cuda:{config.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
    )
    globals()["device"] = device
    globals()["config"] = config

    resultsdir = f"{config.workdir}"
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir, exist_ok=True)

    config["checkpoints_dir"] = f"{resultsdir}/.checkpoints"
    if not os.path.exists(config["checkpoints_dir"]):
        os.makedirs(config["checkpoints_dir"], exist_ok=True)

    # create model, optimizer, trainloader
    if config.aligned:
        dataset = Aligned_Dataset(config)
    else:
        dataset = Unaligned_Dataset(config)

    print("Dataset size: ", len(dataset))
    config.dims = dataset.dims

    model = CycleGANModel(config)
    model.setup(config)
    print("Training model on:", model.device)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=int(config.num_threads),
    )

    best_loss = float("inf")

    # Define the custom x axis metric
    # https://docs.wandb.ai/guides/track/log/customize-logging-axes
    if wandb is not None:
        wandb.define_metric("total_iters")
        # Define which metrics to plot against that x-axis
        wandb.define_metric("Loss", step_metric="total_iters")
        wandb.define_metric(f"lr", step_metric="total_iters")
        for modality in config.modalities:
            wandb.define_metric(f"short_cycle{modality}", step_metric="total_iters")
            wandb.define_metric(f"latent_cycle{modality}", step_metric="total_iters")
            wandb.define_metric(f"G_{modality}", step_metric="total_iters")
            wandb.define_metric(f"D_{modality}", step_metric="total_iters")
    total_iters = 0  # the total number of training iterations
    # Training loop
    for epoch in range(config.n_epochs + config.n_epochs_decay):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        lr = (
            model.update_learning_rate()
        )  # update learning rates in the beginning of every epoch.

        # loop over all batches
        for step, data in enumerate(train_loader):
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1
            epoch_iter += 1
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()
            losses = model.get_current_losses()
            losses["Loss"] = sum(losses.values())
            elapsed = time.time() - epoch_start_time
            # log every 100 iterations
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

    # SAVE FINAL RESULTS
    model.eval()
    with torch.no_grad():
        for i, modality in enumerate(config.modalities):
            enc = getattr(model, f"enc_{modality}")
            d_modality = dataset.dataframes[i].values
            index_modality = list(dataset.dataframes[i].index.values)
            input_modality = torch.from_numpy(d_modality).float().to(device)
            latent_modality = enc(input_modality)
            df_modality = pd.DataFrame(
                latent_modality.detach().cpu().numpy(), index=index_modality
            )
            df_modality.to_csv(f"{resultsdir}/latent_{modality}.txt")

    return model


def gan_predict(
    config,
    wandb=None,
    test_subset=[],
    index_names=[],
):
    # get device
    device = torch.device(
        f"cuda:{config.gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
    )
    globals()["device"] = device
    globals()["config"] = config
    optimizers = []

    resultsdir = f"{config.workdir}/{config.run_name}"
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir, exist_ok=True)

    config.checkpoints_dir = f"{resultsdir}/.checkpoints"
    if not os.path.exists(config.checkpoints_dir):
        os.makedirs(config.checkpoints_dir, exist_ok=True)

    # create model, optimizer, trainloader
    dataset = Unaligned_Dataset(config)
    config.dims = dataset.dims

    model = CycleGANModel(config)
    model.setup(config)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        # num_workers=int(config.num_threads),
    )
    model.load_networks(config.n_epochs)
    return model, dataset
