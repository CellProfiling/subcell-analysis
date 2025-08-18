import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    label_ranking_average_precision_score,
)
from torch.nn import functional as F
from tqdm import tqdm

from models.mlp import MLPClassifier, SigmoidFocalLoss, GatedAttentionClassifier


def set_random_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimal_threshold(y_true, y_pred):
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0
    best_acc = 0
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        acc = f1_score(y_true, y_pred_binary, average="macro")
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold


def compute_metrics(y_true, y_pred):
    best_threshold = get_optimal_threshold(y_true, y_pred)
    y_pred_binary = (y_pred > best_threshold).astype(int)

    return {
        "acc": f1_score(y_true, y_pred_binary, average="samples"),
        "macro_f1": f1_score(y_true, y_pred_binary, average="macro"),
        "micro_f1": f1_score(y_true, y_pred_binary, average="micro"),
        "macro_ap": average_precision_score(y_true, y_pred, average="macro"),
        "micro_ap": average_precision_score(y_true, y_pred, average="micro"),
        "mlrap": label_ranking_average_precision_score(y_true, y_pred),
    }


def train_single_model(train_loader, val_dataset, model, y_true):
    criterion = SigmoidFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_map = 0
    epochs_without_improvement = 0
    best_model = None

    model.to(val_dataset.tensors[0].device)
    for epoch in range(200):
        if epochs_without_improvement >= 20:
            print(
                f"No improvement after 20 epochs, stopping at epoch {epoch}, best map: {best_map}",
                flush=True,
            )
            break

        model.train()
        for _, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        y_pred = F.sigmoid(model(val_dataset.tensors[0])).detach().cpu().numpy()
        val_map = average_precision_score(y_true, y_pred, average="macro")

        if val_map > best_map:
            best_map = val_map
            best_model = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        lr_scheduler.step(best_map)

    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        y_pred = F.sigmoid(model(val_dataset.tensors[0])).detach().cpu().numpy()

    return y_pred, best_map


def train_model(train_dataset, val_dataset, n_layer, hidden_dim, dropout):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=True, drop_last=True
    )

    y_true = val_dataset.tensors[1].cpu().numpy()
    all_y_pred = []
    all_val_map = []

    for random_seed in range(10):
        set_random_seed(random_seed)
        model = MLPClassifier(
            input_dim=train_loader.dataset.tensors[0].shape[1],
            output_dim=train_loader.dataset.tensors[1].shape[1],
            hidden_dim=hidden_dim,
            n_layers=n_layer,
            dropout=dropout,
        )

        y_pred, best_map = train_single_model(train_loader, val_dataset, model, y_true)

        all_val_map.append(best_map)
        all_y_pred.append(y_pred)

    y_pred = np.mean(all_y_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred)

    print(
        f"Hidden Dim: {hidden_dim}, Dropout: {dropout}, Number of Layers: {n_layer}",
        f"accuracy: {metrics['acc']} micro_AP: {metrics['micro_ap']}, macro_AP: {metrics['macro_ap']}",
        flush=True,
    )

    return (
        n_layer,
        hidden_dim,
        dropout,
        metrics["acc"],
        metrics["macro_f1"],
        metrics["micro_f1"],
        metrics["macro_ap"],
        metrics["micro_ap"],
        metrics["mlrap"],
    )


def cross_val_mlp(X_train, y_train, X_test, y_test, parameters):
    n_jobs = -1

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    all_hyperparam_combinations = []

    for hidden_dim in parameters["hidden_dim"]:
        for dropout in parameters["dropout"]:
            for n_layer in parameters["n_layers"]:
                all_hyperparam_combinations.append((n_layer, hidden_dim, dropout))

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_model)(train_dataset, val_dataset, *params)
        for params in tqdm(all_hyperparam_combinations)
    )

    results_df = pd.DataFrame(
        results,
        columns=[
            "n_layers",
            "hidden_dim",
            "dropout",
            "acc",
            "macro_f1",
            "micro_f1",
            "macro_ap",
            "micro_ap",
            "mlrap",
        ],
    )
    return results_df
