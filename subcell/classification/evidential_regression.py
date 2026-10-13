"""Deep evidential regression of FUCCI pseudotime on frozen SubCell features.

A two-layer MLP with a Normal-Inverse-Gamma output head (Amini et al., 2020)
predicts pseudotime and its aleatoric and epistemic uncertainty (Fig. 2l). Called
by ``subcell.classification.cellcycle_fucci``; no entry point of its own.

Each fold/seed folder holds ``mlp_best_l1.pth`` (the epoch with the lowest
validation RMSE) and ``mlp_best_l1_final.pth`` (the same weights, as loaded); the
``l1`` in those names is historical, the criterion is RMSE.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        mu = F.sigmoid(mu)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta


def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


def evidential_regression(dist_params, y, lamb=1.0):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)


class MLPRegressor(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
        )

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.5),
            NormalInvGamma(256, 1),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        out = self.regressor(feat)
        return out, feat


def train_mlp_reg(train_x, train_y, val_x, val_y, device, save_folder):
    train_dataloader = DataLoader(
        TensorDataset(
            train_x.float().to(device),
            train_y.float().to(device).unsqueeze(1),
        ),
        batch_size=8192,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        TensorDataset(
            val_x.float().to(device),
            val_y.float().to(device).unsqueeze(1),
        ),
        batch_size=8192,
    )

    model = MLPRegressor(train_x.shape[1], 1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    criterion = evidential_regression

    best_val_rmse = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, 201):
        if epochs_without_improvement >= 20:
            print("No improvement after 20 epochs, stopping")
            break

        _train_epoch(model, train_dataloader, optimizer, criterion)
        val_rmse = _val_epoch(model, val_dataloader, criterion, epoch)
        lr_scheduler.step(val_rmse)

        if val_rmse <= best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{save_folder}/mlp_best_l1.pth")
        else:
            epochs_without_improvement += 1
            print(
                f"Epochs without improvement: {epochs_without_improvement}, Best Val RMSE: {best_val_rmse}",
                flush=True,
            )

    model.load_state_dict(
        torch.load(f"{save_folder}/mlp_best_l1.pth", weights_only=True)
    )
    torch.save(model.state_dict(), f"{save_folder}/mlp_best_l1_final.pth")
    return model


def _train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    for x, y in train_dataloader:
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y, lamb=1e-2)
        loss.backward()
        optimizer.step()


def _val_epoch(model, val_dataloader, criterion, epoch):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        val_loss = 0
        for i, (x, y) in enumerate(val_dataloader):
            out, _ = model(x)
            loss = criterion(out, y, lamb=1e-2)
            y_pred.append(out[0].cpu())
            y_true.append(y.cpu())

            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()

        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    print(
        f"Epoch: {epoch} Val Loss: {val_loss:.5f} Val RMSE: {rmse:.5f}",
        flush=True,
    )
    return rmse


def _test_epoch(model, test_dataloader):
    with torch.no_grad():
        model.eval()

        y_true = []
        y_pred = []
        y_aleatoric = []
        y_var = []
        y_feat = []
        for i, (x, y) in enumerate(test_dataloader):
            out, feat = model(x)
            mu, v, alpha, beta = (d.squeeze() for d in out)
            aleatoric = torch.sqrt(beta / (alpha - 1))
            var = torch.sqrt(beta / (v * (alpha - 1)))
            y_true.append(y.cpu())
            y_pred.append(mu.squeeze(-1).cpu())
            y_aleatoric.append(aleatoric.cpu())
            y_var.append(var.cpu())
            y_feat.append(feat.cpu())

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_aleatoric = torch.cat(y_aleatoric).cpu().numpy()
        y_var = torch.cat(y_var).cpu().numpy()
        y_feat = torch.cat(y_feat).cpu().numpy()

    df = pd.DataFrame(
        {
            "pseudotime_true": y_true,
            "pseudotime_pred": y_pred,
            "aleatoric": y_aleatoric,
            "var": y_var,
        }
    )
    df.loc[:, [f"feat_{i}" for i in range(y_feat.shape[1])]] = y_feat

    return df


def eval_model_reg(x, y, model, seed, device):
    dataloader = DataLoader(
        TensorDataset(x.float().to(device), y.float().to(device)), batch_size=16384
    )
    df_res = _test_epoch(model, dataloader)
    df_res.loc[:, "Seed"] = seed
    return df_res
