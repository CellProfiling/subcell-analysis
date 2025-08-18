import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    coverage_error,
    label_ranking_average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from umap import UMAP
from sklearn.metrics import r2_score


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
    model_path = f"{save_folder}/mlp_best_l1_final.pth"

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
    # criterion = nn.L1Loss()

    best_val_l1 = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, 201):
        if epochs_without_improvement >= 20:
            print("No improvement after 20 epochs, stopping")
            break

        _train_epoch(model, train_dataloader, optimizer, criterion, epoch)
        val_l1 = _val_epoch(model, val_dataloader, criterion, epoch)
        lr_scheduler.step(val_l1)
        print(f"Epoch: {epoch} LR: {optimizer.param_groups[0]['lr']}", flush=True)

        if val_l1 <= best_val_l1:
            best_val_l1 = val_l1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{save_folder}/mlp_best_l1.pth")
        else:
            epochs_without_improvement += 1
            print(
                f"Epochs without improvement: {epochs_without_improvement}, Best Val L1: {best_val_l1}",
                flush=True,
            )

    model.load_state_dict(
        torch.load(f"{save_folder}/mlp_best_l1.pth", weights_only=True)
    )
    torch.save(model.state_dict(), model_path)
    return model


def _train_epoch(model, train_dataloader, optimizer, criterion, epoch):
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y, lamb=1e-2)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch} Batch: {i+1} Loss: {loss.item():.5f}", flush=True)


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

        mean_l1 = np.sqrt(np.mean((y_pred - y_true) ** 2))
        # np.mean((y_pred - y_true) ** 2)  # np.abs(y_true - y_pred).mean() # np.sqrt(np.mean((y_pred - y_true) ** 2))

    print(
        f"Epoch: {epoch} Val Loss: {val_loss:.5f} Mean L1: {mean_l1:.5f}",
        flush=True,
    )
    return mean_l1


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


def plot_umap_reg(save_folder, df):
    feats = df[[col for col in df.columns if "feat" in col]].values
    umap_feat = UMAP().fit_transform(feats)
    umap_df = pd.DataFrame(
        {
            "x": umap_feat[:, 0],
            "y": umap_feat[:, 1],
            "PseudoTime": df["pseudotime"].values,
        }
    )
    fig, ax = plt.subplots(1, figsize=(16, 12))
    sns.scatterplot(
        x="x",
        y="y",
        hue="PseudoTime",
        s=10,
        alpha=0.5,
        palette="crest",
        data=umap_df,
    )
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
    lgd = ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=2,
    )
    plt.savefig(f"{save_folder}/umap.png", dpi=500, bbox_inches="tight")
    plt.close()


def plot_pred(save_folder, df):
    r2 = r2_score(df["pseudotime"], df["pseudotime_pred"])
    sns.lmplot(data=df, x="pseudotime", y="pseudotime_pred", scatter=False, aspect=1)
    sns.scatterplot(
        data=df, x="pseudotime", y="pseudotime_pred", s=1, alpha=0.2, color="black"
    )
    sns.lineplot(
        x=np.arange(0, 1.1, 0.1), y=np.arange(0, 1.1, 0.1), color="red", label="y=x"
    )
    plt.title(
        f"R2: {r2:.3f} | RMSE: {np.sqrt(np.mean((df['pseudotime'] - df['pseudotime_pred']) ** 2)):.3f}"
        f" | MAE: {np.abs(df['pseudotime'] - df['pseudotime_pred']).mean():.3f}",
        fontsize=10,
        fontweight="bold",
    )
    plt.savefig(f"{save_folder}/val_pred.png", dpi=500, bbox_inches="tight")
    plt.close()

    sns.lmplot(data=df, x="pseudotime", y="aleatoric", scatter=False)
    sns.scatterplot(
        data=df, x="pseudotime", y="aleatoric", s=1, alpha=0.2, color="black"
    )
    plt.savefig(f"{save_folder}/val_aleatoric.png", dpi=500, bbox_inches="tight")
    plt.close()

    sns.lmplot(data=df, x="pseudotime", y="var", scatter=False)
    sns.scatterplot(data=df, x="pseudotime", y="var", s=1, alpha=0.2, color="black")
    plt.ylim(0, np.percentile(df["var"], 99))
    plt.savefig(f"{save_folder}/val_var.png", dpi=500, bbox_inches="tight")
    plt.close()
