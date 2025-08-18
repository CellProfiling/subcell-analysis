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

UNIQUE_CATS = np.array(
    [
        cat
        for cat in pd.read_csv("annotations/location_group_mapping.tsv", sep="\t")[
            "Original annotation"
        ]
        .unique()
        .tolist()
        if cat
        not in ["Cleavage furrow", "Midbody ring", "Rods & Rings", "Microtubule ends"]
    ]
    + ["Negative"]
)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class MLPClassifier(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            # nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, out_units),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        out = self.classifier(feat)
        return out, feat


def balanced_weights(train_y):
    freq = train_y.sum(axis=0) / train_y.shape[0]
    weights_per_class = 1 / freq

    weights = []
    for idx in range(train_y.shape[0]):
        idx_label = train_y[idx].bool()
        weights.append(weights_per_class[idx_label].max())
    weights = torch.tensor(weights)
    return weights


def train_mlp(train_x, train_y, val_x, val_y, device, unique_cats, save_folder):
    model_path = f"{save_folder}/mlp_best_map_final.pth"

    train_dataloader = DataLoader(
        TensorDataset(
            train_x.float().to(device),
            train_y.float().to(device),
        ),
        batch_size=8192,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        TensorDataset(
            val_x.float().to(device),
            val_y.float().to(device),
        ),
        batch_size=8192,
    )

    model = MLPClassifier(train_x.shape[1], len(unique_cats))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5
    )

    criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    best_val_map = 0
    epochs_without_improvement = 0

    for epoch in range(1, 201):
        if epochs_without_improvement >= 20:
            print("No improvement after 20 epochs, stopping")
            break

        _train_epoch(model, train_dataloader, optimizer, criterion, epoch)
        val_map = _val_epoch(model, val_dataloader, criterion, epoch)
        lr_scheduler.step(val_map)
        print(f"Epoch: {epoch} LR: {optimizer.param_groups[0]['lr']}", flush=True)

        if val_map >= best_val_map:
            best_val_map = val_map
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{save_folder}/mlp_best_map.pth")
        else:
            epochs_without_improvement += 1
            print(
                f"Epochs without improvement: {epochs_without_improvement}, Best Val MAP: {best_val_map}",
                flush=True,
            )

    model.load_state_dict(
        torch.load(f"{save_folder}/mlp_best_map.pth", weights_only=True)
    )
    torch.save(model.state_dict(), model_path)
    return model


def _train_epoch(model, train_dataloader, optimizer, criterion, epoch):
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y)
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
            loss = criterion(out, y)
            out_onehot = F.sigmoid(out)
            y_pred.append(out_onehot.cpu())
            y_true.append(y.cpu())

            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()

        mean_avg_precision = average_precision_score(y_true, y_pred, average="macro")

    print(
        f"Epoch: {epoch} Val Loss: {val_loss:.5f} Mean Avg Precision: {mean_avg_precision:.5f}",
        flush=True,
    )
    return mean_avg_precision


def _test_epoch(model, test_dataloader, unique_cats):
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        y_feat = []
        for i, (x, y) in enumerate(test_dataloader):
            out, feat = model(x)
            out_sigmoid = F.sigmoid(out)
            y_pred.append(out_sigmoid.cpu())
            y_true.append(y.cpu())
            y_feat.append(feat.cpu())

        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()
        y_feat = torch.cat(y_feat).cpu().numpy()

        mean_avg_precision = average_precision_score(y_true, y_pred, average="macro")

    print(f"Mean Avg Precision: {mean_avg_precision:.5f}")

    df_pred = pd.DataFrame(y_pred, columns=unique_cats)
    df_true = pd.DataFrame(y_true, columns=unique_cats)

    df = pd.merge(
        df_true,
        df_pred,
        suffixes=("_true", "_pred"),
        left_index=True,
        right_index=True,
    )
    df.loc[:, [f"feat_{i}" for i in range(y_feat.shape[1])]] = y_feat
    return df


def eval_model(x, y, unique_cats, model, seed, device):
    dataloader = DataLoader(
        TensorDataset(
            x.float().to(device),
            y.float().to(device),
        ),
        batch_size=16384,
    )
    df_res = _test_epoch(model, dataloader, unique_cats)
    df_res = df_res.reindex(sorted(df_res.columns), axis=1)
    df_res.loc[:, "Seed"] = seed
    return df_res


def get_multilabel_df(df_true, df_pred):
    cols = df_true.columns

    avg_precisions = []
    aucs = []
    all_categories = []
    all_counts = []
    for cat in cols:
        if len(np.unique(df_true[cat])) != 2:
            continue
        avg_precision = average_precision_score(df_true[cat], df_pred[cat])
        avg_precisions.append(avg_precision)
        all_categories.append(cat)
        all_counts.append(df_true[cat].sum())
        auc = roc_auc_score(df_true[cat], df_pred[cat])
        aucs.append(auc)

    avg_precisions.append(average_precision_score(df_true.values, df_pred.values))
    aucs.append(roc_auc_score(df_true.values, df_pred.values))
    all_categories.append("Overall")
    all_counts.append(len(df_true))
    df_multilabel = (
        pd.DataFrame(
            {
                "Category": all_categories,
                "Average Precision": avg_precisions,
                "AUC": aucs,
                "Count": all_counts,
            }
        )
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )
    return df_multilabel


def plot_multilabel_metrics(
    df, metric="Average Precision", label="valid", save_folder="./"
):
    n_cats = len(df)
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, figsize=(16, 10))
    sns.barplot(
        x="Category",
        y=metric,
        hue="Category",
        palette=sns.color_palette(cc.glasbey_dark, n_cats),
        data=df,
        ax=ax,
        orient="v",
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.savefig(
        f"{save_folder}/{label}_{metric}.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()


def get_metrics(save_folder, df_test, tag="test", unique_cats=UNIQUE_CATS):
    df_true = df_test[[col + "_true" for col in unique_cats]]
    df_true = df_true.rename(
        columns={col: col.replace("_true", "") for col in df_true.columns}
    )
    df_pred = df_test[[col + "_pred" for col in unique_cats]]
    df_pred = df_pred.rename(
        columns={col: col.replace("_pred", "") for col in df_pred.columns}
    )

    non_zero_cats = [col for col in unique_cats if df_true[col].sum() > 0]
    df_true = df_true[non_zero_cats]
    df_pred = df_pred[non_zero_cats]

    df_true_label = df_true.values.argmax(1)
    df_pred_label = df_pred.values.argmax(1)

    conf_mat = confusion_matrix(df_true_label, df_pred_label, normalize="true")
    conf_mat_df = pd.DataFrame(conf_mat, index=non_zero_cats, columns=non_zero_cats)
    conf_mat_df.to_csv(f"{save_folder}/{tag}_confusion_matrix.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(1, figsize=(16, 12))
    sns.heatmap(
        conf_mat_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=conf_mat_df.columns,
        yticklabels=conf_mat_df.index,
        linewidths=0.5,
        linecolor="black",
        square=True,
        # annot_kws={"size": 25},
        ax=ax,
    )
    plt.title(f"{tag} Confusion Matrix")
    plt.savefig(
        f"{save_folder}/{tag}_confusion_matrix.png",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()

    cls_rep = classification_report(
        df_true_label,
        df_pred_label,
        output_dict=True,
        target_names=non_zero_cats,
        digits=4,
    )
    cls_rep_df = pd.DataFrame(cls_rep).T
    cls_rep_df.to_csv(f"{save_folder}/{tag}_classification_report.csv")

    df_multilabel = get_multilabel_df(df_true, df_pred)
    df_multilabel["Coverage Error"] = coverage_error(df_true, df_pred)
    df_multilabel["Label Ranking Average Precision"] = (
        label_ranking_average_precision_score(df_true, df_pred)
    )
    df_multilabel["Micro Average Precision"] = average_precision_score(
        df_true, df_pred, average="micro"
    )
    df_multilabel.to_csv(f"{save_folder}/{tag}_metrics.csv", index=False)

    plot_multilabel_metrics(
        df_multilabel,
        metric="Average Precision",
        label=tag,
        save_folder=save_folder,
    )
    plot_multilabel_metrics(
        df_multilabel, metric="AUC", label=tag, save_folder=save_folder
    )


def plot_umap(save_folder, df, tag, unique_cats):
    feats = df[[col for col in df.columns if "feat" in col]].values
    umap_feat = UMAP(verbose=2).fit_transform(feats)
    umap_df = pd.DataFrame(
        {
            "x": umap_feat[:, 0],
            "y": umap_feat[:, 1],
            "Category": df[[col + "_true" for col in unique_cats]].idxmax(axis=1),
        }
    )
    umap_df["Category"] = umap_df["Category"].apply(lambda x: x.replace("_true", ""))
    umap_df["Category"] = pd.Categorical(
        umap_df["Category"], categories=unique_cats, ordered=True
    )

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(1, figsize=(16, 12))
    sns.scatterplot(
        x="x",
        y="y",
        hue="Category",
        s=20,
        alpha=0.5,
        palette="tab10",
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
    plt.savefig(f"{save_folder}/{tag}_umap.png", dpi=500, bbox_inches="tight")
    plt.close()
