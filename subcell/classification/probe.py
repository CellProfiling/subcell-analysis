"""Two-layer MLP probe on frozen SubCell features, and its evaluation.

The probe behind the localization (Fig. 2a-c, f; Supplementary Table 9) and
cell-cycle stage (Fig. 2g-k) benchmarks: ``TwoLayerProbe`` trained with sigmoid
focal loss, AdamW and early stopping on validation macro average precision
(``train_mlp``), then scored per class (``get_metrics``) and embedded
(``plot_umap``). The benchmark drivers (``localization``, ``cellcycle_allencell``,
``cellcycle_fucci``, ``yeast``) call these; this module has no entry point.

Each seed's folder holds ``mlp_best_map.pth`` (best epoch), ``mlp_best_map_final.pth``
(the same weights, as loaded), ``{tag}_preds.csv`` (true labels, sigmoid outputs and
the 256-d ``feat_*`` embedding per sample) and the ``{tag}_*`` metric tables and plots.
"""

import os

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    coverage_error,
    label_ranking_average_precision_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from umap import UMAP

from subcell.utils.losses import SigmoidFocalLoss
from subcell.utils.seed import set_random_seed


class TwoLayerProbe(nn.Module):
    """512-unit hidden layer, 256-d embedding, linear classifier; returns
    ``(logits, embedding)``.

    ``relu_feature`` adds a ReLU after the embedding (the HPA localization
    probes); the parameters are the same either way.
    """

    def __init__(self, in_features, out_units, relu_feature=False):
        super().__init__()

        layers = [
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
        ]
        if relu_feature:
            layers.append(nn.ReLU(inplace=True))
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, out_units),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        out = self.classifier(feat)
        return out, feat


def _loader(x, y, device, batch_size, shuffle=False):
    dataset = TensorDataset(x.float().to(device), y.float().to(device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    train_x, train_y, val_x, val_y, device, unique_cats, save_folder, relu_feature=False
):
    """Train a ``TwoLayerProbe`` and return it with its best-validation weights."""
    train_dataloader = _loader(train_x, train_y, device, 8192, shuffle=True)
    val_dataloader = _loader(val_x, val_y, device, 8192)

    model = TwoLayerProbe(train_x.shape[1], len(unique_cats), relu_feature)
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

        _train_epoch(model, train_dataloader, optimizer, criterion)
        val_map = _val_epoch(model, val_dataloader, criterion, epoch)
        lr_scheduler.step(val_map)

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
    torch.save(model.state_dict(), f"{save_folder}/mlp_best_map_final.pth")
    return model


def _train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    for x, y in train_dataloader:
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


def _predict(model, dataloader):
    """Sigmoid outputs and targets (and embeddings) over a dataloader."""
    model.eval()
    y_pred, y_true, y_feat = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            out, feat = model(x)
            y_pred.append(torch.sigmoid(out).cpu())
            y_true.append(y.cpu())
            y_feat.append(feat.cpu())
    return tuple(torch.cat(a).numpy() for a in (y_pred, y_true, y_feat))


def _val_epoch(model, val_dataloader, criterion, epoch):
    model.eval()
    val_loss = 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in val_dataloader:
            out, _ = model(x)
            val_loss += criterion(out, y).item()
            y_pred.append(torch.sigmoid(out).cpu())
            y_true.append(y.cpu())
    val_loss /= len(val_dataloader)
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    mean_avg_precision = average_precision_score(y_true, y_pred, average="macro")

    print(
        f"Epoch: {epoch} Val Loss: {val_loss:.5f} Mean Avg Precision: {mean_avg_precision:.5f}",
        flush=True,
    )
    return mean_avg_precision


def eval_model(x, y, unique_cats, model, seed, device):
    """Predictions of ``model`` on ``(x, y)``: one row per sample with
    ``{cat}_true``, ``{cat}_pred``, ``feat_*`` and ``Seed`` columns."""
    y_pred, y_true, y_feat = _predict(model, _loader(x, y, device, 16384))
    print(f"Mean Avg Precision: {average_precision_score(y_true, y_pred, average='macro'):.5f}")

    df = pd.concat(
        [
            pd.DataFrame(y_true, columns=unique_cats).add_suffix("_true"),
            pd.DataFrame(y_pred, columns=unique_cats).add_suffix("_pred"),
            pd.DataFrame(y_feat, columns=[f"feat_{i}" for i in range(y_feat.shape[1])]),
        ],
        axis=1,
    )
    df = df.reindex(sorted(df.columns), axis=1)
    df["Seed"] = seed
    return df


def fit_or_load_probe(folder, seed, device, train, eval_sets, unique_cats, relu_feature=False):
    """Seed, train a probe on ``train = (x, y)`` and score each ``{tag: (x, y)}``
    in ``eval_sets``, writing ``{folder}/{tag}_preds.csv``.

    When the last tag's predictions are already on disk, every tag's predictions
    are read back instead. Returns ``{tag: predictions}``.
    """
    os.makedirs(folder, exist_ok=True)
    set_random_seed(seed, deterministic=True)
    if os.path.isfile(f"{folder}/{list(eval_sets)[-1]}_preds.csv"):
        return {tag: pd.read_csv(f"{folder}/{tag}_preds.csv") for tag in eval_sets}

    val_tag = next(iter(eval_sets))
    model = train_mlp(*train, *eval_sets[val_tag], device, unique_cats, folder, relu_feature)
    results = {}
    for tag, (x, y) in eval_sets.items():
        results[tag] = eval_model(x, y, unique_cats, model, seed=seed, device=device)
        results[tag].to_csv(f"{folder}/{tag}_preds.csv", index=False)
    return results


def probe_seeds(save_folder, device, train, eval_sets, unique_cats, seeds=range(10), umap=False):
    """``fit_or_load_probe`` for every seed into ``{save_folder}/seed_{i}``, then
    ``get_metrics`` (and, with ``umap``, ``plot_umap``) for each evaluation set."""
    for i in seeds:
        folder = f"{save_folder}/seed_{i}"
        results = fit_or_load_probe(folder, i, device, train, eval_sets, unique_cats)
        for tag, result in results.items():
            get_metrics(folder, result, tag=tag, unique_cats=unique_cats)
            if umap:
                plot_umap(folder, result, tag=tag, unique_cats=unique_cats)


def true_pred_frames(df, unique_cats):
    """Split a predictions table into label and score frames named by class."""
    df_true = df[[f"{c}_true" for c in unique_cats]]
    df_true.columns = list(unique_cats)
    df_pred = df[[f"{c}_pred" for c in unique_cats]]
    df_pred.columns = list(unique_cats)
    return df_true, df_pred


def get_multilabel_df(df_true, df_pred):
    """Per-class average precision and AUC (classes with both labels present),
    plus an ``Overall`` row, sorted by class size."""
    avg_precisions, aucs, all_categories, all_counts = [], [], [], []
    for cat in df_true.columns:
        if df_true[cat].nunique() != 2:
            continue
        avg_precisions.append(average_precision_score(df_true[cat], df_pred[cat]))
        all_categories.append(cat)
        all_counts.append(df_true[cat].sum())
        aucs.append(roc_auc_score(df_true[cat], df_pred[cat]))

    avg_precisions.append(average_precision_score(df_true.values, df_pred.values))
    aucs.append(roc_auc_score(df_true.values, df_pred.values))
    all_categories.append("Overall")
    all_counts.append(len(df_true))
    return (
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


def plot_multilabel_metrics(df, metric="Average Precision", label="valid", save_folder="./"):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, figsize=(16, 10))
    sns.barplot(
        x="Category",
        y=metric,
        hue="Category",
        palette=sns.color_palette(cc.glasbey_dark, len(df)),
        data=df,
        ax=ax,
        orient="v",
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.savefig(f"{save_folder}/{label}_{metric}.png", dpi=100, bbox_inches="tight")
    plt.close()


def get_metrics(save_folder, df_test, tag="test", *, unique_cats, single_label=True):
    """Write ``{tag}_metrics.csv`` (per-class and overall AP, AUC, coverage error,
    label-ranking AP) and its two bar plots for a predictions table.

    ``single_label`` is for one class per sample (cell-cycle stages): the
    argmax confusion matrix and classification report are written as well.
    """
    df_true, df_pred = true_pred_frames(df_test, unique_cats)
    non_zero_cats = [c for c in unique_cats if df_true[c].sum() > 0]
    df_true = df_true[non_zero_cats]
    df_pred = df_pred[non_zero_cats]

    if single_label:
        true_label = df_true.values.argmax(1)
        pred_label = df_pred.values.argmax(1)

        conf_mat = confusion_matrix(true_label, pred_label, normalize="true")
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
            ax=ax,
        )
        plt.title(f"{tag} Confusion Matrix")
        plt.savefig(f"{save_folder}/{tag}_confusion_matrix.png", dpi=500, bbox_inches="tight")
        plt.close()

        cls_rep = classification_report(
            true_label, pred_label, output_dict=True, target_names=non_zero_cats, digits=4
        )
        pd.DataFrame(cls_rep).T.to_csv(f"{save_folder}/{tag}_classification_report.csv")

    df_multilabel = get_multilabel_df(df_true, df_pred)
    df_multilabel["Coverage Error"] = coverage_error(df_true.values, df_pred.values)
    df_multilabel["Label Ranking Average Precision"] = label_ranking_average_precision_score(
        df_true.values, df_pred.values
    )
    df_multilabel["Micro Average Precision"] = average_precision_score(
        df_true.values, df_pred.values, average="micro"
    )
    df_multilabel.to_csv(f"{save_folder}/{tag}_metrics.csv", index=False)

    for metric in ("Average Precision", "AUC"):
        plot_multilabel_metrics(df_multilabel, metric=metric, label=tag, save_folder=save_folder)


def plot_umap(save_folder, df, tag, unique_cats):
    """UMAP of the probe embedding, coloured by true class (``{tag}_umap.png``)."""
    feats = df[[col for col in df.columns if "feat" in col]].values
    umap_feat = UMAP(verbose=2).fit_transform(feats)
    umap_df = pd.DataFrame(
        {
            "x": umap_feat[:, 0],
            "y": umap_feat[:, 1],
            "Category": df[[f"{c}_true" for c in unique_cats]].idxmax(axis=1),
        }
    )
    umap_df["Category"] = pd.Categorical(
        umap_df["Category"].str.replace("_true", ""), categories=unique_cats, ordered=True
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
    ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=2,
    )
    plt.savefig(f"{save_folder}/{tag}_umap.png", dpi=500, bbox_inches="tight")
    plt.close()
