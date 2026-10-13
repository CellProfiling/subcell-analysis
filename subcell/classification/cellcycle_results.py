"""Summaries of the cell-cycle probe results (Fig. 2g,i,k; Supplementary Table 10).

``plot_allencell`` reads the per-seed test predictions written by
``subcell.classification.cellcycle_allencell`` and ``plot_fucci`` the per-fold
validation predictions of ``subcell.classification.cellcycle_fucci``; both are
called by ``figures.fig2_cell_cycle``. Each takes ``{model name: variant directory}``
(``config.ALLENCELL_VARIANTS`` / ``config.ALLENCELL_SWEEP`` / ``config.FUCCI_VARIANTS``),
the result tree those directories are under, and an output directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ALLENCELL_STAGES = {
    "M0": "interphase",
    "M1M2": "prophase",
    "M3": "early prometaphase",
    "M4M5": "prometaphase/metaphase",
    "M6M7_complete": "anaphase/telophase paired",
    "M6M7_single": "anaphase/telophase unpaired",
}
FUCCI_STAGES = ["G1", "G1S", "G2"]
METRICS = ("accuracy", "macro F1", "micro F1")


def _labels(preds, cats):
    """Argmax true and predicted class of a predictions table."""
    true = preds[[f"{c}_true" for c in cats]].values.argmax(1)
    pred = preds[[f"{c}_pred" for c in cats]].values.argmax(1)
    return true, pred


def _scores(true, pred, method, key, key_name):
    """One row of accuracy and macro / micro F1 for a set of predictions."""
    return pd.DataFrame({
        "accuracy": [accuracy_score(true, pred)],
        "macro F1": [f1_score(true, pred, average="macro")],
        "micro F1": [f1_score(true, pred, average="micro")],
        "Method": method,
        key_name: key,
    })


def _save(stem, **kwargs):
    for ext in ("pdf", "png"):
        plt.savefig(f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close()


def _bar_plots(scores, stem, **barplot_kwargs):
    """One bar plot per metric, ``{stem}_{metric}`` with spaces as underscores."""
    for metric in METRICS:
        _, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Method", y=metric, hue="Method", data=scores, ax=ax, **barplot_kwargs)
        for elem in ax.containers:
            ax.bar_label(elem, fmt="%.3f", label_type="edge", fontsize=26, padding=-30)
        ax.set_box_aspect(0.5)
        plt.xticks(rotation=45, ha="right")
        _save(f"{stem}_{metric.lower().replace(' ', '_')}", dpi=300)


def plot_allencell(variants, root, out_dir, seeds=range(10)):
    """Allen Cell, ``seeds`` (default ten) per model: row-normalised confusion matrix averaged
    over seeds (``fig2i_confusion_<model>``), metric bar plots (``fig2g_<metric>``)
    and their mean +- s.d. table (``fig2g_mean_std_metrics.csv``)."""
    out_dir = Path(out_dir)
    cats = list(ALLENCELL_STAGES)
    names = [ALLENCELL_STAGES[c] for c in cats]
    all_scores = []
    for method, variant_dir in variants.items():
        conf_mats = []
        for i in seeds:
            preds = pd.read_csv(f"{root}/{variant_dir}/classification/seed_{i}/test_preds.csv")
            true, pred = _labels(preds, cats)
            conf_mats.append(confusion_matrix(true, pred, normalize="true"))
            all_scores.append(_scores(true, pred, method, i, "Seed"))
        sns.heatmap(
            np.stack(conf_mats).mean(axis=0),
            annot=True,
            fmt=".2f",
            square=True,
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
        )
        _save(out_dir / f"fig2i_confusion_{method}", dpi=300)

    all_scores = pd.concat(all_scores, ignore_index=True)
    mean_df = all_scores.groupby("Method").mean().drop(columns=["Seed"])
    std_df = all_scores.groupby("Method").std().drop(columns=["Seed"])
    mean_std_df = mean_df.copy().astype(object)
    for col in mean_df.columns:
        for idx in mean_df.index:
            mean_std_df.loc[idx, col] = f"{mean_df.loc[idx, col]:.3f} ± {std_df.loc[idx, col]:.4f}"
    mean_std_df.to_csv(out_dir / "fig2g_mean_std_metrics.csv")

    _bar_plots(all_scores, str(out_dir / "fig2g"))


def plot_fucci(variants, root, out_dir, seeds=range(10)):
    """FUCCI U2OS: for each of the ten cross-validation folds, the stage
    predictions of its ``seeds`` (default ten) pooled and scored (``fig2k_metrics.csv``, one row
    per model and fold), and metric bar plots over folds (``fig2k_<metric>``)."""
    out_dir = Path(out_dir)
    all_scores = []
    for method, variant_dir in variants.items():
        for split in range(10):
            preds = pd.concat(
                [
                    pd.read_csv(
                        f"{root}/{variant_dir}/split_{split}/classification/seed_{seed}/val_preds.csv",
                        usecols=lambda c: "feat" not in c,
                    )
                    for seed in seeds
                ],
                ignore_index=True,
            )
            true, pred = _labels(preds, FUCCI_STAGES)
            all_scores.append(_scores(true, pred, method, split, "Split"))

    all_scores = pd.concat(all_scores, ignore_index=True)
    all_scores["Method"] = pd.Categorical(
        all_scores["Method"], categories=list(variants), ordered=True
    )
    all_scores.to_csv(out_dir / "fig2k_metrics.csv")

    sns.set_style("ticks")
    _bar_plots(all_scores, str(out_dir / "fig2k"), errorbar="sd")
