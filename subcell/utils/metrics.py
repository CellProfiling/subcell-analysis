"""Multilabel classification metrics and significance notation."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    label_ranking_average_precision_score,
)


def get_optimal_threshold(y_true, y_pred):
    """Decision threshold in [0.1, 0.9] that maximises macro F1."""
    best_threshold, best_f1 = 0, 0
    for threshold in np.linspace(0.1, 0.9, 100):
        f1 = f1_score(y_true, (y_pred > threshold).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold


def compute_metrics(y_true, y_pred, threshold=None):
    """F1 (sample, macro, micro), average precision (macro, micro) and MLRAP.

    The F1 scores use ``threshold``, or the macro-F1-optimal threshold on these
    predictions when none is given.
    """
    if threshold is None:
        threshold = get_optimal_threshold(y_true, y_pred)
    y_pred_binary = (y_pred > threshold).astype(int)
    return {
        "acc": f1_score(y_true, y_pred_binary, average="samples"),
        "macro_f1": f1_score(y_true, y_pred_binary, average="macro"),
        "micro_f1": f1_score(y_true, y_pred_binary, average="micro"),
        "macro_ap": average_precision_score(y_true, y_pred, average="macro"),
        "micro_ap": average_precision_score(y_true, y_pred, average="micro"),
        "mlrap": label_ranking_average_precision_score(y_true, y_pred),
    }


def stars(p):
    """Significance stars: * P < 0.05, ** P < 0.01, *** P < 0.001."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
