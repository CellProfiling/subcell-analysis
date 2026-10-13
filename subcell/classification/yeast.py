"""Localization and cell-cycle probes on yeast features (Fig. 2f,h,j; Supplementary Table 1).

For every SubCell variant (encoder and crop resize side length; ``Orig`` is no
resize), an MLP (``subcell.classification.mlp``) is trained on ten seeds on the
train set of each yeast task and scored on its test set. Predictions, metrics and
UMAPs are written per seed.

    python -m subcell.classification.yeast
    python -m subcell.classification.yeast --variants MAE-512 ViT-512 --tasks cellcycle

Inputs: ``config.YEAST_DIR/<variant>/<task>_{train,test}_set.pth``, written by
``subcell.inference.yeast`` as ``(labels, features)``, one row of labels per cell.
Outputs: ``config.YEAST_DIR/<variant>/task_<task>/seed_<i>/``.
"""

import argparse

import torch
from sklearn.preprocessing import MultiLabelBinarizer

from subcell import config
from subcell.classification.mlp import probe_seeds

VARIANTS = {
    "MAE-Orig": "subcell_rbg_mae_resize_-1",
    "MAE-256": "subcell_rbg_mae_resize_256",
    "MAE-512": "subcell_rbg_mae_resize_512",
    "ViT-Orig": "subcell_rbg_vit_resize_-1",
    "ViT-256": "subcell_rbg_vit_resize_256",
    "ViT-512": "subcell_rbg_vit_resize_512",
}
TASKS = ("localization", "cellcycle")


def eval_classification(device, variant_dir, task):
    result_path = config.YEAST_DIR / variant_dir
    train_df, train_x = torch.load(result_path / f"{task}_train_set.pth", weights_only=False)
    val_df, val_x = torch.load(result_path / f"{task}_test_set.pth", weights_only=False)

    le = MultiLabelBinarizer()
    train_y = torch.from_numpy(le.fit_transform(train_df.values))
    val_y = torch.from_numpy(le.transform(val_df.values))

    probe_seeds(
        f"{result_path}/task_{task}", device, (train_x, train_y), {"val": (val_x, val_y)},
        le.classes_, umap=True,
    )


def main(variants=None, tasks=TASKS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for method in variants or VARIANTS:
        for task in tasks:
            print(f"Yeast {task}: {method}", flush=True)
            eval_classification(device, VARIANTS[method], task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), help="default: all")
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    a = ap.parse_args()
    main(a.variants, a.tasks)
