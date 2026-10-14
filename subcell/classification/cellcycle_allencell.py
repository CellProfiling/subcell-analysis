"""Cell-cycle stage probes on Allen Cell hiPSC features (Fig. 2g,i; Supplementary Table 10).

For every model variant, an MLP (``subcell.classification.probe``) is trained on ten
seeds to predict the mitotic stage of a cell from its frozen features, on the
DINO4Cells train / validation / test split of the cells; per-seed predictions and
metrics are written beside the features. ``figures.fig2_cell_cycle`` summarises them.

    python -m subcell.classification.cellcycle_allencell                  # all variants
    python -m subcell.classification.cellcycle_allencell --variants DINO-ImageNet

Inputs, under ``config.ALLENCELL_DIR``: ``train_indices.pth`` and ``test_indices.pth``
(positions of the DINO4Cells split in ``dino/DINO_features_and_df.pth``) and, per
variant in ``config.ALLENCELL_SWEEP``, ``<variant dir>/all_features.pth`` written by
``subcell.inference.allencell`` as ``(metadata, features)`` (the two DINO variants are
``dino/DINO_features_and_df.pth`` and ``pretrained/pretrained_features_and_df.pth``,
as ``(features, metadata)``).
Outputs: ``train_indices1.pth`` / ``val_indices1.pth`` (an 80/20 stratified split of the
training cells, written once and then reused) and ``<variant dir>/classification/seed_<i>/``.
"""

import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from subcell import config
from subcell.classification.probe import probe_seeds

DINO_FILES = {
    "dino": "DINO_features_and_df.pth",
    "pretrained": "pretrained_features_and_df.pth",
}
SEEDS = range(10)


def feature_file(variant_dir):
    name = DINO_FILES.get(variant_dir, "all_features.pth")
    return config.ALLENCELL_DIR / variant_dir / name


def load_features(variant_dir):
    """``(features, metadata)`` of one variant."""
    obj = torch.load(feature_file(variant_dir), weights_only=False)
    return obj if variant_dir in DINO_FILES else obj[::-1]


def split_cells():
    """Cell ids of the train / validation / test split. The validation set is a
    stratified fifth of the DINO4Cells training cells, saved on first use so reruns
    use the same cells."""
    root = config.ALLENCELL_DIR
    _, dino_df = load_features("dino")
    train_path, val_path = root / "train_indices1.pth", root / "val_indices1.pth"
    if not (train_path.exists() and val_path.exists()):
        train_idx = torch.load(root / "train_indices.pth", weights_only=False)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.2,
            random_state=42,
            stratify=dino_df.iloc[train_idx]["cell_stage"].values,
        )
        torch.save(train_idx, train_path)
        torch.save(val_idx, val_path)
    indices = [
        torch.load(p, weights_only=False)
        for p in (train_path, val_path, root / "test_indices.pth")
    ]
    return [dino_df["CellId"][idx] for idx in indices]


def main(variants=None, seeds=SEEDS):
    """Train and evaluate the probes of ``variants`` (names from
    ``config.ALLENCELL_SWEEP``; default all)."""
    variants = {k: config.ALLENCELL_SWEEP[k] for k in (variants or config.ALLENCELL_SWEEP)}
    train_cell_ids, val_cell_ids, test_cell_ids = split_cells()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for method, variant_dir in variants.items():
        print(f"Allen Cell: {method}", flush=True)
        features, df = load_features(variant_dir)

        cell_stage_df = pd.get_dummies(df["cell_stage"])
        y = torch.tensor(cell_stage_df.values)
        unique_cats = cell_stage_df.columns

        sets = {}
        for tag, cell_ids in (("train", train_cell_ids), ("val", val_cell_ids), ("test", test_cell_ids)):
            idx = df["CellId"].isin(cell_ids)
            sets[tag] = (features[idx], y[idx])

        probe_seeds(
            config.ALLENCELL_DIR / variant_dir / "classification", device, sets["train"],
            {"val": sets["val"], "test": sets["test"]}, unique_cats, seeds, umap=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", choices=list(config.ALLENCELL_SWEEP),
                    help="model variants to run (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    main(a.variants, a.seeds)
