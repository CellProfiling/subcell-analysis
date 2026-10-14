"""Move the Kaggle test-set features from the inference output to ``features/<model>/kaggle_features``.

``python -m subcell.inference.hpa -c configs/kaggle.yaml`` writes
``<inference_folder>/<run>/all_features_pooled.pth``. This rewrites each as
``config.FEATURE_DIR/<model>/kaggle_features/all_features.pth``, next to the model's
``hpa_features``, with the metadata columns ``get_challenge_data`` in
``subcell.classification.localization`` reads (``annotated_label``, ``antibody``, ...).

    python -m subcell.inference.kaggle_features [--inference-folder DIR]

The inference folder defaults to ``config.KAGGLE_INFERENCE_DIR``. Runs whose output
does not exist are skipped.
"""

import argparse
import os

import torch

from subcell import config

RUNS = {
    "cell_dino_hpa_vitl16": "Cell-DINO-HPA",
    "dino": "DINO4Cells-HPA",
    "subcell_rybg_mae": "MAE-CellS-ProtS-Pool",
    "subcell_rybg_vit": "ViT-ProtS-Pool",
    "subcell_vit_supervised": "ViT-Supervised",
    "bestfitting": "bestfitting",
}


def main(inference_folder):
    for run, model in RUNS.items():
        src = f"{inference_folder}/{run}/all_features_pooled.pth"
        if not os.path.isfile(src):
            print(f"{run}: no output yet, skipped")
            continue
        df, feats = torch.load(src, map_location="cpu", weights_only=False)
        parts = df["plate_position_sample"].str.split("_", expand=True)
        df["if_plate_id"] = parts[0].astype(int)
        df["position"] = parts[1]
        df["sample"] = parts[2].astype(int)
        df["cell_id"] = df["cell_id"].str.replace("_crop", "").astype(int)
        df = df.rename(columns={"locations": "annotated_label"})
        df = df[["cell_id", "if_plate_id", "position", "sample", "atlas_name", "antibody",
                 "ensembl_ids", "gene_names", "annotated_label"]]
        out = config.FEATURE_DIR / model / "kaggle_features" / "all_features.pth"
        os.makedirs(out.parent, exist_ok=True)
        torch.save((df, feats), out)
        print(f"{run}: {tuple(feats.shape)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inference-folder", default=str(config.KAGGLE_INFERENCE_DIR))
    main(ap.parse_args().inference_folder)
