"""Display names, orderings and identities. One definition each.

Directory names, feature filenames and cached metrics tables key on the
*internal* model names (``MAE-CellS-ProtS-Pool``), which must not change or every
cache invalidates. Manuscript figures use the *published* names (``SubCell``).
Orderings used by more than one figure are defined here so legends and rows
agree between figures.
"""

from __future__ import annotations

# ------------------------------------------------------------------- models

# Published name per internal name. Absent -> the internal name is the published
# one (true for the external baselines, which are cited under their own names).
DISPLAY_NAME = {
    "MAE-CellS-ProtS-Pool": "SubCell",
    "ViT-ProtS-Pool": "SubCell-P",
    "DINO4Cells-HPA": "DINO4Cells-HPA",
    "Cell-DINO-HPA": "Cell-DINO-HPA",
    "DINO4Cells-CP": "DINO4Cells-CP",
    "ViT-Supervised": "ViT-Weak-Supervised",
    "bestfitting": "bestfitting",
}

# The three self-supervised encoders trained on HPA, in figure order:
# SubCell, SubCell-P, DINO4Cells.
SSL_METHODS = ["MAE-CellS-ProtS-Pool", "ViT-ProtS-Pool", "DINO4Cells-HPA"]

# Trained on HPA localization labels. They appear in the UMAP grids as a visual
# reference but are excluded from label-retrieval metrics, where scoring a model
# on the labels it was fitted to is partly circular.
SUPERVISED = ["ViT-Supervised", "bestfitting"]

# Models of Fig. 3 and Extended Data Fig. 8 in plotting order, each with the
# reference model the others are tested against.
FIG3_MODELS = (["MAE-CellS-ProtS-Pool", "DINO4Cells-HPA", "Cell-DINO-HPA",
                "ViT-Supervised", "bestfitting"], "MAE-CellS-ProtS-Pool")
ED8_MODELS = (FIG3_MODELS[0] + ["ViT-ProtS-Pool"], "ViT-ProtS-Pool")


def display(method: str) -> str:
    """Published name for an internal model name."""
    return DISPLAY_NAME.get(method, method)


# ----------------------------------------------------- HPA harmonization keys
#
# Harmony batch keys compared on the HPA cell features (Extended Data Fig. 3).
# Ordered from no correction to the finest key.

HARM_METHODS = SSL_METHODS + ["Cell-DINO-HPA"]
HARM_ORDER = ["none", "microscope", "atlas_name", "if_plate_id"]
HARM_SHORT = {
    "none": "None",
    "microscope": "Microscope",
    "atlas_name": "Cell line",
    "if_plate_id": "Plate",
}

# Covariates the HPA embeddings are coloured by, and whether each is categorical.
# Plate is drawn as a continuous ramp: it has 1,586 levels, and plate ids increase
# roughly with acquisition date, so smooth colour across the manifold is temporal
# drift.
HPA_COVARIATES = ["Location", "atlas_name", "microscope", "if_plate_id"]
GRADIENT_COVARIATES = {"if_plate_id"}


# ------------------------------------------------------- random baselines

# Dimensions of the random-vector baselines (``Random_<dim>``) scored next to the
# embeddings in Fig. 3d / Extended Data Fig. 8c and Fig. 6d / Extended Data Fig. 7b.
RANDOM_BASELINE_DIMS = (256, 512, 1024, 2048)


# --------------------------------------------------------- tissue TMA cohort
#
# 29 cores from 19 donors, 210,482 cells (Extended Data Fig. 2, Supplementary
# Note 2). Non-cancer cores are all cohort G and cancer cores all cohort H, so
# cancer status is confounded with cohort and with acquisition order; no
# non-cancer/cancer comparison is reported. The acquisition-block prefix of the
# image filenames (G1, H1, HX, ...) indexes acquisition order and is not used as
# an identity. Cores are nested within donors; three donors span two blocks.

# Harmony batch keys for the tissue features, coarse to fine, with their display
# titles and numbers of levels. Keys keep the on-disk name ("patient").
TISSUE_HARM_TITLE = {
    "none": "No correction",
    "cohort": "Cohort (G/H)",
    "slide": "Acquisition block",
    "patient": "Donor",
    "patient+slide": "Donor × block",
    "core": "Core",
}
TISSUE_HARM_NLEVELS = {"none": None, "cohort": 2, "slide": 9, "patient": 19,
                       "patient+slide": 22, "core": 29}

# Fields the tissue UMAPs are coloured by: (metadata column, axis label).
TISSUE_FIELDS = [("tissue", "Tissue"), ("cancer", "Cancer"), ("patient", "Donor"),
                 ("slide", "Acquisition block"), ("core_id", "Core")]

# k-means with k = 13 on the core-corrected features is the reported partition.
TISSUE_K = 13
# Clusters with under this share of cells scored "correct" are excluded from the
# prevalence linkage.
TISSUE_MIN_CORRECT = 0.60


def load_cluster_ids(path=None):
    """The canonical c1..c13 cluster crosswalk (``config.TISSUE_CLUSTER_IDS``).

    Columns: ``cluster`` (canonical id), ``source_cluster`` (id as written by the
    clustering run), ``in_clustering`` (False for clusters excluded from the
    prevalence linkage) and ``source_id`` (``source_cluster`` as an integer).
    """
    import pandas as pd

    from . import config

    df = pd.read_csv(path or config.TISSUE_CLUSTER_IDS)
    df["source_id"] = df["source_cluster"].str.lstrip("c").astype(int)
    return df
