"""Filesystem locations used by the analysis.

Every path is derived from a small set of named roots. Each root is resolved, in
order, from an environment variable of the same name, from ``paths.local.toml`` in
the repository root (not version-controlled; see ``paths.example.toml``), and
from a default inside the checkout.

Roots
-----
``SUBCELL_ROOT``       analysis outputs: features, embeddings, cached results
``SUBCELL_FIG_DIR``    rendered figures (default ``$SUBCELL_ROOT/figures``)
``SUBCELL_WEIGHTS``    SubCell model weights
``BASELINE_WEIGHTS``   weights of the compared models (DINO4Cells, Cell-DINO, bestfitting, ViT-Weak-Supervised)
``SUBCELL_CACHE``      library caches (Hugging Face, matplotlib, the copairs null), default ``$SUBCELL_ROOT``
``SUBCELL_FONT_DIR``   Arial- or Helvetica-metric fonts (see ``subcell.style.rc``)
``DINOV2_REPO``        checkout of facebookresearch/dinov2, for Cell-DINO
``HPA_DATA``           HPA v23 single-cell crops and their metadata
``KAGGLE_DATA``        HPA Kaggle single-cell challenge test crops
``CM4AI_DATA``         CM4AI MDA-MB-468 drug-perturbation crops
``ALLENCELL_DATA``     Allen Cell hiPSC images
``OPENCELL_DATA``      OpenCell crops
``YEAST_DATA``         yeast localization and cell-cycle datasets
``FUCCI_DATA``         FUCCI U2OS crops and single-cell statistics
``MICROSCOPE_DATA``    widefield/confocal comparison (Supplementary Note 1)

Configuration files under ``configs/`` refer to these roots as ``${NAME}``;
:func:`load_yaml` substitutes them. Reading a path never creates it; call
:func:`ensure` on a directory before writing into it.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_LOCAL_FILE = REPO / "paths.local.toml"
_LOCAL = tomllib.loads(_LOCAL_FILE.read_text()) if _LOCAL_FILE.exists() else {}


def _root(name: str, default) -> Path:
    return Path(os.environ.get(name) or _LOCAL.get(name) or default)


ROOT = _root("SUBCELL_ROOT", REPO / "results")
DATA = REPO / "data"
ROOTS = {
    "SUBCELL_ROOT": ROOT,
    "SUBCELL_FIG_DIR": _root("SUBCELL_FIG_DIR", ROOT / "figures"),
    "SUBCELL_WEIGHTS": _root("SUBCELL_WEIGHTS", DATA / "weights"),
    "BASELINE_WEIGHTS": _root("BASELINE_WEIGHTS", DATA / "baseline_weights"),
    "SUBCELL_FONT_DIR": _root("SUBCELL_FONT_DIR", DATA / "fonts"),
    "DINOV2_REPO": _root("DINOV2_REPO", DATA / "dinov2"),
    "HPA_DATA": _root("HPA_DATA", DATA / "hpa"),
    "KAGGLE_DATA": _root("KAGGLE_DATA", DATA / "kaggle"),
    "CM4AI_DATA": _root("CM4AI_DATA", DATA / "cm4ai"),
    "ALLENCELL_DATA": _root("ALLENCELL_DATA", DATA / "allencell"),
    "OPENCELL_DATA": _root("OPENCELL_DATA", DATA / "opencell"),
    "YEAST_DATA": _root("YEAST_DATA", DATA / "yeast"),
    "FUCCI_DATA": _root("FUCCI_DATA", DATA / "fucci"),
    "MICROSCOPE_DATA": _root("MICROSCOPE_DATA", DATA / "microscopes"),
}
FIG_DIR = ROOTS["SUBCELL_FIG_DIR"]
FONT_DIR = ROOTS["SUBCELL_FONT_DIR"]

# Libraries that cache into the home directory by default.
_CACHE = _root("SUBCELL_CACHE", ROOT)
os.environ.setdefault("HF_HOME", str(_CACHE / ".hf"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / ".matplotlib"))
COPAIRS_CACHE = _CACHE / ".copairs_null"

# ------------------------------------------------------- annotations in-repo

ANNOT_DIR = REPO / "annotations"
METHODS_CSV = ANNOT_DIR / "methods_info.csv"          # embedding files per (Method, Cell_Line, Harmonized)
MICROSCOPE_CSV = ANNOT_DIR / "HPA_microscope.csv"     # FOV -> imaging instrument
LOCATION_GROUPS = ANNOT_DIR / "location_group_mapping.tsv"   # Supplementary Table 2
HPA_SPLITS = ANNOT_DIR / "splits"                     # {train,valid,test}_antibodies.txt
CM4AI_SPLIT = ANNOT_DIR / "cm4ai_antibody_gene_split.csv"
SUBCELL_LOCATION = ANNOT_DIR / "subcellular_location.tsv"
GENE2UNIPROT = ANNOT_DIR / "gene2uniprot.tsv"
ENSG2UNIPROT = ANNOT_DIR / "ensg2uniprot.tsv"
RNA_CELLINE = ANNOT_DIR / "rna_celline.tsv"           # HPA v24 bulk mRNA per cell line
IF_IMAGES = ANNOT_DIR / "IF-image-orig-bboxes.csv"    # HPA v23 image table
COMPLEXES_DIR = ANNOT_DIR / "complexes"               # GO / Reactome / CORUM / hu.MAP / BioPlex / STRING
PPI_DIR = REPO / "ppi_data"                           # raw BioPlex / STRING downloads for subcell.io.complexes
TISSUE_ANNOT_K13 = ANNOT_DIR / "hpa_tissue_cluster_annotation_k13.tsv"
TISSUE_CORE_META = ANNOT_DIR / "hpa_tissue_core_metadata.tsv"
# Protein labels of Fig. 6c,e,f, as used for the figures (see subcell.multimodal.compartment_labels)
U2OS_LOCATIONS = ANNOT_DIR / "u2os_protein_locations.tsv"   # HPA location annotations per UniProt entry, U2OS cells
SUBORGANELLAR_GOCC = ANNOT_DIR / "suborganellar_gocc_gprofiler.tsv"   # members of 11 GO:CC compartments, g:Profiler 2026-03-14
GOBP_TERMS = ANNOT_DIR / "gobp_gprofiler.tsv"   # members of the 158 GO:BP processes scored, g:Profiler 2026-03
GPROFILER_QUERY_IDS = ANNOT_DIR / "gprofiler_query_ids.txt"   # the UniProt entries both g:Profiler queries were run on

# ------------------------------------------------------------- result trees

FEATURE_DIR = ROOT / "features"                 # <model>/<dataset>/all_features.pth
KAGGLE_INFERENCE_DIR = ROOT / "kaggle_inference"   # inference output of configs/kaggle.yaml
MEAN_FEATURES_CSV = FEATURE_DIR / "mean_features_paths.csv"   # per-gene mean feature tables
PPI_EMBEDDING = FEATURE_DIR / "ppi" / "ppi_emd.tsv"             # node2vec protein-interaction embedding
HPA_UMAP_DIR = ROOT / "fov_umap_analysis"       # FOV means, UMAP coordinates, retrieval mAP
MUTUAL_INFO_DIR = ROOT / "mutual_info"          # per-dimension mutual information
MORPHOLOGY_DIR = ROOT / "cell_morphology"       # cell-line / localization similarity, Mantel
ATTN_CORR_DIR = ROOT / "hpa_attn_corr"          # attention-map vs channel correlation

TISSUE_DIR = ROOT / "HPA_tissue"                # tissue microarray cohort (Extended Data Fig. 2)
ALLENCELL_DIR = ROOT / "allencell"              # cell-cycle stage (Fig. 2g,i)
FUCCI_DIR = ROOT / "fucci"                      # FUCCI U2OS stage and pseudotime (Fig. 2k,l)
YEAST_DIR = ROOT / "yeast"                      # yeast localization and cell cycle (Fig. 2f,h,j)
CM4AI_DIR = ROOT / "b2ai_analysis"              # CM4AI drug perturbation (Fig. 4a-c)
CM4AI_CROPS = ROOT / "bridge2ai_crops"          # masked CM4AI single-cell crops
CM4AI_METADATA = ROOTS["CM4AI_DATA"] / "full_subcell_crops_all_z_filtered.csv"   # one row per cell crop
MULTIMODAL_DIR = ROOT / "multimodal_integration"  # image + sequence co-embedding (Fig. 6)
MULTIMODAL_ENRICHMENT_DIR = ROOT / "bionic_eval_embed_imgseq"   # interaction enrichment (Fig. 6d, Extended Data Fig. 7b)
CLUSTER_TRANSFER_DIR = ROOT / "cluster_transfer"   # image and sequence clusters on the co-embedding (Fig. 6a,b, Extended Data Fig. 7a)
COMPARTMENT_ENRICHMENT_DIR = ROOT / "compartment_enrichment"   # within- vs between-compartment and process similarity (Fig. 6c,f, Extended Data Fig. 7c)
PARALOG_RANKS_DIR = ROOT / "paralog_ranks"   # sequence vs multimodal ranks of protein pairs (Fig. 6e)

# GO / Reactome membership prediction from per-gene mean embeddings (Fig. 3d).
# Repeat r draws its ten-fold split with seed r.
FUNCTION_PREDICTION_DIR = ROOT / "bionic_eval_v2"
FUNCTION_PREDICTION_REPEATS = [FUNCTION_PREDICTION_DIR / f"repeat_{r}" for r in range(3)]
FUNCTION_PREDICTION_HARMONIZED = [
    "none",
    "atlas_name",
    "if_plate_id",
    "if_plate_id_microscope",
    "if_plate_id_microscope_atlas_name",
    "microscope",
]
FUNCTION_PREDICTION_MODELS = {
    "Cell-DINO-HPA": FUNCTION_PREDICTION_HARMONIZED,
    "DINO4Cells-HPA": FUNCTION_PREDICTION_HARMONIZED,
    "MAE-CellS-ProtS-Pool": FUNCTION_PREDICTION_HARMONIZED,
    "ViT-ProtS-Pool": FUNCTION_PREDICTION_HARMONIZED,
    "ViT-Supervised": ["none"],
    "bestfitting": ["none"],
}
FUNCTION_PREDICTION_CELL_LINES = ["U2OS", "all"]

# --------------------------------------------------------------- tissue tree

TISSUE_FEATURES = TISSUE_DIR / "features"
TISSUE_FEATURE_CACHE = TISSUE_DIR / "harmonization_v2"   # concatenated MT + VIM features and base metadata
TISSUE_HARM = TISSUE_DIR / "harmonization_patient"   # Harmony over each candidate batch key
TISSUE_UMAPS = TISSUE_HARM / "umaps_patient.npz"
TISSUE_META = TISSUE_HARM / "metadata.csv"
TISSUE_INTENSITY = TISSUE_FEATURES / "intensity_metrics.csv"   # measurements withheld from clustering
TISSUE_CLUSTERS = TISSUE_DIR / "clustering_core_k13" / "kmeans_k13_core_v1labels"   # reported partition
TISSUE_SWEEP = TISSUE_DIR / "clustering_compare_4keys"
TISSUE_CLUSTER_IDS = TISSUE_HARM / "figures" / "cluster_id_map.csv"   # cluster number -> c1..c13
TISSUE_ORIGINAL_LABELS = TISSUE_DIR / "clustering" / "kmeans_k19" / "cell_clusters.csv"   # first labelling, k = 19
TISSUE_REF_LABELS = TISSUE_DIR / "clustering_v2" / "kmeans_k13_v1labels" / "cell_clusters.csv"

# ------------------------------------------------- downstream model variants
#
# Model variants evaluated on each downstream dataset; values are directories
# under the dataset's result tree, each holding an ``all_features.pth``.

# Allen Cell (Fig. 2g,i).
ALLENCELL_VARIANTS = {
    "DINO4Cells-WTC": "dino",
    "DINO-ImageNet": "pretrained",
    "MAE-CellS-ProtS-Pool": "subcell_bg_mae_channel_comb_concat_norm_perchannel",
    "ViT-ProtS-Pool": "subcell_bg_vit_channel_comb_concat_norm_perchannel",
}
# Channel combinations and normalizations (Supplementary Table 10).
ALLENCELL_SWEEP = {
    "DINO4Cells-WTC": "dino",
    "DINO-ImageNet": "pretrained",
    **{f"{arch}-{name}": f"subcell_{pre}_{arch.lower()}_channel_comb_{suf}"
       for arch in ("MAE", "ViT")
       for name, pre, suf in [
           ("DNA-Struct-PerCh", "bg", "nuc_prot_only_norm_perchannel"),
           ("DNA-Struct-PerIm", "bg", "nuc_prot_only_norm_perimage"),
           ("DNA-Struct-Plasma-Concat-PerCh", "bg", "concat_norm_perchannel"),
           ("DNA-Struct-Plasma-Concat-PerIm", "bg", "concat_norm_perimage"),
           ("DNA-Struct-Plasma-PerCh", "rbg", "plasma_nuc_prot_norm_perchannel"),
           ("DNA-Struct-Plasma-PerIm", "rbg", "plasma_nuc_prot_norm_perimage"),
       ]},
}

# FUCCI U2OS (Fig. 2k,l): the dataset has no nuclear stain, so each encoder is run
# with three substitutes for the DNA channel.
FUCCI_VARIANTS = {
    "MAE-Nuc-Zero-Resize": "subcell_rbg_mae_channel_comb_nuc_zero_resize_True",
    "ViT-Nuc-Zero-Resize": "subcell_rbg_vit_channel_comb_nuc_zero_resize_True",
    "MAE-Nuc-Const-Resize": "subcell_rbg_mae_channel_comb_nuc_const_resize_True",
    "ViT-Nuc-Const-Resize": "subcell_rbg_vit_channel_comb_nuc_const_resize_True",
    "MAE-Nuc-CDT-GMNN-Resize":
        "subcell_rbg_mae_channel_comb_nuc_cdt_gmmn_resize_True",
    "ViT-Nuc-CDT-GMNN-Resize":
        "subcell_rbg_vit_channel_comb_nuc_cdt_gmmn_resize_True",
}

# CM4AI drug perturbation (Fig. 4a-c).
CM4AI_VARIANTS = {
    "MAE-CellS-ProtS-Pool": "subcell_rybg_mae",
    "ViT-ProtS-Pool": "subcell_rybg_vit",
    "DINO4Cells-HPA": "dino",
    "Cell-DINO-HPA": "cell_dino_hpa_vitl16",
    "bestfitting": "bestfitting",
}

# ---------------------------------- widefield vs confocal (Supplementary Note 1)
#
# Inputs come from the image-processing pipeline (illumination correction,
# stitching, segmentation, cell matching, crops); results are written beside them.
MICROSCOPE_ROOT = ROOTS["MICROSCOPE_DATA"]
MICROSCOPE_CROPS = MICROSCOPE_ROOT / "crops"                   # <mag>/<well>/<id><channel>.png, crops_index.parquet
MICROSCOPE_MEASURE = MICROSCOPE_ROOT / "measure"               # <well>/<mag>.parquet, per-object measurements
MICROSCOPE_CELLS = MICROSCOPE_ROOT / "benchmark" / "cells_4063.parquet"      # the 21,002 matched cells
MICROSCOPE_ALIGN = MICROSCOPE_ROOT / "benchmark4063c" / "align40x.parquet"   # 40x zoom + rotation onto the 63x window
MICROSCOPE_FEATURES = MICROSCOPE_ROOT / "subcell_features_4063c_nomask"     # <run>/<mag>_features_pooled.pth
MICROSCOPE_DIR = MICROSCOPE_ROOT / "microscope_benchmark"                     # analysis results


# ------------------------------------------------------------------ helpers

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def expand(value):
    """Substitute ``${NAME}`` placeholders with the matching root.

    Applied recursively to dicts and lists; other values are returned unchanged.
    An unknown name raises ``KeyError``.
    """
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: str(ROOTS[m.group(1)]), value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def load_yaml(path):
    """Read a configuration file and expand its ``${NAME}`` placeholders."""
    import yaml

    with open(path) as f:
        return expand(yaml.safe_load(f))


def ensure(path) -> Path:
    """Create *path* as a directory if needed and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def features(model: str, dataset: str = "hpa_features", harmonized: str = "none") -> Path:
    """Path to a model's cell-level feature tensor.

    ``harmonized`` names the Harmony batch key; ``"none"`` is the uncorrected
    matrix, and the corrected matrices sit beside it.
    """
    p = FEATURE_DIR / model / dataset / "all_features.pth"
    if harmonized != "none":
        p = p.with_name(f"harmonized_features_{harmonized}.pth")
    return p


def figure_dir(subdir: str | None = None) -> Path:
    """Output directory for a figure, created on demand."""
    return ensure(FIG_DIR / subdir if subdir else FIG_DIR)


def describe() -> str:
    """Resolved roots, one per line, flagging those that do not exist."""
    return "\n".join(f"{k:<18} {v}{'' if Path(v).exists() else '   (missing)'}"
                     for k, v in ROOTS.items())


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print(describe())
