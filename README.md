# subcell-analysis

Analysis and figure code for

> Gupta, A. *et al.* Proteome-aware vision foundation models for microscopy capture cell biology.

The SubCell models are trained with
[CellProfiling/subcell-embed](https://github.com/CellProfiling/subcell-embed) and run on new
images with [czi-ai/SubCellPortable](https://github.com/czi-ai/SubCellPortable). This
repository reproduces the evaluations, analyses and figures of the paper from the model
weights and the public datasets.

## Contents

```
subcell/                     importable package
  config.py                  every filesystem path, from a few named roots (see "Paths")
  naming.py                  display names and model orderings used in the figures
  runlog.py                  timestamped progress log
  utils/                     helpers shared across packages (seeding, losses, metrics, ...)
  io/                        features, annotation tables, image datasets, GO/Reactome/PPI tables
  models/                    SubCell and baseline encoders, probe heads, MIRAGE
  style/                     Nature figure specification: type, geometry, colour, saving
  inference/                 GPU feature and attention extraction, one module per dataset
  classification/            probes on frozen features: localization, cell cycle, function prediction
  feature_analysis/          harmonization, mutual information, retrieval, similarity, attention
  perturbation/              CM4AI drug prediction and perturbation signatures
  multimodal/                image + sequence co-embedding (MuSIC, MIRAGE) and its evaluation
  microscope_robustness/     the widefield vs confocal comparison (Supplementary Note 1)
  tissue/                    the tissue microarray analysis (Supplementary Note 2)
figures/                     one module per figure, rendering cached results
notebooks/                   the multiscale cell map (Fig. 5)
jump/                        the JUMP-Pilot benchmark (Fig. 4e-h; see jump/README.md)
configs/                     inference configurations (models, datasets, channels)
scripts/                     data preparation and Slurm job scripts
annotations/                 small curated tables the results depend on
LICENSES/                    licences of the third-party code included here
paths.example.toml           template for paths.local.toml (see "Paths")
```

## System requirements

- Linux (tested on CentOS 7, kernel 3.10), Python 3.12.
- The dependencies and the versions the results were produced with are listed in
  `pyproject.toml`.
- Feature extraction, attention extraction and the probe benchmarks need an NVIDIA GPU
  (tested on A100 80 GB and H100 80 GB, CUDA 12.4). The figures and most analyses of cached
  results run on a CPU.
- Memory: up to 250 GB of RAM for the tissue UMAPs and for Harmony over all HPA cells; most
  steps need less than 64 GB. Disk: about 30 GB of features per model and dataset.

## Installation

```bash
git clone --recurse-submodules https://github.com/CellProfiling/subcell-analysis
cd subcell-analysis
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # add ".[multimodal]" for MuSIC and MIRAGE
cp paths.example.toml paths.local.toml   # then edit the paths
```

Installation with pip typically takes 5-15 minutes, mostly downloading PyTorch. Cell-DINO additionally needs a checkout of
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) (`DINOV2_REPO`).

## Paths

Every path is derived from a few roots, read from an environment variable of the same name,
else from `paths.local.toml` (not version-controlled), else from a default under `data/` or
`results/` in the checkout. `python -m subcell.config` prints the resolved roots.

| Root | Contents |
| --- | --- |
| `SUBCELL_ROOT` | features, embeddings and cached analysis results |
| `SUBCELL_FIG_DIR` | rendered figures (default `$SUBCELL_ROOT/figures`) |
| `SUBCELL_CACHE` | library caches: Hugging Face, matplotlib, the copairs null distribution (default `$SUBCELL_ROOT`) |
| `SUBCELL_WEIGHTS` | SubCell weights, one file per channel variant (e.g. `all_channels_MAE-CellS-ProtS-Pool.pth`) |
| `BASELINE_WEIGHTS` | weights of the compared models: `dino_weights/` (DINO4Cells-HPA), `cell_dino/` (Cell-DINO-HPA), `bestfitting/`, `vit-supervised/` (ViT-Weak-Supervised) |
| `SUBCELL_FONT_DIR` | Arial- or Helvetica-metric fonts, e.g. Liberation Sans |
| `DINOV2_REPO` | dinov2 checkout, for Cell-DINO |
| `HPA_DATA`, `KAGGLE_DATA`, `CM4AI_DATA`, `ALLENCELL_DATA`, `OPENCELL_DATA`, `YEAST_DATA`, `FUCCI_DATA`, `MICROSCOPE_DATA` | the image datasets |

The files in `configs/` refer to these roots as `${NAME}`.

## Data

- SubCell weights and embeddings: <https://virtualcellmodels.cziscience.com/>. The compared
  models are those of their publications: DINO4Cells-HPA (Doron et al.), Cell-DINO-HPA
  (facebookresearch/dinov2), bestfitting (winner of the HPA Kaggle single-cell challenge) and
  ViT-Weak-Supervised (trained with the SubCell code).
- HPA v23 single-cell crops: `s3://czi-subcell-public/`;
  `python -m scripts.build_hpa_cell_metadata` writes their `crops_metadata.csv`. From
  <https://www.proteinatlas.org>: the image table (`annotations/IF-image-orig-bboxes.csv`)
  and the cell-line RNA data (`annotations/rna_celline.tsv`, HPA v24). The antibody split
  used for training and evaluation is in `annotations/splits/`.
- HPA Kaggle single-cell challenge test set:
  <https://www.kaggle.com/competitions/hpa-single-cell-image-classification>;
  `python -m scripts.build_kaggle_metadata` writes its metadata and
  `python -m scripts.export_kaggle_features` files its features beside the HPA ones.
- OpenCell: <https://opencell.czbiohub.org>. Allen Cell: <https://www.allencell.org>. Yeast:
  <https://github.com/BooneAndrewsLab/CycleNET>. FUCCI U2OS: BioImage Archive S-BIAD34.
  CM4AI (June 2025 release): <https://doi.org/10.18130/V3/F3TD5R>. The widefield/confocal
  images (Supplementary Note 1) and the tissue microarray images (Supplementary Note 2): see
  the Data availability statement of the paper.
- Gene and protein identifiers: `annotations/gene2uniprot.tsv`; `annotations/ensg2uniprot.tsv`
  and `ppi_data/ensp2uniprot.tsv` are UniProt ID-mapping exports (Ensembl gene and Ensembl
  protein to UniProtKB, 2025).
- Protein groups and interactions (`annotations/complexes/`): Gene Ontology release
  2025-03-16, Reactome V92, CORUM 5.1, hu.MAP 2.0, BioPlex 3.0 and STRING v12.0.
  `python -m subcell.io.complexes` builds the BioPlex and STRING tables from the raw
  downloads in `ppi_data/`; its docstring lists every table and the file it is read from.

## Reproducing the figures

Each figure has a module in `figures/` that renders cached results; the results are written
by the analysis modules listed beside it, which in turn read the features written by
`subcell.inference`. Every runnable module has a `--help`.

```bash
python -m figures.make_all --list      # every figure, and whether its inputs exist
python -m figures.make_all             # render all figures whose inputs exist
python -m figures.fig3c_mantel         # one figure
```

| Figure | Figure module | Analysis | Features |
| --- | --- | --- | --- |
| Fig. 2a-c, Supplementary Table 9 | — (tables from `classification.localization`) | `subcell.classification.localization` | `inference.hpa` with `config_subcell_*.yaml`, `config_kaggle.yaml` |
| Fig. 2f,h,j, Supplementary Table 1 | — (tables) | `subcell.classification.yeast` | `inference.yeast` |
| Fig. 2g,i, Supplementary Table 10 | `fig2_cell_cycle` | `subcell.classification.cellcycle_allencell` | `inference.allencell_crops`, `inference.allencell` |
| Fig. 2k | `fig2_cell_cycle` | `subcell.classification.cellcycle_fucci` | `inference.fucci_crops`, `inference.fucci` |
| Fig. 2l | — (predictions from `classification.cellcycle_fucci`) | `subcell.classification.cellcycle_fucci` | as above |
| Fig. 3a,b; Extended Data Fig. 8a,b | `fig3ab_similarity` | `subcell.feature_analysis.similarity` | `inference.hpa` |
| Fig. 3c; Extended Data Fig. 8e | `fig3c_mantel` | `subcell.feature_analysis.similarity` | `inference.hpa` |
| Fig. 3d; Extended Data Fig. 8c | `fig3d_function_prediction` | `subcell.feature_analysis.harmonization`, then `subcell.classification.function_prediction` (`scripts/function_prediction.sbatch`) | `inference.hpa` |
| Fig. 3f,g; Extended Data Fig. 8f,g | `fig3fg_attention` | `subcell.feature_analysis.attention_correlation` | `inference.attention` with `config_attention_hpa.yaml` |
| Fig. 4a | — (table) | `subcell.perturbation.cm4ai_drug_prediction crops`, then `predict` | `inference.hpa` with `config_b2ai.yaml` |
| Fig. 4b, the tables behind Fig. 4c; Extended Data Fig. 5 | `fig4_perturbation` | `subcell.perturbation.cm4ai_signatures` | `inference.hpa` with `config_b2ai.yaml` |
| Fig. 6a | `fig6a_cluster_transfer` | `subcell.multimodal.cluster_transfer` (`subcell/multimodal/slurm/cluster_transfer.sbatch`) | `multimodal.music` (the image, ESM-2 and co-embedding of one run) |
| Extended Data Fig. 7a | `ed7a_cluster_percentile` | `subcell.multimodal.cluster_transfer`, then `subcell.multimodal.cluster_percentile` | as above |
| Fig. 6d; Extended Data Fig. 7b | `ed7b_interaction_enrichment` | `subcell.multimodal.music`, `subcell.multimodal.mirage`, `subcell.multimodal.interaction_enrichment` | `feature_analysis.harmonization` |
| Fig. 4e-h; Extended Data Figs. 6 and 9 | `jump/notebooks/` | the JUMP-Pilot pipeline in `jump/` (own environments and configuration; see `jump/README.md`) | `jump/scripts/inference_*.py` |
| Fig. 5 | `notebooks/fig5_multiscale_cell_map.ipynb` | hierarchical Leiden clustering and g:Profiler enrichment of the U2OS protein embeddings, in the notebook | Harmony-corrected SubCell features |
| Extended Data Fig. 1 | `ed1_microscopes` | `python -m subcell.microscope_robustness` | `inference.microscopes` with `config_cellcycle_microscopes.yaml` |
| Extended Data Fig. 2i | `ed2i_tissue_umaps`, `ed2i_cluster_umap` | `subcell.tissue.harmonization` (`scripts/tissue_*.sbatch`), `subcell.tissue.kmeans_k13` | SubCell features of the TMA cells (SubCellPortable) |
| Extended Data Fig. 2j,k | `ed2jk_cluster_annotation` | `subcell.tissue.kmeans_k13`, `subcell.tissue.reference_labels`, `subcell.tissue.prevalence_support` | as above |
| Extended Data Fig. 3 | `ed3_harmonization` | `subcell.feature_analysis.harmonization`, `.retrieval`, `.mutual_info` | `inference.hpa` |

Not produced by this repository:

- Fig. 1, Fig. 3e and Extended Data Figs. 4 and 8d are schematics or image panels; the
  attention images are written by the `subcell.inference` modules when `save_attention` is
  set in their configuration.
- Fig. 2d (a decision tree), Fig. 4d (images) and the joint intensity-versus-embedding plot of
  Fig. 4c. Fig. 2e (OpenCell clustering) has its features (`inference.opencell`) but no
  analysis code here.
- Fig. 6b,c,e,f, Extended Data Fig. 7c.
- Extended Data Fig. 2a-h,l (core segregation, dispersion, vimentin classification and the
  image panels).

## Figure style

`subcell/style/` encodes the Nature figure specification: 5-7 pt sans-serif type, 0.25-1 pt
lines, 89/120/183 mm widths, Type 42 fonts, and the 180 x 170 mm Extended Data page.
`style.save` checks the rendered figure against it before writing.
`python -m subcell.style.palettes` reports the colour separation of the palettes under
simulated colour-vision deficiencies.

## License

MIT; see `LICENSE`. Code adapted from other projects keeps its licence, reproduced in
`LICENSES/`: `subcell/models/vision_transformer.py` (DINO, Apache 2.0),
`subcell/models/inception_v3.py` (torchvision, BSD 3-Clause),
`subcell/models/cls_inception_v3.py` (HPA Single Cell Classification winning solution,
MIT), `subcell/models/mirage/` (pytorch-CycleGAN-and-pix2pix, BSD 2-Clause) and
`subcell/perturbation/spherize.py` (pycytominer, BSD 3-Clause).
