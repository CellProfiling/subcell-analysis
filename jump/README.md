# SubCell JUMP Benchmark

Benchmarks SubCell (MAE and ViT) against Cell-DINO, DINO4Cells, DeepProfiler, and CellProfiler on the JUMP Pilot (cpg0000) dataset. Evaluates morphological embedding quality via replicate and mechanism-of-action (MoA) retrieval using mean Average Precision (mAP).

## Pipeline

```
Raw crops (192x1728 PNGs, 9 channels)
  → Inference (SubCell MAE/ViT, DINO4Cells, Cell-DINO)  [GPU]
  → Prepare DeepProfiler / download CellProfiler
  → Aggregate cells → wells                              [CPU]
  → Postprocess (normalize, feature-select)               [CPU]
  → Compute mAP (replicate + MoA)                        [CPU]
  → Choose best postprocessing config per model          [notebook]
  → Attention maps + UMAPs → figures                     [GPU / CPU, notebooks]
```

All models are evaluated on unmasked inference. Each model's postprocessing config is chosen by average rank over cell types × mAP metrics in `notebooks/visualize_postprocessing.ipynb`, which writes `configs/BEST_POSTPROC_CONFIGS.yaml`. The UMAP launcher and the other notebooks read that file.

## Setup

### 1. Clone

This benchmark lives in the `jump/` folder of `subcell-analysis`. Clone with submodules, then work from `jump/`:

```bash
git clone --recurse-submodules https://github.com/CellProfiling/subcell-analysis.git
cd subcell-analysis/jump
```

Model code lives in git submodules under `jump/models/`: `SubCellPortable`, `DINO4Cells_code`, `DeepProfiler`, and `dinov2` (Cell-DINO, pinned at the commit used). If you cloned without `--recurse-submodules`, run `git submodule update --init` from the repo root.

`results/` and `figures/` start empty; the pipeline below writes into them.

### 2. Environments

| Env | Purpose | Create |
|-----|---------|--------|
| `inference_gpu` | GPU inference (SubCell, DINO4Cells, Cell-DINO) and attention maps | `conda env create -f envs/inference_gpu.yaml` |
| `moa` | Analysis, postprocessing, mAP, UMAPs, notebooks | `conda env create -f envs/analysis.yaml` then `pip install copairs` |

DeepProfiler runs in its own virtualenv (`DEEPPROF_ENV`, below).

For `inference_gpu` on systems with GLIBC < 2.27 (e.g., CentOS 7), see the workaround in `envs/inference_gpu.yaml`.

### 3. Environment Variables

`jump/.env` is sourced by the SLURM launchers (`set -a; source ../.env; set +a`). It is committed as a template: **fill in every empty value for your setup before running anything.** Values written as `${PROJECT_ROOT}/...` work as-is once `PROJECT_ROOT` is set. Don't commit your filled-in paths back.

| Variable | Description |
|----------|-------------|
| `PROJECT_ROOT` | Absolute path to this `jump/` folder |
| `CROP_ROOT` | Path to JUMP cpg0000 pre-cropped single-cell images |
| `DP_ROOT` | Path to DeepProfiler data |
| `EMBED_CELL_ROOT` | Output dir for cell-level embeddings (.pth) |
| `EMBED_WELL_ROOT` | Output dir for well-level CSVs |
| `EMBED_PROCESSED_ROOT` | Output dir for postprocessed parquets |
| `RESULTS_ROOT` | Output dir for mAP results and UMAP coordinates |
| `SUBCELL_MAE_CKPT` | SubCell MAE encoder checkpoint |
| `SUBCELL_MAE_CLF` | SubCell MAE classifier checkpoint |
| `SUBCELL_VIT_CKPT` | SubCell ViT encoder checkpoint |
| `SUBCELL_VIT_CLF` | SubCell ViT classifier checkpoint |
| `DINO_CKPT` | DINO4Cells checkpoint |
| `DINOV2_ROOT` | Path to the dinov2 submodule: `$PROJECT_ROOT/models/dinov2` |
| `CELL_DINO_CKPT` | Cell-DINO Cell Painting checkpoint (`cell_dino_vits8_pretrain_cp-37d20e9c.pth`) |
| `METADATA_ROOT` | Path to `metadata/` folder (included in repo) |
| `MOA_METADATA` | Path to `metadata/usable_moa_metadata.csv` |
| `CONDA_ROOT` | Path to miniconda/anaconda installation |
| `CONDA_ENV_ANALYSIS` | Name of analysis conda env (default: `moa`) |
| `CONDA_ENV_INFERENCE` | Name of GPU inference conda env (default: `inference_gpu`) |
| `DEEPPROF_ENV` | Path to the DeepProfiler virtualenv |
| `CP_S3_BUCKET` | S3 bucket for CellProfiler profiles: `cellpainting-gallery` |
| `CP_S3_PREFIX` | S3 prefix for CellProfiler profiles: `cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1` |

### 4. Data

Input crops are not included in this repo. They must be pre-generated from the JUMP Pilot cpg0000 dataset as 192x1728 uint8 PNGs: 9 channels of 192×192 stacked horizontally, in the order Mito, AGP, RNA, ER, DNA, HighZBF, LowZBF, Brightfield, Mask (cell segmentation). They are read from:

```
$CROP_ROOT/{plate}/{well}/{fov}/crops/crop_{X}_x_{Y}.png
```

`utils/crop_loader.py` unstacks the channels and center-crops to the size each model expects.

## Running

Submit launchers from inside `slurm_launchers/` (they source `../.env` and call `../scripts/...`):

```bash
cd slurm_launchers
sbatch 01_inference_subcell.sh
```

Each launcher is set up to run on all models. Steps in order:

| Step | Launcher / notebook | What it does |
|------|---------------------|--------------|
| 0 | `00_download_subcell_models.sh` | Download SubCell checkpoints |
| 1 | `01_inference_subcell.sh` | SubCell MAE/ViT inference, unmasked and masked |
| 2 | `02_inference_dino.sh`, `02b_inference_cell_dino.sh` | DINO4Cells and Cell-DINO inference, unmasked and masked |
| 3–4 | `03_deepprofiler_data.sh`, `04a`–`04c` (or `04_deepprofiler_run.sh`) | DeepProfiler data prep, profiling, conversion |
| 5 | `05_aggregate.sh` | Aggregate cell → well embeddings (mean, median) |
| 6 | `06_download_cellprofiler_features.sh` | Download CellProfiler well-level features |
| 7 | `07_postprocess.sh` | Full postprocessing grid for every well-level CSV |
| 8 | `08a_compute_map.sh`, then `08b_merge_map_summaries.sh` | mAP per model, merged into `results/results_summary.csv` |
| 9 | `notebooks/visualize_postprocessing.ipynb` | best config per model → `configs/BEST_POSTPROC_CONFIGS.yaml` |
| 10 | `10_attention_maps_{subcell,dino,cell_dino}.sh` | Attention-map figures |
| 11 | `11_compute_umap.sh` | UMAPs for each model's best config (reads the YAML) |
| 12 | `notebooks/per_moa_analysis.ipynb`, `notebooks/visualize_umaps.ipynb` | Per-MoA and UMAP figures |


The attention-map launchers pass `configs/attention_crops_seed0.csv`, which lists the exact crops that were randomly selected when we generated figures for our publication.