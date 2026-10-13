"""MuSIC co-embedding of the image embeddings with a second protein embedding (Fig. 6).

Co-embeds each per-gene mean image embedding listed in ``config.MEAN_FEATURES_CSV``
(``experiment_cell_line`` -> ``path``, with ``${SUBCELL_ROOT}`` placeholders) with
the protein embedding at ``config.PPI_EMBEDDING`` (``ppi_emd.tsv``, the node2vec
protein-protein-interaction embedding), restricted to their common UniProt entries,
using ``cellmaps_coembedding`` (MUSE, triplet margin 0.1).

    python -m subcell.multimodal.music [--save-folder DIR] [--dimensions 128] [--n-k 10]
    python -m subcell.multimodal.music --skip-prepare     # re-run the co-embedding only

Output, per experiment, under ``config.MULTIMODAL_DIR`` (or ``--save-folder``)::

    <experiment>/img_data/image_emd.tsv
    <experiment>/ppi_data/ppi_emd.tsv
    <experiment>/coembedding_k_<n_k>_d_<dimensions>/coembedding_emd.tsv
"""

from __future__ import annotations

import argparse
import shutil
import warnings

import pandas as pd
from tqdm import tqdm

from subcell import config


def mean_feature_paths() -> dict[str, str]:
    """Experiment name -> per-gene mean image embedding file."""
    df = pd.read_csv(config.MEAN_FEATURES_CSV)
    return {row["experiment_cell_line"]: config.expand(row["path"]) for _, row in df.iterrows()}


def save_gene_feature_data(save_folder):
    """Write each experiment's image and PPI embeddings over their shared UniProt entries."""
    ppi_df = pd.read_csv(config.PPI_EMBEDDING, sep="\t", index_col=0)
    gene2uniprot_df = pd.read_csv(config.GENE2UNIPROT, sep="\t", index_col=0)
    gene2uniprot_df = gene2uniprot_df[gene2uniprot_df["Reviewed"] == "reviewed"]
    ppi_df = pd.merge(ppi_df, gene2uniprot_df[["Entry"]], left_index=True, right_index=True)
    ppi_genes = sorted(set(ppi_df["Entry"].to_list()))
    ppi_df = ppi_df.groupby("Entry").mean().reset_index().set_index("Entry")

    for method, result_path in mean_feature_paths().items():
        method_df = pd.read_csv(
            result_path, sep="\t" if result_path.endswith(".tsv") else ",", index_col=0
        )
        method_df = method_df[method_df.index.isin(ppi_genes)]
        ppi_save_df = ppi_df[ppi_df.index.isin(method_df.index.to_list())]

        image_dir = config.ensure(f"{save_folder}/{method}/img_data")
        method_df.to_csv(f"{image_dir}/image_emd.tsv", sep="\t", index=True)
        ppi_dir = config.ensure(f"{save_folder}/{method}/ppi_data")
        ppi_save_df.to_csv(f"{ppi_dir}/ppi_emd.tsv", sep="\t", index=True)


def process_single_experiment(exp_name, save_folder, hidden_dim, n_k):
    """Co-embed one experiment; returns a status message instead of raising."""
    from cellmaps_coembedding.runner import CellmapsCoEmbedder, MuseCoEmbeddingGenerator

    try:
        print(f"Processing {exp_name}...")

        embeddingdir = f"{save_folder}/{exp_name}"
        outdir = f"{embeddingdir}/coembedding_k_{n_k}_d_{hidden_dim}"
        shutil.rmtree(outdir, ignore_errors=True)
        config.ensure(outdir)

        ppi_embeddingdir = f"{embeddingdir}/ppi_data/"
        image_embeddingdir = f"{embeddingdir}/img_data/"

        gen = MuseCoEmbeddingGenerator(
            ppi_embeddingdir=ppi_embeddingdir,
            image_embeddingdir=image_embeddingdir,
            outdir=outdir,
            k=n_k,
            triplet_margin=0.1,
            dimensions=hidden_dim,
        )
        CellmapsCoEmbedder(
            outdir=outdir,
            inputdirs=[ppi_embeddingdir, image_embeddingdir],
            embedding_generator=gen,
        ).run()
        return f"Successfully processed {exp_name}"
    except Exception as e:
        return f"Error processing {exp_name}: {e}"


def main(save_folder=None, hidden_dim=128, n_k=10, prepare=True):
    """Prepare the per-experiment inputs (unless ``prepare`` is False) and co-embed each."""
    warnings.filterwarnings("ignore", category=UserWarning)
    save_folder = config.ensure(save_folder or config.MULTIMODAL_DIR)

    if prepare:
        save_gene_feature_data(save_folder)

    experiments = list(mean_feature_paths())
    for exp_name in tqdm(experiments, desc="Processing experiments"):
        print(process_single_experiment(exp_name, save_folder, hidden_dim, n_k))
    print(f"Completed processing {len(experiments)} experiments")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save-folder", default=None, help="default: config.MULTIMODAL_DIR")
    ap.add_argument("--dimensions", type=int, default=128, help="co-embedding dimensions")
    ap.add_argument("--n-k", type=int, default=10, help="number of nearest neighbours")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="reuse the img_data/ppi_data already written; re-run the co-embedding only")
    a = ap.parse_args()
    main(a.save_folder, a.dimensions, a.n_k, prepare=not a.skip_prepare)
