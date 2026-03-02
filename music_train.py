import argparse
import os
import warnings
import shutil

import pandas as pd

from cellmaps_coembedding.runner import CellmapsCoEmbedder, MuseCoEmbeddingGenerator

from tqdm import tqdm

from utils.feature_analysis import get_per_gene_mean_features

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
import matplotlib

matplotlib.use("Agg")

EXP_NAME_DIR_DICT = {
    row["experiment_cell_line"]: row["path"]
    for i, row in pd.read_csv("annotations/mean_features_paths.csv").iterrows()
}
PPI_PATH = "/scratch/groups/emmalu/subcell_ankit/features/ppi/ppi_emd.tsv"


def save_gene_feature_data(save_folder):
    ppi_df = pd.read_csv(PPI_PATH, sep="\t", index_col=0)
    gene2uniprot_df = pd.read_csv("annotations/gene2uniprot.tsv", sep="\t", index_col=0)
    gene2uniprot_df = gene2uniprot_df[gene2uniprot_df["Reviewed"] == "reviewed"]
    ppi_df = pd.merge(
        ppi_df,
        gene2uniprot_df[["Entry"]],
        left_index=True,
        right_index=True,
    )
    ppi_genes = sorted(list(set(ppi_df["Entry"].to_list())))
    ppi_df = ppi_df.groupby("Entry").mean().reset_index()
    ppi_df = ppi_df.set_index("Entry")

    for method, result_path in EXP_NAME_DIR_DICT.items():
        method_df = pd.read_csv(
            result_path,
            sep=("\t" if result_path.endswith(".tsv") else ","),
            index_col=0,
        )
        method_df = method_df[method_df.index.isin(ppi_genes)]
        ppi_save_df = ppi_df[ppi_df.index.isin(method_df.index.to_list())]

        method_save_folder = f"{save_folder}/{method}"
        os.makedirs(method_save_folder, exist_ok=True)
        method_image_emd_folder = f"{method_save_folder}/img_data"
        os.makedirs(method_image_emd_folder, exist_ok=True)
        method_df.to_csv(
            f"{method_image_emd_folder}/image_emd.tsv",
            sep="\t",
            index=True,
        )

        method_ppi_save_folder = f"{save_folder}/{method}/ppi_data"
        os.makedirs(method_ppi_save_folder, exist_ok=True)
        ppi_save_df.to_csv(
            f"{method_ppi_save_folder}/ppi_emd.tsv",
            sep="\t",
            index=True,
        )


def process_single_experiment(exp_name, save_folder, hidden_dim, n_k):
    """Process a single experiment."""
    try:
        print(f"Processing {exp_name}...")

        embeddingdir = f"{save_folder}/{exp_name}"
        outdir = f"{embeddingdir}/coembedding_k_{n_k}_d_{hidden_dim}"
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)

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

        x = CellmapsCoEmbedder(
            outdir=outdir,
            inputdirs=[ppi_embeddingdir, image_embeddingdir],
            embedding_generator=gen,
        )

        x.run()
        return f"Successfully processed {exp_name}"
    except Exception as e:
        return f"Error processing {exp_name}: {str(e)}"


def generate_coembedding_features_sequential(save_folder, hidden_dim, n_k):
    """Sequential processing as fallback."""
    print("Generating co-embedding features (sequential)...")

    results = []
    for exp_name in tqdm(EXP_NAME_DIR_DICT.keys(), desc="Processing experiments"):
        result = process_single_experiment(exp_name, save_folder, hidden_dim, n_k)
        results.append(result)
        print(result)

    print(f"Completed processing {len(results)} experiments")


def main(save_folder, hidden_dim, n_k):
    save_folder = "/scratch/groups/emmalu/subcell_ankit/subcell_results/music_eval"
    os.makedirs(save_folder, exist_ok=True)

    # save_gene_feature_data(save_folder)
    generate_coembedding_features_sequential(
        save_folder, hidden_dim=hidden_dim, n_k=n_k
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--save_folder",
        help="Folder to save results in",
        type=str,
        default="/scratch/groups/emmalu/subcell_ankit/subcell_results/music_eval",
    )
    args.add_argument(
        "--dimensions", help="Hidden coembedding dimensions", type=int, default=128
    )
    args.add_argument(
        "--n_k", help="number of k-nearest neighbours", type=int, default=10
    )
    args = args.parse_args()
    main(save_folder=args.save_folder, hidden_dim=args.dimensions, n_k=args.n_k)
