"""Similarity enrichment of interacting protein pairs (Fig. 6d, Extended Data Fig. 7b).

For every row of ``methods_info.csv`` and every embedding it lists (image, sequence,
MuSIC co-embedding, MIRAGE co-embedding) plus random-vector baselines, proteins are
compared by cosine similarity (rescaled to [0, 1]). The pairs of a database
(CORUM, hu.MAP, STRING, BioPlex) split into interacting and non-interacting; an
enrichment is the approximated Cliff's delta between the two similarity
distributions, estimated 1,000 times from 10,000 pairs sampled with replacement
from each set (``2 * (U / (n1 * n2) - 0.5)`` of a one-sided Mann-Whitney test).

    python -m subcell.multimodal.interaction_enrichment [--save-folder DIR]
        [--complex-name corum humap string bioplex bioplex_u2os] [--hierarchy-level 0]

Per row and database, ``<save_folder>/<Method_Name>/<complex>_level_<h>/`` receives
``<embedding>_results.csv`` (the 1,000 enrichments), ``<embedding>_kde_results.csv``
(all pair similarities), ``all_results.csv`` and ``all_results_mean.csv``; the
interaction matrices are cached as ``<complex>_level_<h>_cell_line_<cell line>.csv``
in ``<save_folder>``. ``figures.ed7b_interaction_enrichment`` plots the results.
The default folder is ``config.MULTIMODAL_ENRICHMENT_DIR``.
"""

from __future__ import annotations

import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from subcell import config
from subcell.io.complexes import load_protein_complex_data_embed
from subcell.io.metadata import read_methods_csv
from subcell.naming import RANDOM_BASELINE_DIMS

NUM_SAMPLINGS = 1000
NUM_EDGES = 10000


def evaluation_table() -> pd.DataFrame:
    """The rows of ``methods_info.csv`` to evaluate, sorted by (Method, Cell_Line, Harmonized).

    The sequence embedding and the random baselines do not depend on the image
    model or on the harmonization, so they are evaluated once per cell line, on its
    first row (``Seq_Embed`` is blanked on the others).
    """
    df = read_methods_csv()
    df = df.sort_values(["Method", "Cell_Line", "Harmonized"]).reset_index(drop=True)
    df.loc[df.duplicated("Cell_Line"), "Seq_Embed"] = np.nan
    return df


def read_embedding(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", index_col=0)


def cosine_similarity_scaled(features):
    """Cosine similarity between each pair of rows, rescaled into [0, 1]."""
    sim = cosine_similarity(features)
    sim -= sim.min()
    sim /= sim.max()
    return sim


def pair_similarities(data, gene_index, prot_df):
    """Similarity of every gene pair (upper triangle) and whether the pair interacts."""
    features = data[[gene_index[gene] for gene in prot_df.index]]
    similarity = cosine_similarity_scaled(features)
    upper_tri_indices = np.triu_indices_from(similarity, k=1)
    return similarity[upper_tri_indices], prot_df.values[upper_tri_indices]


def embed_eval(all_method_data, common_genes, prot_df, save_folder):
    """Enrichment of interacting pairs for each embedding; results are cached per embedding."""
    gene_index = {gene: i for i, gene in enumerate(common_genes)}
    all_result_dfs = []
    for method, data in all_method_data.items():
        print(f"Evaluating {method}", flush=True)
        results_file = f"{save_folder}/{method}_results.csv"
        kde_file = f"{save_folder}/{method}_kde_results.csv"

        sim_values = edges_values = None
        if not os.path.exists(results_file) or not os.path.exists(kde_file):
            sim_values, edges_values = pair_similarities(data, gene_index, prot_df)

        if os.path.exists(results_file):
            print(f"Skipping {method} as data already exists")
            result_df = pd.read_csv(results_file)
        else:
            interacting = sim_values[edges_values == 1]
            non_interacting = sim_values[edges_values == 0]
            results = []
            for _ in tqdm(np.arange(NUM_SAMPLINGS), total=NUM_SAMPLINGS):
                true = random.choices(interacting, k=NUM_EDGES)
                false = random.choices(non_interacting, k=NUM_EDGES)
                u, _ = mannwhitneyu(true, false, alternative="greater")
                effect_size = (u / (len(true) * len(false)) - 0.5) * 2
                results.append({"Embedding": method, "Result": effect_size})
            result_df = pd.DataFrame(results)
            result_df.to_csv(results_file, index=False)
            print(f"Results saved for {method}", flush=True)

        if not os.path.exists(kde_file):
            true = sim_values[edges_values == 1]
            false = sim_values[edges_values == 0]
            pd.DataFrame(
                {
                    "Similarity": np.concatenate([true, false]),
                    "Type": ["Complex pair"] * len(true) + ["non-Complex pair"] * len(false),
                }
            ).to_csv(kde_file)

        all_result_dfs.append(result_df)

    all_result_df = pd.concat(all_result_dfs, ignore_index=True)
    all_result_df.to_csv(f"{save_folder}/all_results.csv", index=False)
    all_result_df.groupby(["Embedding"]).mean().reset_index().to_csv(
        f"{save_folder}/all_results_mean.csv", index=False
    )


def load_row_embeddings(row):
    """Per-gene embeddings of one table row, restricted to the genes common to all of them.

    Returns ``(common_genes, {embedding name: (n_genes, dim) tensor})``. A MIRAGE
    entry (``Mirage_Image_Seq_Embed``) points at ``latent_image.txt``; the latent
    file of every modality in its name is read and the latents are concatenated.
    """
    embeddings = {}
    for col, path in row.items():
        if "Embed" not in col or pd.isna(path):
            continue
        if "Mirage" not in col:
            embeddings[col] = read_embedding(path)
        else:
            modalities = col.split("Mirage_")[1].split("_Embed")[0].split("_")
            embeddings[f"{col}_concat"] = pd.concat(
                [
                    read_embedding(path.replace("_image.txt", f"_{modality.lower()}.txt"))
                    for modality in modalities
                ],
                axis=1,
            )

    common_genes = sorted(set.intersection(*(set(df.index) for df in embeddings.values())))
    data = {
        name: torch.from_numpy(df.loc[common_genes].values).float()
        for name, df in embeddings.items()
    }
    if not pd.isna(row["Seq_Embed"]):
        for dim in RANDOM_BASELINE_DIMS:
            data[f"Random_{dim}"] = torch.randn(len(common_genes), dim)
    return common_genes, data


ALL_COMPLEX_NAMES = ("corum", "humap", "string", "bioplex", "bioplex_u2os")


def main(save_folder=None, complex_names=ALL_COMPLEX_NAMES, hierarchy_level=0):
    warnings.filterwarnings("ignore", category=UserWarning)
    save_folder = config.ensure(save_folder or config.MULTIMODAL_ENRICHMENT_DIR)

    for _, row in evaluation_table().iterrows():
        print(f"Processing {row['Method_Name']}...")
        method_save_folder = config.ensure(f"{save_folder}/{row['Method_Name']}")
        common_genes, all_method_data = load_row_embeddings(row)

        for complex_name in complex_names:
            prot_df_path = (
                f"{save_folder}/{complex_name}_level_{hierarchy_level}_cell_line_{row['Cell_Line']}.csv"
            )
            if not os.path.exists(prot_df_path):
                prot_df = load_protein_complex_data_embed(common_genes, complex_name, hierarchy_level)
                prot_df.to_csv(prot_df_path)
            else:
                prot_df = pd.read_csv(prot_df_path, index_col=0)

            assert (prot_df.values == prot_df.values.T).all(), "Protein complex dataframe is not symmetric"

            print(
                f"Evaluating on {complex_name} at hierarchy level {hierarchy_level}. "
                f"Found {len(prot_df)} proteins with {prot_df.values.sum()} interactions.",
                flush=True,
            )
            method_complex_save_folder = config.ensure(
                f"{method_save_folder}/{complex_name}_level_{hierarchy_level}"
            )
            embed_eval(all_method_data, common_genes, prot_df, method_complex_save_folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save-folder", default=None, help="default: config.MULTIMODAL_ENRICHMENT_DIR")
    ap.add_argument(
        "--complex-name", nargs="+", default=list(ALL_COMPLEX_NAMES),
        choices=list(ALL_COMPLEX_NAMES),
        help="databases of interacting pairs to evaluate",
    )
    ap.add_argument("--hierarchy-level", type=int, choices=[0, 1, 2, 3, 4], default=0,
                    help="level of the complex hierarchy (corum and humap only)")
    a = ap.parse_args()
    main(a.save_folder, a.complex_name, a.hierarchy_level)
