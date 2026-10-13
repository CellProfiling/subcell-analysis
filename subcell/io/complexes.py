"""Protein group memberships for function prediction and interaction enrichment.

Used by ``subcell.classification.function_prediction`` (GO / Reactome term
membership, Fig. 3d and Extended Data Fig. 8c) through
:func:`load_protein_complex_data_cls`, and by
``subcell.multimodal.interaction_enrichment`` (CORUM, hu.MAP, STRING, BioPlex
interactions, Fig. 6d and Extended Data Fig. 7b) through
:func:`load_protein_complex_data_embed`. Both read tables under
``config.COMPLEXES_DIR``, which are not version-controlled:

====================================  ========================================
``go_{bp,cc,mf}_hierarchy.csv``       Gene Ontology (release 2025-03-16), terms by level
``reactome_hierarchy.csv``            Reactome V92, pathways by level
``corum_humanComplexes.json``         CORUM 5.1
``Complexes_huMAP.csv``               hu.MAP 2.0
``bioplex_ppi_uniprot.csv``           BioPlex 3.0, HEK293T network
``bioplex_u2os_ppi_uniprot.csv``      BioPlex 3.0, U2OS network
``string_only_ppi_uniprot.csv``       STRING v12.0 (score > 700) pairs absent from BioPlex HEK293T
====================================  ========================================

The BioPlex and STRING tables are built from the raw downloads in ``PPI_DIR``
(``<repository>/ppi_data``) by::

    python -m subcell.io.complexes

which needs ``BioPlex_293T_Network_*.tsv`` (and ``BioPlex_U2OS_Network*.tsv`` for the
U2OS table), ``9606.protein.links.v12.0.txt`` from STRING, and ``ensp2uniprot.tsv``,
a UniProt ID-mapping export (Ensembl Protein -> UniProtKB with ``From``, ``Entry``
and ``Reviewed`` columns). The other tables are used as downloaded or derived from
the sources above.
"""

import json
from glob import glob

import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer

from subcell import config

PPI_DIR = config.PPI_DIR
BIOPLEX_NETWORKS = {                      # output name -> raw network file pattern
    "bioplex": "BioPlex_293T_Network_*.tsv",
    "bioplex_u2os": "BioPlex_U2OS_Network*.tsv",
}


# ------------------------------------------------------- building the tables


def load_and_map_bioplex_ppi(name="bioplex"):
    """Write ``<name>_ppi_uniprot.csv``: UniProt pairs of one BioPlex network
    (isoform suffixes stripped, unmapped proteins dropped)."""
    matches = glob(str(PPI_DIR / BIOPLEX_NETWORKS[name]))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file matching {BIOPLEX_NETWORKS[name]} in {PPI_DIR}, found {len(matches)}")
    path = matches[0]
    ppi_data = pd.read_csv(path, sep="\t")
    ppi_data = ppi_data.rename(columns={"UniprotA": "protein1", "UniprotB": "protein2"})
    ppi_data = ppi_data[["protein1", "protein2"]]
    ppi_data["protein1"] = ppi_data["protein1"].apply(lambda x: x.split("-")[0])
    ppi_data["protein2"] = ppi_data["protein2"].apply(lambda x: x.split("-")[0])
    ppi_data = ppi_data[ppi_data["protein1"] != "UNKNOWN"]
    ppi_data = ppi_data[ppi_data["protein2"] != "UNKNOWN"]
    ppi_data.to_csv(config.ensure(config.COMPLEXES_DIR) / f"{name}_ppi_uniprot.csv",
                    index=False)
    return ppi_data


def load_and_map_string_ppi(threshold=700):
    """Write ``string_only_ppi_uniprot.csv``: STRING pairs with combined score above
    *threshold*, mapped to reviewed UniProt entries, that are not BioPlex HEK293T
    pairs. Needs ``bioplex_ppi_uniprot.csv`` (built here)."""
    ppi_data = pd.read_csv(PPI_DIR / "9606.protein.links.v12.0.txt", sep=" ")
    ppi_data["protein1"] = ppi_data["protein1"].apply(lambda x: x.replace("9606.", ""))
    ppi_data["protein2"] = ppi_data["protein2"].apply(lambda x: x.replace("9606.", ""))

    ensp2uniprot = pd.read_csv(PPI_DIR / "ensp2uniprot.tsv", sep="\t")
    ensp2uniprot = ensp2uniprot[ensp2uniprot["Reviewed"] == "reviewed"].reset_index(
        drop=True
    )
    ensp2uniprot = ensp2uniprot.groupby("From")["Entry"].apply(list).reset_index()
    ppi_data["protein1"] = ppi_data["protein1"].map(
        ensp2uniprot.set_index("From")["Entry"]
    )
    ppi_data["protein2"] = ppi_data["protein2"].map(
        ensp2uniprot.set_index("From")["Entry"]
    )
    ppi_data = ppi_data.dropna().reset_index(drop=True)
    ppi_data = ppi_data.explode("protein1").reset_index(drop=True)
    ppi_data = ppi_data.explode("protein2").reset_index(drop=True)
    ppi_data = ppi_data[ppi_data["combined_score"] > threshold].reset_index(drop=True)
    ppi_data = ppi_data[["protein1", "protein2"]]
    bioplex_ppi_data = load_and_map_bioplex_ppi("bioplex")
    string_only_data = ppi_data[
        ~ppi_data.apply(tuple, axis=1).isin(bioplex_ppi_data.apply(tuple, axis=1))
    ].reset_index(drop=True)
    string_only_data.to_csv(
        config.ensure(config.COMPLEXES_DIR) / "string_only_ppi_uniprot.csv", index=False
    )
    return string_only_data


def main(string_threshold=700):
    """Build the BioPlex tables whose raw network is present, and the STRING table."""
    for name, pattern in BIOPLEX_NETWORKS.items():
        if glob(str(PPI_DIR / pattern)):
            load_and_map_bioplex_ppi(name)
    load_and_map_string_ppi(string_threshold)


# --------------------------------------------------------- reading the tables


def _filter_complexes(df, min_genes, max_categories):
    """Keep genes with at most *max_categories* complexes and complexes with at
    least *min_genes* such genes. *df* has one row per complex: ``Complex`` and
    ``Genes`` (a list). Returns one row per gene with its list of complexes."""
    df = df[["Complex", "Genes"]]

    df = df.explode("Genes").groupby("Genes").agg(list).reset_index()
    if max_categories is not None:
        df = df[df["Complex"].apply(len) <= max_categories].reset_index(drop=True)

    df = df.explode("Complex").groupby("Complex").agg(list).reset_index()
    df = df[df["Genes"].apply(len) >= min_genes].reset_index(drop=True)

    df = df.explode("Genes").groupby("Genes").agg(list).reset_index()
    df["n_complexes"] = df["Complex"].apply(len)
    return df


def _hierarchy_members(df, level_column, common_genes, split_terms=False):
    """Complexes (terms) of one hierarchy level with their member genes in
    *common_genes*; ``split_terms`` splits ``;``-joined term ids onto rows."""
    df = df.drop(columns=[x for x in df.columns if x not in ["UniProt", level_column]])
    df = df.rename(columns={level_column: "Complex"})
    df = df[df["Complex"].notna()].reset_index(drop=True)
    if split_terms:
        df["Complex"] = df["Complex"].apply(lambda x: x.split(";"))
        df = df.explode("Complex")

    df = df.groupby("Complex").agg(lambda x: list(set(x))).reset_index()
    df["Genes"] = df["UniProt"].apply(lambda x: [gene for gene in x if gene in common_genes])
    return df


def get_CORUM_complexes(common_genes, min_genes=5, max_categories=5):
    with open(config.COMPLEXES_DIR / "corum_humanComplexes.json") as f:
        prot_complex = json.load(f)

    complex_name = []
    complex_genes = []
    for complex in prot_complex:
        name = str(complex["complex_id"])
        genes = [subunit["swissprot"]["uniprot_id"] for subunit in complex["subunits"]]
        genes = [gene for gene in genes if gene is not None]

        common_genes_intersect = list(set(genes).intersection(set(common_genes)))
        if len(common_genes_intersect) == 0:
            continue

        complex_name.append(name)
        complex_genes.append(common_genes_intersect)
    prot_complex_df = pd.DataFrame({"Complex": complex_name, "Genes": complex_genes})
    return _filter_complexes(prot_complex_df, min_genes, max_categories)


def get_HUMAP_complexes(common_genes, min_genes=5, max_categories=5):
    prot_complex_df = pd.read_csv(config.COMPLEXES_DIR / "Complexes_huMAP.csv")
    prot_complex_df["Genes"] = (
        prot_complex_df["Uniprot_ACCs"]
        .apply(lambda x: x.split(" "))
        .apply(lambda x: [k for k in x if k in common_genes])
    )
    prot_complex_df = prot_complex_df.rename(columns={"HuMAP2_ID": "Complex"})
    return _filter_complexes(prot_complex_df, min_genes, max_categories)


def get_REACTOME_complexes(hierarchy_level, common_genes, min_genes=5, max_categories=5):
    prot_complex_df = pd.read_csv(config.COMPLEXES_DIR / "reactome_hierarchy.csv")
    prot_complex_df = _hierarchy_members(
        prot_complex_df, f"Level_{hierarchy_level}", common_genes
    )
    return _filter_complexes(prot_complex_df, min_genes, max_categories)


def get_go_complexes(hierarchy_level, common_genes, complex_name, min_genes=5,
                     max_categories=5):
    prot_complex_df = pd.read_csv(config.COMPLEXES_DIR / f"{complex_name}_hierarchy.csv")
    prot_complex_df = _hierarchy_members(
        prot_complex_df, f"level_{hierarchy_level}", common_genes, split_terms=True
    )
    return _filter_complexes(prot_complex_df, min_genes, max_categories)


def load_complex_data(complex_name, common_genes, hierarchy_level, min_genes=10,
                      max_categories=5):
    """Genes in *common_genes* with the ``; ``-joined complexes (or GO / Reactome
    terms) they belong to. ``complex_name`` is ``corum``, ``humap``, ``reactome`` or
    ``go_bp`` / ``go_cc`` / ``go_mf``; ``hierarchy_level`` applies to the last two
    kinds."""
    if complex_name == "corum":
        prot_df = get_CORUM_complexes(common_genes, min_genes, max_categories)
    elif complex_name == "humap":
        prot_df = get_HUMAP_complexes(common_genes, min_genes, max_categories)
    elif complex_name == "reactome":
        prot_df = get_REACTOME_complexes(
            hierarchy_level, common_genes, min_genes, max_categories
        )
    elif complex_name.startswith("go"):
        prot_df = get_go_complexes(
            hierarchy_level, common_genes, complex_name, min_genes, max_categories
        )
    else:
        raise ValueError("Invalid complex name")

    prot_df["Complex"] = prot_df["Complex"].apply(lambda x: "; ".join(x))
    return prot_df


def load_protein_complex_data_cls(common_genes, complex_name, hierarchy_level):
    """Multi-hot gene x term table (terms with >= 10 genes, genes in <= 5 terms)
    with a ``Fold`` column from a ten-fold iterative stratification."""
    prot_complex_df = load_complex_data(
        complex_name,
        common_genes,
        hierarchy_level,
        min_genes=10,
        max_categories=5,
    )

    prot_complex_df["Complex"] = prot_complex_df["Complex"].apply(
        lambda x: x.split("; ")
    )
    mlb = MultiLabelBinarizer()
    multi_hot_encoding = mlb.fit_transform(list(prot_complex_df["Complex"]))

    istratf = IterativeStratification(n_splits=10, order=5)
    for i, (train_idxs, test_idxs) in enumerate(
        istratf.split(prot_complex_df["Genes"].values, multi_hot_encoding)
    ):
        prot_complex_df.loc[test_idxs, "Fold"] = i + 1

    prot_complex_df["Fold"] = prot_complex_df["Fold"].astype(int)

    prot_df = pd.DataFrame(multi_hot_encoding, columns=mlb.classes_)
    prot_df.loc[:, "Genes"] = prot_complex_df["Genes"]
    prot_df.loc[:, "Fold"] = prot_complex_df["Fold"]
    return prot_df


def load_protein_complex_data_embed(common_genes, complex_name, hierarchy_level):
    """Symmetric gene x gene 0/1 interaction matrix: two genes interact when they
    share a complex (``corum``, ``humap``, GO / Reactome terms) or are a listed pair
    (``string``, ``bioplex``, ``bioplex_u2os``)."""
    if complex_name not in ["string", "bioplex", "bioplex_u2os"]:
        prot_complex_df = load_complex_data(
            complex_name,
            common_genes,
            hierarchy_level,
            min_genes=3,
            max_categories=None,
        )
        interaction_matrix = np.zeros(
            (len(prot_complex_df["Genes"]), len(prot_complex_df["Genes"]))
        )
        for i, row in prot_complex_df.iterrows():
            complexes = row["Complex"].split("; ")
            for complex in complexes:
                other_genes_idx = prot_complex_df[
                    prot_complex_df["Complex"].apply(lambda x: complex in x)
                ].index
                interaction_matrix[i, other_genes_idx] = 1
        genes_unique = prot_complex_df["Genes"].tolist()
    else:
        table = "string_only" if complex_name == "string" else complex_name
        prot_complex_df = pd.read_csv(config.COMPLEXES_DIR / f"{table}_ppi_uniprot.csv")
        prot_complex_df = prot_complex_df[
            prot_complex_df["protein1"].isin(common_genes)
        ].reset_index(drop=True)
        prot_complex_df = prot_complex_df[
            prot_complex_df["protein2"].isin(common_genes)
        ].reset_index(drop=True)
        prot_complex_df = prot_complex_df.drop_duplicates()
        genes_unique = list(
            set(prot_complex_df["protein1"].tolist()).union(
                set(prot_complex_df["protein2"].tolist())
            )
        )
        interaction_matrix = np.zeros((len(genes_unique), len(genes_unique)))
        gene2idx = {gene: i for i, gene in enumerate(genes_unique)}
        prot_complex_df["protein1"] = prot_complex_df["protein1"].apply(
            lambda x: gene2idx[x]
        )
        prot_complex_df["protein2"] = prot_complex_df["protein2"].apply(
            lambda x: gene2idx[x]
        )
        interaction_matrix[
            prot_complex_df["protein1"].values, prot_complex_df["protein2"].values
        ] = 1

    interaction_matrix = interaction_matrix + interaction_matrix.T
    interaction_matrix[interaction_matrix > 1] = 1

    interaction_df = pd.DataFrame(
        interaction_matrix, index=genes_unique, columns=genes_unique
    )
    return interaction_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--string-threshold", type=int, default=700,
                        help="keep STRING pairs with combined score above this")
    main(parser.parse_args().string_threshold)
