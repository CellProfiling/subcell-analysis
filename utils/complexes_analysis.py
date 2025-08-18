import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def load_and_map_bioplex_ppi():
    ppi_data = pd.read_csv("ppi_data/BioPlex_293T_Network_10K_Dec_2019.tsv", sep="\t")
    ppi_data = ppi_data.rename(columns={"UniprotA": "protein1", "UniprotB": "protein2"})
    ppi_data = ppi_data[["protein1", "protein2"]]
    ppi_data["protein1"] = ppi_data["protein1"].apply(lambda x: x.split("-")[0])
    ppi_data["protein2"] = ppi_data["protein2"].apply(lambda x: x.split("-")[0])
    ppi_data = ppi_data[ppi_data["protein1"] != "UNKNOWN"]
    ppi_data = ppi_data[ppi_data["protein2"] != "UNKNOWN"]
    ppi_data.to_csv("annotations/complexes/bioplex_ppi_uniprot.csv", index=False)
    return ppi_data


def load_and_map_string_ppi(threshold=700):
    ppi_data = pd.read_csv("ppi_data/9606.protein.links.v12.0.txt", sep=" ")
    ppi_data["protein1"] = ppi_data["protein1"].apply(lambda x: x.replace("9606.", ""))
    ppi_data["protein2"] = ppi_data["protein2"].apply(lambda x: x.replace("9606.", ""))

    ensp2uniprot = pd.read_csv("ppi_data/ensp2uniprot.tsv", sep="\t")
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
    ppi_data.to_csv("annotations/complexes/string_ppi_uniprot.csv", index=False)
    bioplex_ppi_data = load_and_map_bioplex_ppi()
    string_only_data = ppi_data[
        ~ppi_data.apply(tuple, axis=1).isin(bioplex_ppi_data.apply(tuple, axis=1))
    ].reset_index(drop=True)
    string_only_data.to_csv(
        "annotations/complexes/string_only_ppi_uniprot.csv", index=False
    )
    return ppi_data





def get_CORUM_complexes(
    common_genes, min_genes=5, only_common_complexes=False, max_categories=5
):
    prot_complex = json.load(open("annotations/complexes/corum_humanComplexes.json"))

    complex_name = []
    complex_genes = []
    n_genes = []
    for complex in prot_complex:
        name = str(complex["complex_id"])
        genes = [subunit["swissprot"]["uniprot_id"] for subunit in complex["subunits"]]
        genes = [gene for gene in genes if gene is not None]

        common_genes_intersect = list(set(genes).intersection(set(common_genes)))

        if only_common_complexes and (len(common_genes_intersect) != len(genes)):
            continue

        if len(common_genes_intersect) == 0:
            continue

        complex_name.append(name)
        complex_genes.append(common_genes_intersect)
        n_genes.append(len(common_genes_intersect))
    prot_complex_df = pd.DataFrame(
        {"Complex": complex_name, "Genes": complex_genes}  # , "n_genes": n_genes}
    )

    ## Filter genes by number of complexes
    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    if max_categories is not None:
        prot_complex_df = prot_complex_df[
            prot_complex_df["Complex"].apply(len) <= max_categories
        ].reset_index(drop=True)

    ## Filter complexes by number of genes
    prot_complex_df = (
        prot_complex_df.explode("Complex").groupby("Complex").agg(list).reset_index()
    )
    prot_complex_df = prot_complex_df[
        prot_complex_df["Genes"].apply(len) >= min_genes
    ].reset_index(drop=True)

    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    prot_complex_df["n_complexes"] = prot_complex_df["Complex"].apply(len)

    common_genes_corum = prot_complex_df["Genes"].tolist()
    return prot_complex_df, common_genes_corum


def get_HUMAP_complexes(
    common_genes, min_genes=5, only_common_complexes=False, max_categories=5
):
    prot_complex_df = pd.read_csv("annotations/complexes/Complexes_huMAP.csv")
    prot_complex_df["genenames_common"] = (
        prot_complex_df["Uniprot_ACCs"]
        .apply(lambda x: x.split(" "))
        .apply(lambda x: [k for k in x if k in common_genes])
    )

    if only_common_complexes:
        prot_complex_df = prot_complex_df[
            prot_complex_df["genenames"] == prot_complex_df["genenames_common"]
        ]

    prot_complex_df = prot_complex_df.drop(columns=["genenames", "Confidence"])
    prot_complex_df = prot_complex_df.rename(
        columns={"HuMAP2_ID": "Complex", "genenames_common": "Genes"}
    )

    ## Filter genes by number of complexes
    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    if max_categories is not None:
        prot_complex_df = prot_complex_df[
            prot_complex_df["Complex"].apply(len) <= max_categories
        ].reset_index(drop=True)

    ## Filter complexes by number of genes
    prot_complex_df = (
        prot_complex_df.explode("Complex").groupby("Complex").agg(list).reset_index()
    )
    prot_complex_df = prot_complex_df[
        prot_complex_df["Genes"].apply(len) >= min_genes
    ].reset_index(drop=True)

    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    prot_complex_df["n_complexes"] = prot_complex_df["Complex"].apply(len)

    common_genes_humap = prot_complex_df["Genes"].tolist()
    return prot_complex_df, common_genes_humap


def get_REACTOME_complexes(
    heirarchy_level,
    common_genes,
    min_genes=5,
    only_common_complexes=False,
    max_categories=5,
):
    prot_complex_df = pd.read_csv("annotations/complexes/reactome_hierarchy.csv")
    level_column = f"Level_{heirarchy_level}"
    prot_complex_df = prot_complex_df.drop(
        columns=[
            x for x in prot_complex_df.columns if x not in ["UniProt", level_column]
        ]
    ).rename(columns={level_column: "Complex"})
    prot_complex_df = prot_complex_df[prot_complex_df["Complex"].notna()].reset_index(
        drop=True
    )

    prot_complex_df = (
        prot_complex_df.groupby(f"Complex").agg(lambda x: list(set(x))).reset_index()
    )
    prot_complex_df["genes_common"] = prot_complex_df["UniProt"].apply(
        lambda x: [gene for gene in x if gene in common_genes]
    )

    if only_common_complexes:
        prot_complex_df = prot_complex_df[
            prot_complex_df["UniProt"] == prot_complex_df["genes_common"]
        ].reset_index(drop=True)

    prot_complex_df = prot_complex_df.drop(columns=["UniProt"]).rename(
        columns={"genes_common": "Genes"}
    )

    ## Filter genes by number of complexes
    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    if max_categories is not None:
        prot_complex_df = prot_complex_df[
            prot_complex_df["Complex"].apply(len) <= max_categories
        ].reset_index(drop=True)

    ## Filter complexes by number of genes
    prot_complex_df = (
        prot_complex_df.explode("Complex").groupby("Complex").agg(list).reset_index()
    )
    prot_complex_df = prot_complex_df[
        prot_complex_df["Genes"].apply(len) >= min_genes
    ].reset_index(drop=True)

    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    prot_complex_df["n_complexes"] = prot_complex_df["Complex"].apply(len)

    common_genes_react = prot_complex_df["Genes"].tolist()
    return prot_complex_df, common_genes_react


def get_go_complexes(
    heirarchy_level,
    common_genes,
    complex_name,
    min_genes=5,
    only_common_complexes=False,
    max_categories=5,
):
    prot_complex_df = pd.read_csv(f"annotations/complexes/{complex_name}_hierarchy.csv")
    level_column = f"level_{heirarchy_level}"
    prot_complex_df = prot_complex_df.drop(
        columns=[
            x for x in prot_complex_df.columns if x not in ["UniProt", level_column]
        ]
    ).rename(columns={level_column: "Complex"})
    prot_complex_df = prot_complex_df[prot_complex_df["Complex"].notna()].reset_index(
        drop=True
    )
    prot_complex_df["Complex"] = prot_complex_df["Complex"].apply(
        lambda x: x.split(";")
    )
    prot_complex_df = prot_complex_df.explode("Complex")

    prot_complex_df = (
        prot_complex_df.groupby(f"Complex").agg(lambda x: list(set(x))).reset_index()
    )
    prot_complex_df["genes_common"] = prot_complex_df["UniProt"].apply(
        lambda x: [gene for gene in x if gene in common_genes]
    )

    if only_common_complexes:
        prot_complex_df = prot_complex_df[
            prot_complex_df["UniProt"] == prot_complex_df["genes_common"]
        ].reset_index(drop=True)

    prot_complex_df = prot_complex_df.drop(columns=["UniProt"]).rename(
        columns={"genes_common": "Genes"}
    )

    ## Filter genes by number of complexes
    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    if max_categories is not None:
        prot_complex_df = prot_complex_df[
            prot_complex_df["Complex"].apply(len) <= max_categories
        ].reset_index(drop=True)

    ## Filter complexes by number of genes
    prot_complex_df = (
        prot_complex_df.explode("Complex").groupby("Complex").agg(list).reset_index()
    )
    prot_complex_df = prot_complex_df[
        prot_complex_df["Genes"].apply(len) >= min_genes
    ].reset_index(drop=True)

    prot_complex_df = (
        prot_complex_df.explode("Genes").groupby("Genes").agg(list).reset_index()
    )
    prot_complex_df["n_complexes"] = prot_complex_df["Complex"].apply(len)

    common_genes_go = prot_complex_df["Genes"].tolist()
    return prot_complex_df, common_genes_go


def load_complex_data(
    complex_name,
    common_genes,
    heirarchy_level,
    min_genes=10,
    only_common_complexes=False,
    max_categories=5,
):
    if complex_name == "corum":
        prot_df, common_genes_complex = get_CORUM_complexes(
            common_genes,
            min_genes=min_genes,
            only_common_complexes=only_common_complexes,
            max_categories=max_categories,
        )
    elif complex_name == "humap":
        prot_df, common_genes_complex = get_HUMAP_complexes(
            common_genes,
            min_genes=min_genes,
            only_common_complexes=only_common_complexes,
            max_categories=max_categories,
        )
    elif complex_name == "reactome":
        prot_df, common_genes_complex = get_REACTOME_complexes(
            heirarchy_level,
            common_genes,
            min_genes=min_genes,
            only_common_complexes=only_common_complexes,
            max_categories=max_categories,
        )
    elif "go" in complex_name:
        prot_df, common_genes_go = get_go_complexes(
            heirarchy_level,
            common_genes,
            complex_name,
            min_genes=min_genes,
            only_common_complexes=only_common_complexes,
            max_categories=max_categories,
        )
    else:
        raise ValueError("Invalid complex name")

    prot_df["Complex"] = prot_df["Complex"].apply(lambda x: "; ".join(x))
    return prot_df


def load_protein_complex_data_cls(common_genes, complex_name, heirarchy_level):
    prot_complex_df = load_complex_data(
        complex_name,
        common_genes,
        heirarchy_level,
        min_genes=10,
        only_common_complexes=False,
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


def load_protein_complex_data_embed(common_genes, complex_name, heirarchy_level):
    if not complex_name in ["string", "bioplex", "bioplex_u2os"]:
        prot_complex_df = load_complex_data(
            complex_name,
            common_genes,
            heirarchy_level,
            min_genes=3,
            only_common_complexes=False,
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
        prot_complex_df = pd.read_csv(
            f"annotations/complexes/{complex_name}_only_ppi_uniprot.csv"
            if complex_name == "string"
            else f"annotations/complexes/{complex_name}_ppi_uniprot.csv"
        )
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
