"""Hierarchical Leiden clusters of the image and sequence embeddings, shown on the co-embedding (Fig. 6).

The SubCell image embedding and the ESM-2 sequence embedding are each clustered with
:func:`subcell.utils.hierarchical_leiden.hierarchical_leiden`, at the levels of the
multiscale cell map (Fig. 5). Each space's labels are then carried, by UniProt
entry, to the UMAP of the MuSIC co-embedding the two were combined into, which
shows how much of each modality's structure the co-embedding keeps.

The three embeddings are the inputs and output of one MuSIC run, the row
``--method-name`` of ``methods_info.csv`` (``Image_Embed``, ``Seq_Embed``,
``Music_Embed``), so they cover the same proteins.

    python -m subcell.multimodal.cluster_transfer [--method-name NAME] [--run-dir DIR] [--out-dir DIR]

``--run-dir`` reads the three files from another copy of the run folder
(``<run>/img_data/``, ``<run>/seq_data/``, ``<run>/coembedding_k_10_d_128/``).

Output ``config.CLUSTER_TRANSFER_DIR``/``<method name>``/clusters.csv, one row per
UniProt entry: ``<space>_leiden_<res>`` for the image and sequence clusters and
``<space>_umap_1``, ``<space>_umap_2`` for every space. ``figures.fig6_cluster_transfer``
plots it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from subcell import config
from subcell.io.metadata import read_methods_csv
from subcell.runlog import log
from subcell.utils.hierarchical_leiden import RESOLUTIONS, hierarchical_leiden

METHOD_NAME = "MAE-CellS-ProtS-Pool_U2OS_harmonized_if_plate_id_microscope"

# space -> methods_info.csv column; the image and sequence spaces are clustered
SPACES = {"image": "Image_Embed", "sequence": "Seq_Embed", "multimodal": "Music_Embed"}
CLUSTERED = ("image", "sequence")


def input_paths(method_name=METHOD_NAME, run_dir=None) -> dict[str, Path]:
    """space -> embedding file of the MuSIC run *method_name*."""
    table = read_methods_csv().set_index("Method_Name")
    paths = {space: Path(table.loc[method_name, col]) for space, col in SPACES.items()}
    if run_dir is not None:
        # every file sits one folder below the run folder
        paths = {space: Path(run_dir) / p.parent.name / p.name for space, p in paths.items()}
    return paths


def read_embedding(path) -> pd.DataFrame:
    """A MuSIC embedding table, indexed by UniProt entry.

    Read without its header: ``coembedding_emd.tsv`` names one column fewer than
    it has, so the header would shift the index into the first dimension.
    """
    df = pd.read_csv(path, sep="\t", header=None, skiprows=1, index_col=0)
    df.index = df.index.astype(str)
    return df.astype("float32")


def umap(adata):
    sc.pp.neighbors(adata, n_neighbors=125, n_pcs=None, metric="euclidean")
    sc.tl.umap(adata, random_state=42, min_dist=0.5, spread=0.6, n_components=2)
    return adata.obsm["X_umap"]


def main(method_name=METHOD_NAME, run_dir=None, out_dir=None):
    paths = input_paths(method_name, run_dir)
    tables = {space: read_embedding(p) for space, p in paths.items()}
    proteins = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    for space, t in tables.items():
        log(f"{space}: {t.shape[0]:,} proteins x {t.shape[1]:,} dims  ({paths[space]})")
    log(f"{len(proteins):,} proteins in all three")

    out = pd.DataFrame(index=pd.Index(proteins, name="uniprot"))
    for space, t in tables.items():
        adata = sc.AnnData(t.loc[proteins].to_numpy())
        adata.obs_names = proteins
        if space in CLUSTERED:
            log(f"hierarchical Leiden on {space}")
            hierarchical_leiden(adata)
            for res in RESOLUTIONS:
                labels = adata.obs[f"leiden_{res}"]
                out[f"{space}_leiden_{res}"] = labels.astype(str)
                log(f"  res {res}: {labels.nunique()} clusters")
        log(f"UMAP of {space}")
        out[[f"{space}_umap_1", f"{space}_umap_2"]] = umap(adata)

    out_dir = config.ensure(out_dir or config.CLUSTER_TRANSFER_DIR / method_name)
    out.to_csv(out_dir / "clusters.csv")
    log(f"wrote {out_dir / 'clusters.csv'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME,
                    help=f"row of methods_info.csv (default: {METHOD_NAME})")
    ap.add_argument("--run-dir", default=None,
                    help="read the run's embeddings from this folder instead")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.run_dir, a.out_dir)
