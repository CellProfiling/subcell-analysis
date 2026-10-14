"""Hierarchical Leiden clusters of the image and sequence embeddings, shown on the co-embedding (Fig. 6a,b).

The SubCell image embedding and the ESM-2 sequence embedding are each clustered with
:func:`subcell.utils.hierarchical_leiden.hierarchical_leiden`, at the levels of the
multiscale cell map (Fig. 5). Each space's labels are then carried, by UniProt
entry, to the UMAP of the MuSIC co-embedding the two were combined into, which
shows how much of each modality's structure the co-embedding keeps.

The three embeddings are those of one MuSIC run, the row ``--method-name`` of
``methods_info.csv`` (see ``subcell.multimodal.embeddings``).

    python -m subcell.multimodal.cluster_transfer [--method-name NAME] [--run-dir DIR] [--out-dir DIR]

Output ``config.CLUSTER_TRANSFER_DIR``/``<method name>``/clusters.csv, one row per
UniProt entry: ``<space>_leiden_<res>`` for the image and sequence clusters and
``<space>_umap_1``, ``<space>_umap_2`` for every space. ``figures.fig6ab_cluster_transfer``
plots it.
"""

from __future__ import annotations

import argparse

import pandas as pd
import scanpy as sc

from subcell import config
from subcell.multimodal.embeddings import METHOD_NAME, input_paths, read_embedding
from subcell.runlog import log
from subcell.utils.hierarchical_leiden import RESOLUTIONS, hierarchical_leiden

CLUSTERED = ("image", "sequence")


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
