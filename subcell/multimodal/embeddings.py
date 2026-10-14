"""The image, sequence and co-embedding of one MuSIC run, as used by the Fig. 6 analyses.

The three embeddings are the inputs and output of one MuSIC run, a row of
``methods_info.csv`` (``Image_Embed``, ``Seq_Embed``, ``Music_Embed``), so they
cover the same proteins, keyed by UniProt entry. ``run_dir`` reads them from
another copy of the run folder instead (``<run>/img_data/``, ``<run>/seq_data/``,
``<run>/coembedding_k_10_d_128/``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from subcell.io.metadata import read_methods_csv

METHOD_NAME = "MAE-CellS-ProtS-Pool_U2OS_harmonized_if_plate_id_microscope"

# space -> methods_info.csv column
SPACES = {"image": "Image_Embed", "sequence": "Seq_Embed", "multimodal": "Music_Embed"}


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


def centered_cosine(X) -> np.ndarray:
    """Cosine similarity between every pair of rows, after subtracting the mean row."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T
