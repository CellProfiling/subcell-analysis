"""Global geometry of the embeddings between the microscopes: linear CKA and the Mantel correlation.

Both compare the overall arrangement of the 21,002 matched cells in the 40x and 63x embeddings (centred per
microscope, rows L2-normalised) without matching individual cells: linear centred kernel alignment (Kornblith et al.
2019) on all cells, and the Mantel correlation (Spearman correlation of the condensed cosine-distance matrices of
the same 2,000 random cells, as in representational similarity analysis), averaged over 10 draws."""

import numpy as np
import pandas as pd
from scipy import stats as sps

from subcell import config
from subcell.microscope_robustness import data


def cka_linear(X, Y):
    """linear CKA of matched rows, feature-space form"""
    X = (X - X.mean(0)).astype(np.float32)
    Y = (Y - Y.mean(0)).astype(np.float32)
    f = lambda M: float(np.linalg.norm(M.astype(np.float64)))
    return f(X.T @ Y) ** 2 / (f(X.T @ X) * f(Y.T @ Y))


def mantel(X, Y, n=2000, reps=10, seed=0):
    """Spearman correlation of the cosine-distance matrices of the same n random cells, one value per draw"""
    rng = np.random.default_rng(seed)
    U, V = data.unit(X), data.unit(Y)
    out = []
    for _ in range(reps):
        s = rng.choice(len(X), min(n, len(X)), replace=False)
        iu = np.triu_indices(len(s), 1)
        out.append(sps.spearmanr(1 - (U[s] @ U[s].T)[iu], 1 - (V[s] @ V[s].T)[iu])[0])
    return np.array(out)


def run():
    c = data.cells()
    rows = []
    for r in data.ORDER:
        X = data.centred(data.features(r, c.id.values))
        A, B = data.unit(X["40x"]), data.unit(X["63x"])
        mt = mantel(A, B)
        rows.append(dict(run=r, cka_linear=cka_linear(A, B), mantel=float(mt.mean()), mantel_sd=float(mt.std(ddof=1)), n_cells=len(c)))
    T = pd.DataFrame(rows)
    T.to_csv(config.ensure(config.MICROSCOPE_DIR) / "geometry.csv", index=False)
    return T
