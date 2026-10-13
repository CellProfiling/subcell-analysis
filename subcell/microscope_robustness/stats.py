"""Uncertainty and tests of the cross-microscope benchmark.

Cells imaged in the same field are not independent, and 63x fields and 40x tiles overlap, so the statistical unit is
the connected component of overlapping fields and tiles (data.cells, column comp; 157 components). Each cell line
was grown in its own well, which is a fixed stratum."""

import numpy as np
import pandas as pd


def boot_weights(well, cluster, B, seed=0, chunk=250):
    """two-level bootstrap (Saravanan et al. 2020) stratified by well: per replicate, within every well its clusters
    are drawn with replacement, then the cells of every drawn cluster with replacement. Yields (b, N) float32 count
    matrices, b <= chunk, B in total."""
    rng = np.random.default_rng(seed)
    well, cluster = np.asarray(well), np.asarray(cluster)
    N = len(well)
    strata = []
    for w in pd.unique(well):
        cid = pd.factorize(cluster[well == w])[0]
        members = np.flatnonzero(well == w)[np.argsort(cid, kind="stable")]
        size = np.bincount(cid)
        strata.append((members, np.r_[0, np.cumsum(size)[:-1]], size))
    for b0 in range(0, B, chunk):
        W = np.zeros((min(chunk, B - b0), N), np.float32)
        for r in range(len(W)):
            for members, start, size in strata:
                j = rng.integers(0, len(size), len(size))
                n = size[j]
                pick = np.repeat(start[j], n) + (rng.random(n.sum()) * np.repeat(n, n)).astype(int)
                W[r] += np.bincount(members[pick], minlength=N)
        yield W


def weights(well, cluster, B, seed=0):
    """all B replicates as one (B, N) matrix"""
    return np.concatenate(list(boot_weights(well, cluster, B, seed)))


def boot_means(scores, well, cluster, B=2000, seed=0):
    """(B, M) bootstrap replicates of the cell means of the columns of scores (N, M); rows with a NaN are dropped"""
    S = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(S).all(1)
    S, well, cluster = S[ok], np.asarray(well)[ok], np.asarray(cluster)[ok]
    return np.concatenate([W @ S / W.sum(1, keepdims=True) for W in boot_weights(well, cluster, B, seed)])


def ci(reps):
    """95% percentile interval"""
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def cluster_totals(x, cluster):
    s = pd.DataFrame({"x": x, "c": cluster}).dropna()
    g = s.groupby("c", sort=True).x
    return g.sum().values, g.size().values


def signflip(diff, cluster, B=100000, seed=0):
    """paired sign-flip permutation test of the mean per-cell difference with clusters as the exchangeable units
    (cluster totals get random signs); two-sided P = (1 + #{|T*| >= |T|}) / (1 + B) (Phipson & Smyth 2010).
    Returns (T, P), T the mean difference."""
    tot, n = cluster_totals(diff, cluster)
    T = tot.sum() / n.sum()
    rng = np.random.default_rng(seed)
    cnt = 0
    for b0 in range(0, B, 1000):
        s = rng.choice([-1.0, 1.0], size=(min(1000, B - b0), len(tot)))
        cnt += int((np.abs(s @ tot / n.sum()) >= abs(T) - 1e-15).sum())
    return float(T), (1 + cnt) / (1 + B)


def paired_boot_p(d):
    """two-sided P of a paired bootstrap of a difference d (one value per replicate):
    2 (1 + #{replicates on the other side of 0}) / (1 + B), at most 1"""
    d = np.asarray(d)
    return min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (1 + len(d)))


def holm(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out
