"""Retrieval across the microscopes: normalised mAP@k for cell, cell-line and phase identity.

Each complete cell imaged on one microscope is a query against the images of the other microscope, in both
directions (40x -> 63x, 63x -> 40x); embeddings centred per microscope, cosine similarity.
  cell        candidates: all complete cells; correct: the match (the same cell)
  cell_line   candidates: all complete cells but the match; correct: cells of the same line
  phase       queries and candidates: the phased cells of the cycling lines; candidates: the other cells of the query's
              line; correct: cells in the same phase (G1 / S / G2, from the 63x images)
AP is copairs' (Kalinin et al. 2025): candidates ranked by similarity, ties in favour of correct matches. AP@k is AP
truncated at the k most similar candidates, AP@k = (1 / min(M, k)) sum_{i <= k} P_i r_i (M correct matches, P_i the
precision among the i most similar, r_i = 1 for a correct match at rank i); for one correct match, AP@1 = the match
ranked first. Each query's AP@k is normalised to (AP@k - E) / (1 - E), E its exact expectation under a random ranking
of the L candidates (expected_apk), and the two directions of a cell are averaged (M and L are the same in both).
Uncertainty: stratified component bootstrap (B 2,000); model pairs: component sign-flip test (1e5), Holm over the
15 pairs per task and k."""

import itertools

import numpy as np
import pandas as pd
import torch

from subcell import config
from subcell.microscope_robustness import data, phases, stats

TASKS = ["cell", "cell_line", "phase"]
TOPK = (1, 5, 10)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def expected_apk(M, L, k):
    """E[AP@k] under a random ranking of L candidates with M correct matches: E[P_i r_i] = (M / L)(1 + (i - 1)
    (M - 1) / (L - 1)) / i, summed over i <= k and divided by min(M, k) (Manzhos, Ianevych & Melnyk 2026); at k = L it equals the
    expected AP of copairs"""
    M, L = np.asarray(M, float), np.asarray(L, float)
    out = np.zeros_like(M)
    for i in range(1, k + 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            term = (M / L) * (1 + (i - 1) * np.where(L > 1, (M - 1) / (L - 1), 0)) / i
        out += np.where(i <= L, term, 0)
    return out / np.minimum(M, k)


def task_cells(task, sub, ph):
    """(valid-cell mask over sub, label codes): every complete cell, or the phased cells of the cycling lines"""
    if task in ("cell", "cell_line"):
        return np.ones(len(sub), bool), None
    lab = sub.id.map(ph)
    v = lab.isin(data.PHASES).values & sub.cell_line.isin(data.CYCLING).values
    return v, np.where(v, lab.map({p: i for i, p in enumerate(data.PHASES)}).fillna(-1).astype(int).values, -1)


def masks(task, line, label, rows):
    """correct-match and candidate masks of the query cells `rows` against all cells"""
    n = len(line)
    eye = torch.zeros((len(rows), n), dtype=torch.bool, device=DEV)
    eye[torch.arange(len(rows)), rows] = True
    if task == "cell":
        return eye, torch.ones_like(eye)
    same_line = line[rows][:, None] == line[None, :]
    if task == "cell_line":
        return same_line & ~eye, ~eye
    same = label[rows][:, None] == label[None, :]
    return same_line & same & ~eye, same_line & ~eye


def ap_at_k(Q, G, task, line, label, chunk=1024, ks=TOPK):
    """AP@k (k in ks) of each query row of Q against the gallery G (row i of Q and G = the same cell), with M
    (correct matches) and L (candidates) per query; k = len(G) gives the full AP"""
    n = len(Q)
    ar = torch.arange(1, n + 1, device=DEV, dtype=torch.float64)
    apk = {k: torch.empty(n, dtype=torch.float64, device=DEV) for k in ks}
    Mq, Lq = torch.empty(n, device=DEV), torch.empty(n, device=DEV)
    for s in range(0, n, chunk):
        rows = torch.arange(s, min(n, s + chunk), device=DEV)
        pos, cand = masks(task, line, label, rows)
        S = (Q[rows] @ G.T).masked_fill(~cand, -float("inf"))
        lab = pos.to(torch.int8)
        o1 = torch.sort(-lab, dim=1, stable=True).indices              # correct matches first ...
        S1, L1 = S.gather(1, o1), lab.gather(1, o1)
        o2 = torch.sort(-S1, dim=1, stable=True).indices               # ... then by similarity: ties keep them first
        rel = L1.gather(1, o2).double()
        cum = rel.cumsum(1)
        npos = pos.sum(1).double()
        for k in ks:                                                    # non-candidates are at -inf, after every candidate
            apk[k][rows] = ((cum[:, :k] / ar[:k]) * rel[:, :k]).sum(1) / torch.clamp(npos, max=k)
        Mq[rows], Lq[rows] = pos.sum(1).float(), cand.sum(1).float()
    return {k: v.cpu().numpy() for k, v in apk.items()}, Mq.cpu().numpy(), Lq.cpu().numpy()


def per_cell(c=None, keep=None):
    """per complete cell, run and task: AP@k of both directions averaged, M, L; keep (mask over the complete cells)
    restricts queries and candidates to a subset"""
    c = data.cells() if c is None else c
    sub = data.complete(c)
    keep = np.ones(len(sub), bool) if keep is None else np.asarray(keep)
    sub = sub[keep].reset_index(drop=True)
    ph = phases.load()
    line = torch.as_tensor(pd.factorize(sub.cell_line)[0], device=DEV)
    valid = {t: task_cells(t, sub, ph) for t in TASKS}
    rows = []
    for r in data.ORDER:
        X = data.centred(data.features(r, c.id.values))
        Z = {m: data.unit(x)[c.complete.values][keep] for m, x in X.items()}
        for t in TASKS:
            v, q = valid[t]
            idx = np.flatnonzero(v)
            A = torch.as_tensor(Z["40x"][idx], dtype=torch.float64, device=DEV)
            B = torch.as_tensor(Z["63x"][idx], dtype=torch.float64, device=DEV)
            ln = line[torch.as_tensor(idx, device=DEV)]
            qb = torch.as_tensor(q[idx] if q is not None else np.zeros(len(idx), int), device=DEV)
            k1, M1, L1 = ap_at_k(A, B, t, ln, qb)
            k2, M2, L2 = ap_at_k(B, A, t, ln, qb)
            assert np.array_equal(M1, M2) and np.array_equal(L1, L2)
            rows.append(pd.DataFrame({"id": sub.id.values[idx], "run": r, "task": t, "M": M1, "L": L1,
                                      **{f"ap{k}": (k1[k] + k2[k]) / 2 for k in TOPK}}))
        del X, Z
    return pd.concat(rows, ignore_index=True)


def summarise(C, c=None, boot=2000, flips=100000, keep=None):
    """-> (table: task, run, metric, value (normalised mean), ci_lo, ci_hi, raw, chance, n_cells;
           pairs: task, metric, a, b, delta, p, p_holm)"""
    c = data.cells() if c is None else c
    sub = data.complete(c)
    if keep is not None:
        sub = sub[np.asarray(keep)].reset_index(drop=True)
    ph = phases.load()
    rows, prs = [], []
    for t in TASKS:
        v, _ = task_cells(t, sub, ph)
        s_ = sub[v].reset_index(drop=True)
        Ct = C[C.task == t]
        for k in TOPK:
            S, RAW, EXP = {}, {}, None
            for r in data.ORDER:
                g = Ct[Ct.run == r].set_index("id").loc[s_.id]
                E = expected_apk(g.M.values, g.L.values, k)
                x = g[f"ap{k}"].values
                with np.errstate(invalid="ignore", divide="ignore"):
                    S[r] = np.where(1 - E > 1e-12, (x - E) / (1 - E), np.nan)
                RAW[r], EXP = x, E
            Mx = pd.DataFrame(S)
            okc = np.isfinite(Mx.values).all(1)
            assert okc.all(), (t, k)                                    # E < 1 for every query of these tests
            reps = stats.boot_means(Mx.values, s_.well.values, s_.comp.values, B=boot, seed=0)
            for j, r in enumerate(data.ORDER):
                lo, hi = stats.ci(reps[:, j])
                rows.append(dict(task=t, run=r, metric=f"map{k}", value=float(np.nanmean(Mx[r].values)), ci_lo=lo, ci_hi=hi,
                                 raw=float(np.mean(RAW[r])), chance=float(np.mean(EXP)), n_cells=int(okc.sum())))
            P = []
            for a, b in itertools.combinations(data.MODELS, 2):
                T, p = stats.signflip(Mx[a].values[okc] - Mx[b].values[okc], s_.comp.values[okc], B=flips)
                P.append(dict(task=t, metric=f"map{k}", a=a, b=b, delta=T, p=p))
            P = pd.DataFrame(P)
            P["p_holm"] = stats.holm(P.p.values)
            prs.append(P)
    return pd.DataFrame(rows), pd.concat(prs, ignore_index=True)


def run():
    c = data.cells()
    C = per_cell(c)
    od = config.ensure(config.MICROSCOPE_DIR)
    C.to_parquet(od / "identity_cells.parquet", index=False)
    T, P = summarise(C, c)
    T.to_csv(od / "identity.csv", index=False)
    P.to_csv(od / "identity_pairs.csv", index=False)
    return T, P

