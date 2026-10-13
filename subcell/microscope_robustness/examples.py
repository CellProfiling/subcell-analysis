"""The example cells of Extended Data Fig. 1a and the queries of its gallery (Extended Data Fig. 1f-m).

Panel a: one cell per line, drawn at random (seed 0) among the complete, high-confidence cells whose window lies
inside both images, in fields with typical staining (the field's median 63x cell intensities of DNA, MT, GMNN and
CDT1 all within the line's interquartile range of field medians).

Gallery: for each direction (40x query -> the 63x images of all complete cells; 63x query -> their 40x images), every
model ranks the gallery by cosine similarity (centred embeddings, float64). Outcome of a query (correct = the match
among a model's five most similar cells): every model correct / SubCell correct, not every model / another model
correct, not SubCell / no model correct. One query per outcome and direction, drawn at random (seed 20261001) among the
full-window cells, cell lines not yet shown in that direction first, and not a cell shown in the other direction. If
no cell is correct for every model (63x queries), the cells correct for the most models are used. Per query and
model: the five most similar cells, the match's rank, and the query's similarity profile (histogram of its cosine
similarity to every other gallery cell over [0.5th percentile, maximum], 48 bins; the fifth-highest similarity; the
match's similarity).

Misses (every model and Pixels, both directions, all complete cells): the fraction of queries with the match among the
five most similar cells, and the fraction of the five most similar cells of the query's line, separately for the
queries whose match was missed and those whose match was found (chance: the sum of the squared line proportions)."""

import json

import numpy as np
import pandas as pd
import torch

from subcell import config
from subcell.microscope_robustness import data

K = 5
SEED = 20261001
DIRS = {"ab": ("40x", "63x"), "ba": ("63x", "40x")}                  # (query, gallery)
CATS = {"all_correct": "top 5 in every model", "sub_right": "top 5 in SubCell, not in every model",
        "sub_wrong": "top 5 in another model, not in SubCell", "all_wrong": "top 5 in no model"}


def panel_a_cells(seed=0):
    c = data.cells()
    s = c[c.complete.values & c.hc.values].reset_index(drop=True)
    full = data.full_window(s)
    rng = np.random.default_rng(seed)
    rows = []
    for line in data.LINES:
        well = s.well[s.cell_line == line].iloc[0]
        x = pd.read_parquet(config.MICROSCOPE_MEASURE / well / "63x.parquet",
                            columns=["unit", "qc_ok", "cell_dapi_mean", "cell_mt_mean", "cell_gmnn_mean", "cell_cdt1_mean"])
        f = x[x.qc_ok].groupby("unit")[["cell_dapi_mean", "cell_mt_mean", "cell_gmnn_mean", "cell_cdt1_mean"]].median()
        q1, q3 = f.quantile(0.25), f.quantile(0.75)
        typical = set(f.index[((f >= q1) & (f <= q3)).all(1)])
        pool = s[(s.cell_line == line).values & full & s.field.isin(typical).values]
        r = pool.iloc[int(rng.integers(len(pool)))]
        rows.append(dict(cell_line=line, id=r.id, well=r.well, field=r.field, pool_n=len(pool), n_typical_fields=len(typical),
                         n_fields=len(f), seed=seed))
    return pd.DataFrame(rows)


def gallery():
    c = data.cells()
    sub = data.complete(c)
    N, ids, line = len(sub), sub.id.values, sub.cell_line.values
    full = data.full_window(sub)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Z, R = {}, {d: {} for d in DIRS}
    for r in data.MODELS:
        X = data.centred(data.features(r, c.id.values))
        Z[r] = {m: torch.as_tensor(data.unit(x)[c.complete.values], dtype=torch.float64) for m, x in X.items()}
        for d, (q, g) in DIRS.items():
            S = Z[r][q].to(dev) @ Z[r][g].to(dev).T
            dg = S.diagonal()[:, None]
            R[d][r] = (1 + (S > dg).sum(1) + 0.5 * ((S == dg).sum(1) - 1)).double().cpu().numpy()   # mid-rank of the match
            del S
    rng = np.random.default_rng(SEED)
    rows, picked = [], {}
    for d, (q, g) in DIRS.items():
        good = (pd.DataFrame(R[d])[data.MODELS] <= K).values
        n_good, sc = good.sum(1), good[:, 0]
        cat = np.select([n_good == len(data.MODELS), sc & (n_good < len(data.MODELS)), ~sc & (n_good > 0), n_good == 0], list(CATS))
        used = set()
        picked[d] = []
        for ct in CATS:
            other = {x for x, *_ in picked.get("ab" if d == "ba" else "ba", [])}
            pool = [i for i in np.flatnonzero((cat == ct) & full) if ids[i] not in other]
            label = CATS[ct]
            if not pool:
                pool = list(np.flatnonzero((cat == ct) & full))
            if not pool and ct == "all_correct":
                mx = int(n_good[full].max())
                pool = list(np.flatnonzero(full & (n_good == mx)))
                label = f"top 5 in {mx} models"
            fresh = [i for i in pool if line[i] not in used]
            i = int(rng.choice(fresh if fresh else pool))
            used.add(line[i])
            picked[d].append((ids[i], ct, label))
            for r in data.MODELS:
                s_ = (Z[r][g] @ Z[r][q][i]).numpy()
                top = np.argsort(-s_, kind="stable")[:K]
                neg = np.delete(s_, i)
                lo, hi = float(np.percentile(neg, 0.5)), float(s_.max())
                hist, _ = np.histogram(np.clip(neg, lo, hi), np.linspace(lo, hi, 49))
                rows.append(dict(direction=d, query=ids[i], category=ct, label=label, n_category=int((cat == ct).sum()), cell_line=line[i],
                                 run=r, rank_true=float(R[d][r][i]), **{f"cand{k + 1}": ids[j] for k, j in enumerate(top)},
                                 sim_true=float(s_[i]), sim_kth=float(np.sort(s_)[-K]), sim_lo=lo, sim_hi=hi, hist=json.dumps(hist.tolist())))
    return pd.DataFrame(rows)


def misses():
    """per representation and direction: top-5 retrieval of the match, and the share of the five most similar cells
    that are of the query's line among the queries whose match was missed (or found)"""
    c = data.cells()
    sub = data.complete(c)
    line = sub.cell_line.values
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for r in data.MODELS + [data.PIXELS]:
        X = data.centred(data.features(r, c.id.values))
        Z = {m: torch.as_tensor(data.unit(x)[c.complete.values], dtype=torch.float64, device=dev) for m, x in X.items()}
        for d, (q, g) in DIRS.items():
            S = Z[q] @ Z[g].T
            dg = S.diagonal()[:, None]
            rank = (1 + (S > dg).sum(1) + 0.5 * ((S == dg).sum(1) - 1)).cpu().numpy()   # mid-rank of the match
            same = line[torch.topk(S, K, dim=1).indices.cpu().numpy()] == line[:, None]
            miss = rank > K
            rows.append(dict(model=data.NAME[r], direction=d, query=q, n=len(line), top5=float((~miss).mean()), n_missed=int(miss.sum()),
                             same_line_top5_missed=float(same[miss].mean()), same_line_top5_found=float(same[~miss].mean()),
                             chance=float((pd.Series(line).value_counts(normalize=True) ** 2).sum())))
            del S
    return pd.DataFrame(rows)


def run():
    od = config.ensure(config.MICROSCOPE_DIR)
    a = panel_a_cells()
    a.to_csv(od / "panel_a_cells.csv", index=False)
    G = gallery()
    G.to_csv(od / "gallery.csv", index=False)
    M = misses()
    M.to_csv(od / "gallery_misses.csv", index=False)
    return a, G, M
