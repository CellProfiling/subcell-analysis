"""Readout transfer across the microscopes.

Linear readouts trained on one microscope's embeddings (centred per microscope) and tested on held-out cells of the
same microscope (within: 40x -> 40x, 63x -> 63x) or of the other one (across: 40x -> 63x, 63x -> 40x). Folds:
GroupKFold(5) over the connected components of overlapping fields and tiles, so that no test cell shares an image
with a training cell. All preprocessing is fitted on the training microscope only and applied unchanged to the test
cells, so a readout transfers only if the same direction of the embedding encodes the property on both microscopes.
  phase      G1 / S / G2 (63x; the phased cells of the cycling lines)     classifier, balanced accuracy
  cell_line  the four lines (every complete cell)                         classifier, balanced accuracy
  gmnn_cdt1  log ratio of nuclear GMNN to CDT1 at 63x (complete cells whose 63x nucleus passes the object QC, as
             for the phases)                                               ridge, Spearman's rho
Classifier: standardisation -> 128 principal components -> L2-penalised multinomial logistic regression, C from
10^-3..10^2 (6 values) by inner GroupKFold(3) on balanced accuracy. Ridge: standardisation (target too), alpha from
10^-2..10^6 (17 values) by inner GroupKFold(3) on the mean squared error (the same grouped selection as the
classifier: cells of one field are correlated, so a leave-one-cell-out error would favour too little penalty).
Standardisation uses scale 1 for a column constant on the training cells (scikit-learn's convention). Both probes
run on the GPU in float64 and agree with scikit-learn (check_sklearn, one fold 40x -> 63x: 97% of phase predictions, ridge predictions within
5e-4).
Metrics of a kind = the mean of its two directions, recomputed on every bootstrap replicate (stratified component
bootstrap: B 2,000 for balanced accuracy, 10,000 for rho). SubCell vs each model: balanced accuracy by the component
sign-flip test of the per-cell contributions (1e5), rho by the paired bootstrap; Holm over the five comparisons.
Direction: (63x -> 40x) - (40x -> 63x) and (63x -> 63x) - (40x -> 40x) for every representation, paired over the
same cells and replicates, the same tests, Holm over the seven representations."""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

from subcell import config
from subcell.microscope_robustness import data, phases, stats

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CS = np.logspace(-3, 2, 6)
ALPHAS = np.logspace(-2, 6, 17)
DIRS = [("40x", "40x"), ("63x", "63x"), ("40x", "63x"), ("63x", "40x")]
KINDS = {"within": [("40x", "40x"), ("63x", "63x")], "across": [("40x", "63x"), ("63x", "40x")]}
DELTAS = {"across": (("63x", "40x"), ("40x", "63x")), "within": (("63x", "63x"), ("40x", "40x"))}
TARGETS = [("phase", "acc"), ("cell_line", "acc"), ("gmnn_cdt1", "rho")]
B_ACC, B_RHO = 2000, 10000


# ----------------------------------------------------------------------------------------------- the probes
def _scale(Xt):
    """the standard deviation of each column, 1 for a constant column (scikit-learn's StandardScaler): a feature
    constant on the training microscope must not be amplified where it varies on the other microscope"""
    sd = Xt.std(0, unbiased=False)
    return torch.where(sd > 0, sd, torch.ones_like(sd))


def fit_classifier(X, y, groups):
    """standardisation -> PCA 128 -> multinomial logistic regression, objective 1/2 |W|^2 + C sum(cross-entropy)
    (scikit-learn's; intercept unpenalised), L-BFGS; C by inner GroupKFold(3) on balanced accuracy"""
    cls = np.unique(y)
    yi = torch.as_tensor(np.searchsorted(cls, y), device=DEV)
    Xt = torch.as_tensor(X, dtype=torch.float64, device=DEV)
    mu, sd = Xt.mean(0), _scale(Xt)
    Z = (Xt - mu) / sd
    zm = Z.mean(0)
    _, _, V = torch.linalg.svd(Z - zm, full_matrices=False)
    P = V[:128].T
    F = (Z - zm) @ P

    def proj(Xn):
        return ((torch.as_tensor(Xn, dtype=torch.float64, device=DEV) - mu) / sd - zm) @ P

    def train(F_, yy, C):
        Wt = torch.zeros((F_.shape[1], len(cls)), dtype=torch.float64, device=DEV, requires_grad=True)
        bt = torch.zeros(len(cls), dtype=torch.float64, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([Wt, bt], lr=1, max_iter=500, tolerance_grad=1e-8, tolerance_change=1e-12, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = 0.5 * (Wt ** 2).sum() + C * torch.nn.functional.cross_entropy(F_ @ Wt + bt, yy, reduction="sum")
            loss.backward()
            return loss
        opt.step(closure)
        return Wt.detach(), bt.detach()

    best, bc = -1, None
    for C in CS:
        acc = []
        for tr, te in GroupKFold(3).split(np.zeros(len(y)), groups=groups):
            Wt, bt = train(F[tr], yi[tr], C)
            p = (F[te] @ Wt + bt).argmax(1).cpu().numpy()
            yt = yi[te].cpu().numpy()
            acc.append(np.mean([np.mean(p[yt == c] == c) for c in np.unique(yt)]))
        if np.mean(acc) > best:
            best, bc = np.mean(acc), C
    Wt, bt = train(F, yi, bc)
    return lambda Xn: cls[(proj(Xn) @ Wt + bt).argmax(1).cpu().numpy()]


def _ridge(Z, y):
    """ridge with intercept for every alpha of ALPHAS: (column means, target mean, {alpha: coefficients})"""
    zm, ym = Z.mean(0), y.mean()
    U, S, Vh = torch.linalg.svd(Z - zm, full_matrices=False)
    Uy = U.T @ (y - ym)
    return zm, ym, {a: Vh.T @ ((S / (S ** 2 + a)) * Uy) for a in ALPHAS}


def fit_regressor(X, y, groups):
    """standardisation (features and target) -> ridge with intercept, alpha by inner GroupKFold(3) on the mean squared
    error (the mean over the three folds, as scikit-learn's GridSearchCV)"""
    Xt = torch.as_tensor(X, dtype=torch.float64, device=DEV)
    mu, sd = Xt.mean(0), _scale(Xt)
    Z = (Xt - mu) / sd
    yt = torch.as_tensor(y, dtype=torch.float64, device=DEV)
    ym, ys = yt.mean(), yt.std(unbiased=False)
    yc = (yt - ym) / ys
    err = {a: [] for a in ALPHAS}
    for tr, te in GroupKFold(3).split(np.zeros(len(y)), groups=groups):
        zm, cm, coef = _ridge(Z[tr], yc[tr])
        for a, w in coef.items():
            err[a].append(torch.mean(((Z[te] - zm) @ w + cm - yc[te]) ** 2).item())
    ba = min(ALPHAS, key=lambda a: np.mean(err[a]))                  # ties: the smaller alpha, as GridSearchCV
    zm, cm, coef = _ridge(Z, yc)
    w = coef[ba]
    return lambda Xn: ((((torch.as_tensor(Xn, dtype=torch.float64, device=DEV) - mu) / sd - zm) @ w + cm) * ys + ym).cpu().numpy()


def targets(sub):
    """{target: (mask over sub, values)}"""
    ph = sub.id.map(phases.load()).values
    okp = pd.Series(ph).isin(data.PHASES).values & sub.cell_line.isin(data.CYCLING).values
    g = data.gmnn_cdt1(sub)
    qc = data.measurements(sub, "63x", ["qc_ok"]).qc_ok.values == 1                  # the 63x nucleus passes the object QC
    return {"phase": (okp, ph), "cell_line": (np.ones(len(sub), bool), sub.cell_line.values), "gmnn_cdt1": (np.isfinite(g) & qc, g)}


def predictions(run, c=None, which=None):
    """out-of-fold predictions of one representation: {target: {(train, test): predictions, "idx", "ids", "y"}}
    for the targets in which (default: all)"""
    c = data.cells() if c is None else c
    sub = data.complete(c)
    folds = list(GroupKFold(5).split(np.zeros(len(sub)), groups=sub.comp.values))
    X = data.features(run, c.id.values)
    E = {m: (x - x.mean(0))[c.complete.values].astype(np.float32) for m, x in X.items()}    # centred per microscope
    out = {}
    for t, (ok, y) in targets(sub).items():
        if which is not None and t not in which:
            continue
        idx = np.flatnonzero(ok)
        cl = t != "gmnn_cdt1"
        pred = {d: np.empty(len(sub), dtype=object) if cl else np.full(len(sub), np.nan) for d in DIRS}
        for tr, te in folds:
            tr, te = np.intersect1d(tr, idx), np.intersect1d(te, idx)
            for a in data.MAGS:
                f = (fit_classifier(E[a][tr], y[tr], sub.comp.values[tr]) if cl
                     else fit_regressor(E[a][tr], y[tr].astype(float), sub.comp.values[tr]))
                for b in data.MAGS:
                    pred[(a, b)][te] = f(E[b][te])
        out[t] = {d: v[idx] for d, v in pred.items()} | {"idx": idx, "ids": sub.id.values[idx], "y": y[idx]}
    return out


def check_sklearn(run=data.SUBCELL, c=None):
    """the GPU probes against scikit-learn on one fold (40x -> 63x): class agreement, max prediction difference"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    c = data.cells() if c is None else c
    sub = data.complete(c)
    tr, te = next(GroupKFold(5).split(np.zeros(len(sub)), groups=sub.comp.values))
    X = data.features(run, c.id.values)
    E = {m: (x - x.mean(0))[c.complete.values].astype(np.float32) for m, x in X.items()}
    T = targets(sub)
    res = {}
    ok, y = T["phase"]
    idx = np.flatnonzero(ok)
    t1, t2 = np.intersect1d(tr, idx), np.intersect1d(te, idx)
    sc = StandardScaler().fit(E["40x"][t1])
    pca = PCA(128, random_state=0, svd_solver="randomized").fit(sc.transform(E["40x"][t1]))
    Z = pca.transform(sc.transform(E["40x"][t1]))
    best, bc = -1, None
    for C in CS:
        acc = []
        for a, b in GroupKFold(3).split(Z, groups=sub.comp.values[t1]):
            p = LogisticRegression(C=C, max_iter=3000).fit(Z[a], y[t1][a]).predict(Z[b])
            acc.append(np.mean([np.mean(p[y[t1][b] == k] == k) for k in np.unique(y[t1][b])]))
        if np.mean(acc) > best:
            best, bc = np.mean(acc), C
    p_cpu = LogisticRegression(C=bc, max_iter=3000).fit(Z, y[t1]).predict(pca.transform(sc.transform(E["63x"][t2])))
    p_gpu = fit_classifier(E["40x"][t1], y[t1], sub.comp.values[t1])(E["63x"][t2])
    res["phase_class_agreement"] = float(np.mean(p_cpu == p_gpu))
    ok, y = T["gmnn_cdt1"]
    idx = np.flatnonzero(ok)
    t1, t2 = np.intersect1d(tr, idx), np.intersect1d(te, idx)
    sc = StandardScaler().fit(E["40x"][t1])
    ys = (y[t1].mean(), y[t1].std())
    gs = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=GroupKFold(3), scoring="neg_mean_squared_error")
    gs.fit(sc.transform(E["40x"][t1]), (y[t1] - ys[0]) / ys[1], groups=sub.comp.values[t1])
    p_cpu = gs.predict(sc.transform(E["63x"][t2])) * ys[1] + ys[0]
    p_gpu = fit_regressor(E["40x"][t1], y[t1], sub.comp.values[t1])(E["63x"][t2])
    res["gmnn_cdt1_max_abs_diff"] = float(np.max(np.abs(p_cpu - p_gpu)))
    return res


# ---------------------------------------------------------------------------------------------- the metrics
def balanced_accuracy(y, p):
    return float(np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]))


def boot_balanced_accuracy(y, p, W):
    """balanced accuracy on every replicate (rows of W = cell counts)"""
    out = np.zeros(len(W))
    cls = np.unique(y)
    for k in cls:
        m = y == k
        out += (W[:, m] @ (p[m] == k).astype(float)) / W[:, m].sum(1)
    return out / len(cls)


def rho(y, p):
    """Spearman's rho (Pearson correlation of the average ranks)"""
    ry, rp = rankdata(y), rankdata(p)
    ry -= ry.mean()
    rp -= rp.mean()
    return float((ry * rp).sum() / np.sqrt((ry ** 2).sum() * (rp ** 2).sum()))


_G = {}                                                                  # shared with the forked workers


def _rho_reps(key):
    y, Ws, p = _G["y"], _G["Ws"], _G["preds"][key]
    out = np.empty(len(Ws))
    for b, w in enumerate(Ws):
        ii = np.repeat(np.arange(len(y)), w)
        out[b] = rho(y[ii], p[ii])
    return key, out


def score(parts, sub, workers=None):
    """per target, representation and direction: the full-sample metric, its bootstrap replicates and (balanced
    accuracy) the per-cell contributions whose mean it is"""
    from multiprocessing import Pool
    T = targets(sub)
    res = {}
    for t, kind in TARGETS:
        idx = parts[data.ORDER[0]][t]["idx"]
        y = T[t][1][idx]
        for r in data.ORDER:                                            # the cached predictions belong to these cells and labels
            assert np.array_equal(parts[r][t]["idx"], idx) and np.array_equal(parts[r][t]["ids"], sub.id.values[idx]), (r, t)
            assert np.array_equal(parts[r][t]["y"], y) if kind == "acc" else np.allclose(parts[r][t]["y"], y, rtol=0, atol=0, equal_nan=True), (r, t)
        s_ = sub.iloc[idx]
        P = {(r, d): parts[r][t][d] for r in data.ORDER for d in DIRS}
        if kind == "acc":
            K = len(np.unique(y))
            nc = pd.Series(y).map(pd.Series(y).value_counts()).values
            W = stats.weights(s_.well.values, s_.comp.values, B_ACC, seed=0)
            full = {k: balanced_accuracy(y, p) for k, p in P.items()}
            reps = {k: boot_balanced_accuracy(y, p, W) for k, p in P.items()}
            cell = {k: (p == y) / (K * nc) * len(y) for k, p in P.items()}
            for k in full:
                assert abs(cell[k].mean() - full[k]) < 1e-9
            res[t] = dict(kind=kind, full=full, reps=reps, cell=cell, chance=1 / K, n=len(idx), comp=s_.comp.values)
        else:
            y = y.astype(float)
            Ws = stats.weights(s_.well.values, s_.comp.values, B_RHO, seed=0).astype(int)
            _G.update(y=y, Ws=Ws, preds={k: p.astype(float) for k, p in P.items()})
            full = {k: rho(y, p) for k, p in _G["preds"].items()}
            with Pool(workers or int(os.environ.get("N_WORKERS", 8))) as pool:
                reps = dict(pool.map(_rho_reps, list(P)))
            res[t] = dict(kind=kind, full=full, reps=reps, cell=None, chance=np.nan, n=len(idx), comp=s_.comp.values)
    return res


def tables(res, flips=100000):
    """-> transfer (kind means), transfer_pairs (SubCell vs each model), direction (per direction),
    direction_delta (paired differences between directions)"""
    rows, prs, drows, dd = [], [], [], []
    for t, kind in TARGETS:
        R = res[t]
        metric = "balanced_accuracy" if kind == "acc" else "spearman"
        mean = lambda d, r, k: np.mean([d[(r, x)] for x in KINDS[k]], axis=0)
        for k in KINDS:
            for r in data.ORDER:
                lo, hi = stats.ci(mean(R["reps"], r, k))
                rows.append(dict(target=t, metric=metric, kind=k, run=r, value=float(np.mean([R["full"][(r, x)] for x in KINDS[k]])),
                                 ci_lo=lo, ci_hi=hi, chance=R["chance"], n_cells=R["n"]))
            P = []
            for o in data.MODELS[1:]:
                delta = float(np.mean([R["full"][(data.SUBCELL, x)] for x in KINDS[k]]) - np.mean([R["full"][(o, x)] for x in KINDS[k]]))
                if kind == "acc":
                    _, p = stats.signflip(mean(R["cell"], data.SUBCELL, k) - mean(R["cell"], o, k), R["comp"], B=flips)
                else:
                    p = stats.paired_boot_p(mean(R["reps"], data.SUBCELL, k) - mean(R["reps"], o, k))
                P.append(dict(target=t, kind=k, other=o, delta=delta, p=p))
            P = pd.DataFrame(P)
            P["p_holm"] = stats.holm(P.p.values)
            prs.append(P)
        for r in data.ORDER:
            for d in DIRS:
                lo, hi = stats.ci(R["reps"][(r, d)])
                drows.append(dict(target=t, metric=metric, run=r, direction=f"{d[0]}->{d[1]}", value=R["full"][(r, d)], ci_lo=lo, ci_hi=hi,
                                  n_cells=R["n"]))
        for comp, (a, b) in DELTAS.items():
            D = []
            for r in data.ORDER:
                dr = R["reps"][(r, a)] - R["reps"][(r, b)]
                if kind == "acc":
                    _, p = stats.signflip(R["cell"][(r, a)] - R["cell"][(r, b)], R["comp"], B=flips)
                else:
                    p = stats.paired_boot_p(dr)
                lo, hi = stats.ci(dr)
                D.append(dict(target=t, comparison=comp, run=r, delta=R["full"][(r, a)] - R["full"][(r, b)], ci_lo=lo, ci_hi=hi, p=p))
            D = pd.DataFrame(D)
            D["p_holm"] = stats.holm(D.p.values)
            dd.append(D)
    return pd.DataFrame(rows), pd.concat(prs, ignore_index=True), pd.DataFrame(drows), pd.concat(dd, ignore_index=True)


def cache(run, target):
    return config.MICROSCOPE_DIR / "transfer_oof" / run / f"{target}.pkl"


def run(runs=None, which=None, combine=True):
    """out-of-fold predictions of each representation and target (cached: transfer_oof/<run>/<target>.pkl, written once
    complete, checked against the current cells and labels when scored), then the tables"""
    c = data.cells()
    sub = data.complete(c)
    for r in runs or data.ORDER:
        todo = [t for t, _ in TARGETS if (which is None or t in which) and not cache(r, t).exists()]
        if todo:
            for t, part in predictions(r, c, todo).items():
                f = cache(r, t)
                config.ensure(f.parent)
                with open(f.with_suffix(".tmp"), "wb") as fh:
                    pickle.dump(part, fh)
                f.with_suffix(".tmp").rename(f)
    if not combine:
        return None
    parts = {r: {t: pickle.load(open(cache(r, t), "rb")) for t, _ in TARGETS} for r in data.ORDER}
    out = tables(score(parts, sub))
    for name, df in zip(("transfer", "transfer_pairs", "transfer_direction", "transfer_direction_delta"), out):
        df.to_csv(config.MICROSCOPE_DIR / f"{name}.csv", index=False)
    return out
