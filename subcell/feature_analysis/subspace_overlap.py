"""Overlap between the cell-line and localization subspaces of an embedding.

The subspace-overlap analysis of the attention profiles reported in the Methods
("Evaluating attention maps"); called from
``subcell.feature_analysis.attention_correlation``.
Measures whether the directions in a learned embedding Z that discriminate cell
lines (y_cell) overlap with the directions that discriminate protein
localizations (y_loc).

Two complementary statistics are reported:

  Subspace overlap coefficient  in [0, 1]
    Geometric overlap of the two discriminative subspaces:
      overlap = sum_k cos^2(theta_k) / min(rank_cell, rank_loc)
    0 = perfectly orthogonal (disentangled)
    1 = identical subspace   (entangled)

  CCA r1  in [0, 1]
    First canonical correlation between the two low-dimensional projections.
    Measures the maximum linear correlation achievable between any linear
    combination of the cell-line projection and any linear combination of
    the localization projection.
    0 = no correlation (disentangled)
    1 = perfectly correlated (entangled)

Both are tested against a random-subspace null (Haar-uniform subspaces of the
localization rank, i.e. random points on the Grassmannian, for which
E[overlap] = rank_loc / D) and accompanied by 95% bias-corrected bootstrap CIs
(see ``_bias_corrected_ci`` for how the acceleration term is estimated).

Usage::

    from subcell.feature_analysis.subspace_overlap import run_subspace_entanglement

    results = run_subspace_entanglement(
        Z          = embedding,          # (n, D)
        y_cell     = cell_line_labels,   # (n,)
        y_loc      = localization_matrix,# (n, n_loc) binary
        output_dir = "./results",
        loc_names  = ["nucleus", "cytoplasm", ...],
        max_rank   = 8,
    )
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from numpy.random import default_rng
from scipy import linalg
from scipy.stats import norm as _norm
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from subcell.runlog import log
from subcell.utils.metrics import stars


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Subspace construction
#     Both subspaces use regularised LDA with Ledoit-Wolf shrinkage.
# ─────────────────────────────────────────────────────────────────────────────


def _cell_line_subspace(Z, y_cell, max_rank):
    """
    Top max_rank discriminant directions for cell-line labels.

    Uses multiclass regularised LDA (solver='eigen', shrinkage='auto').
    Ledoit-Wolf shrinkage regularises the within-class scatter matrix,
    which is near-singular when the number of classes is large relative
    to the embedding dimension D (Ledoit & Wolf, JMVA 2004).
    QR orthonormalises the resulting scalings.

    Returns W_cell : (D, rank_cell)  orthonormal basis
    """
    n_comp = min(len(np.unique(y_cell)) - 1, Z.shape[1], max_rank)
    lda = LinearDiscriminantAnalysis(
        n_components=n_comp,
        solver="eigen",  # populates scalings_; lsqr does not
        shrinkage="auto",  # Ledoit-Wolf analytical estimator
    )
    lda.fit(Z, y_cell)
    W, _ = np.linalg.qr(lda.scalings_[:, :n_comp])
    return W[:, :n_comp]


def _one_loc_direction(Z, y_k):
    """
    One binary Fisher discriminant for a single localization label.

    Computes  w_k ∝ Σ_pooled,k⁻¹ (μ_{k,1} − μ_{k,0})  via regularised LDA
    (solver='eigen', shrinkage='auto'). This is the Mahalanobis-normalised
    Fisher direction, NOT a plain mean difference — it accounts for within-
    class variance via the Ledoit-Wolf pooled covariance estimator.

    Returns a unit-norm vector of shape (D,).
    """
    y_k = np.asarray(y_k, dtype=int)
    vals, counts = np.unique(y_k, return_counts=True)
    if len(vals) < 2 or counts.min() < 2:
        return np.zeros(Z.shape[1], dtype=np.float32)

    lda = LinearDiscriminantAnalysis(
        n_components=1,
        solver="eigen",  # solver='lsqr' does NOT set scalings_
        shrinkage="auto",  # Ledoit-Wolf
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lda.fit(Z, y_k)
    w = lda.scalings_[:, 0].astype(np.float32)
    return w / (np.linalg.norm(w) + 1e-12)


def _localization_subspace(Z, y_loc, max_rank, n_jobs=-1):
    """
    Top max_rank directions of the localization discriminative subspace.

    Fits one binary Fisher discriminant per localization label (in parallel),
    stacks the n_loc unit-norm directions into D = [w_1 | … | w_{n_loc}],
    then extracts the leading max_rank orthonormal components via SVD.
    SVD pools information across all labels: the top singular vectors span
    the directions most discriminative across all localizations jointly,
    analogous to PCA applied to a set of class-separation vectors.

    Returns
        W_loc   : (D, rank_loc)  orthonormal basis
        sv      : (rank_loc,)    singular values (scree diagnostic)
    """
    directions = Parallel(n_jobs=n_jobs)(
        delayed(_one_loc_direction)(Z, y_loc[:, k]) for k in range(y_loc.shape[1])
    )
    D_mat = np.column_stack(directions).astype(np.float32)  # (D, n_loc)

    U, s, _ = np.linalg.svd(D_mat, full_matrices=False)
    rank = max(1, min(max_rank, int((s > s[0] * 1e-6).sum())))
    return U[:, :rank].astype(np.float32), s[:rank]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Statistics
# ─────────────────────────────────────────────────────────────────────────────


def _overlap_coefficient(W1, W2):
    """
    Subspace overlap  =  Σ_k cos²(θ_k) / min(rank1, rank2)  ∈ [0, 1].
    Returns (overlap, principal_angles_in_degrees).
    """
    angles = linalg.subspace_angles(W1, W2)
    k = min(W1.shape[1], W2.shape[1])
    overlap = float(np.sum(np.cos(angles[:k]) ** 2) / k)
    return overlap, np.degrees(angles)


def _cca_first_r(Z, W1, W2):
    """
    First canonical correlation r₁ between Z @ W1 and Z @ W2.
    Returns |r₁| ∈ [0, 1].
    """
    n_comp = min(W1.shape[1], W2.shape[1], 5)
    if n_comp < 1:
        return float("nan")
    cca = CCA(n_components=n_comp, max_iter=1000, tol=1e-6)
    cca.fit(Z @ W1, Z @ W2)
    Uc, Ul = cca.transform(Z @ W1, Z @ W2)
    return float(abs(np.corrcoef(Uc[:, 0], Ul[:, 0])[0, 1]))


def _bias_corrected_ci(obs, boot_samples, alpha=0.05):
    """
    Bias-corrected, accelerated percentile bootstrap 95% CI, after Efron (1987).

    The bias correction z0 is the standard one (share of draws below the observed
    value). The acceleration is NOT the standard BCa jackknife of the data: it is
    the skewness of leave-one-out means of the bootstrap draws themselves, so this
    approximates BCa and is not identical to it.
    Returns (lower, upper).
    """
    boot = boot_samples[~np.isnan(boot_samples)]
    if len(boot) < 10:
        return (float("nan"), float("nan"))
    z0 = _norm.ppf(np.clip((boot < obs).mean(), 1e-10, 1 - 1e-10))
    jack_means = np.array([np.nanmean(np.delete(boot, i)) for i in range(len(boot))])
    diff = jack_means.mean() - jack_means
    a = (diff**3).sum() / (6 * (diff**2).sum() ** 1.5 + 1e-12)
    lo = float(
        np.nanpercentile(
            boot,
            100
            * _norm.cdf(
                z0 + (z0 + _norm.ppf(alpha / 2)) / (1 - a * (z0 + _norm.ppf(alpha / 2)))
            ),
        )
    )
    hi = float(
        np.nanpercentile(
            boot,
            100
            * _norm.cdf(
                z0
                + (z0 + _norm.ppf(1 - alpha / 2))
                / (1 - a * (z0 + _norm.ppf(1 - alpha / 2)))
            ),
        )
    )
    return (lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main analysis
# ─────────────────────────────────────────────────────────────────────────────


def run_subspace_entanglement(
    Z,
    y_cell,
    y_loc,
    output_dir,
    loc_names=None,
    max_rank=8,
    n_perm=1000,
    n_bootstrap=500,
    seed=0,
    n_jobs=-1,
):
    """
    Measure whether cell-line and localization are encoded in overlapping
    directions within the learned embedding Z.

    Parameters
    ----------
    Z           : (n, D) float array — embedding to evaluate
    y_cell      : (n,)   int or str  — cell-line labels
    y_loc       : (n, n_loc) binary  — multi-label localization matrix
    output_dir  : str                — directory for CSV + figure
    loc_names   : list[str]          — localization label names (optional)
    max_rank    : int                — rank cap for both subspaces.
                  Keep rank_cell + rank_loc <= D/2: beyond that, overlap
                  is forced geometrically rather than by the data.
    n_perm      : int                — random-subspace null draws (default 1000)
    n_bootstrap : int                — bootstrap replicates for CI (default 500)
    seed        : int                — random seed
    n_jobs      : int                — CPU cores (-1 = all)

    Returns
    -------
    dict
        overlap       float   subspace overlap coefficient ∈ [0, 1]
        ci_overlap    tuple   95% bias-corrected bootstrap CI for overlap
        pval_overlap  float   random-subspace-null p-value for overlap
        cca_r1        float   first canonical correlation ∈ [0, 1]
        ci_cca        tuple   95% bias-corrected bootstrap CI for CCA r₁
        pval_cca      float   random-subspace-null p-value for CCA r₁
        angles_deg    array   all principal angles in degrees
        rank_cell     int     actual rank of W_cell
        rank_loc      int     actual rank of W_loc
        sv_loc        array   singular values of localization direction matrix
        summary       DataFrame  one row per statistic, ready for export
    """
    t0 = time.time()
    rng = default_rng(seed)

    Z = np.asarray(Z, dtype=np.float32)
    y_cell = np.asarray(y_cell)
    y_loc = np.asarray(y_loc, dtype=np.int8)
    n, D = Z.shape
    n_loc = y_loc.shape[1]
    loc_names = (
        list(loc_names) if loc_names is not None else [f"loc_{k}" for k in range(n_loc)]
    )

    print("─" * 60)
    print("Subspace overlap analysis")
    print(
        f"  n={n:,}  D={D}  cell labels={len(np.unique(y_cell))}"
        f"  loc labels={n_loc}  max_rank={max_rank}"
    )
    print("─" * 60)

    # ── Step 1: build observed subspaces ──────────────────────────────────────
    print("\n[1/4] Building discriminative subspaces")
    print("  Cell line  → regularised multiclass LDA ...")
    W_cell = _cell_line_subspace(Z, y_cell, max_rank)

    print(f"  Localization → {n_loc} binary LDA fits in parallel ...")
    W_loc, sv_loc = _localization_subspace(Z, y_loc, max_rank, n_jobs)

    rank_c, rank_l = W_cell.shape[1], W_loc.shape[1]
    print(f"  W_cell rank = {rank_c}   W_loc rank = {rank_l}   D = {D}")
    if rank_c + rank_l > D // 2:
        print(
            f"  ⚠  rank_cell + rank_loc = {rank_c + rank_l} > D/2 = {D//2}."
            f" Consider lowering max_rank."
        )

    # ── Step 2: observed statistics ───────────────────────────────────────────
    print("\n[2/4] Computing observed statistics")
    obs_overlap, angles_deg = _overlap_coefficient(W_cell, W_loc)
    obs_cca_r1 = _cca_first_r(Z, W_cell, W_loc)
    print(f"  Subspace overlap  = {obs_overlap:.4f}")
    print(f"  CCA r₁            = {obs_cca_r1:.4f}")
    print(f"  Principal angles  = {angles_deg.round(1)}")

    # ── Step 3: random-subspace null ──────────────────────────────────────────
    # Null: a Haar-uniform random k-dimensional subspace of R^D (a random point
    # on the Grassmannian Gr(k, D), from the QR decomposition of a Gaussian
    # matrix), with k = rank_loc. The p-value answers whether W_loc is more
    # aligned with W_cell than a subspace of the same rank placed at random.
    # Shuffling y_loc and refitting LDA is not used: with a strong cell-line
    # signal in Z, LDA on shuffled labels picks up cell-line directions.
    #
    # Analytical baseline: for two Haar-uniform k-dimensional subspaces,
    # E[overlap] = k / D (Bjorck & Golub, 1973).

    analytical_null = rank_l / D  # E[overlap] under Haar null = k/D
    normalized_excess = (obs_overlap - analytical_null) / (1 - analytical_null + 1e-12)
    print(f"\n[3/4] Random-subspace null  ({n_perm} draws, Grassmannian sampling)")
    print(
        f"  Analytical null (Haar): E[overlap] = rank_loc/D = "
        f"{rank_l}/{D} = {analytical_null:.4f}"
    )
    print(
        f"  Normalized excess overlap = {normalized_excess:.4f}  "
        f"(0=random, 1=max entanglement)"
    )

    null_overlap = np.empty(n_perm, dtype=np.float32)
    null_cca_r1 = np.empty(n_perm, dtype=np.float32)

    for b in tqdm(range(n_perm), desc="  null draws", unit="draw", ncols=72):
        # QR of a random Gaussian matrix gives a Haar-uniform orthonormal basis
        Q_rand = np.linalg.qr(rng.standard_normal((D, rank_l)).astype(np.float32))[0]
        null_overlap[b], _ = _overlap_coefficient(W_cell, Q_rand)
        null_cca_r1[b] = _cca_first_r(Z, W_cell, Q_rand)

    pval_overlap = float((1 + (null_overlap >= obs_overlap).sum()) / (n_perm + 1))
    pval_cca = float((1 + (null_cca_r1 >= obs_cca_r1).sum()) / (n_perm + 1))
    print(f"  p(overlap) = {pval_overlap:.4f}   p(CCA r₁) = {pval_cca:.4f}")

    # ── Step 4: bootstrap 95% bias-corrected CIs ──────────────────────────────
    # Resample cells with replacement → refit both subspaces → recompute statistics.
    print(f"\n[4/4] Bootstrap 95% bias-corrected CI  ({n_bootstrap} replicates)")
    boot_overlap = np.full(n_bootstrap, np.nan, dtype=np.float32)
    boot_cca = np.full(n_bootstrap, np.nan, dtype=np.float32)
    n_failed = 0

    for b in tqdm(range(n_bootstrap), desc="  bootstrap", unit="rep", ncols=72):
        idx = rng.choice(n, size=n, replace=True)
        try:
            W_c_b = _cell_line_subspace(Z[idx], y_cell[idx], max_rank)
            W_l_b, _ = _localization_subspace(Z[idx], y_loc[idx], max_rank, n_jobs)
            boot_overlap[b], _ = _overlap_coefficient(W_c_b, W_l_b)
            boot_cca[b] = _cca_first_r(Z[idx], W_c_b, W_l_b)
        except Exception as err:
            # left as nan; _bias_corrected_ci filters them out
            n_failed += 1
            if n_failed == 1:
                log(f"  bootstrap replicate {b} failed: {err!r}")
    if n_failed:
        log(f"  WARNING: {n_failed}/{n_bootstrap} bootstrap replicates failed and "
            f"are excluded from the CIs")

    ci_overlap = _bias_corrected_ci(obs_overlap, boot_overlap)
    ci_cca = _bias_corrected_ci(obs_cca_r1, boot_cca)
    print(
        f"  overlap  {obs_overlap:.3f}  95% CI [{ci_overlap[0]:.3f}, {ci_overlap[1]:.3f}]"
    )
    print(f"  CCA r₁   {obs_cca_r1:.3f}  95% CI [{ci_cca[0]:.3f}, {ci_cca[1]:.3f}]")

    elapsed = time.time() - t0

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame(
        [
            {
                "statistic": "Subspace overlap coefficient",
                "value": round(obs_overlap, 4),
                "ci_95_low": round(ci_overlap[0], 4),
                "ci_95_high": round(ci_overlap[1], 4),
                "p_value": round(pval_overlap, 4),
                "null_mean": round(float(null_overlap.mean()), 4),
                "interpretation": "0 = disentangled   1 = fully entangled",
            },
            {
                "statistic": "Analytical null (Haar, k/D)",
                "value": round(analytical_null, 4),
                "ci_95_low": "",
                "ci_95_high": "",
                "p_value": "",
                "null_mean": "",
                "interpretation": f"Expected overlap for 2 random rank-{rank_l} subspaces in R^{D}",
            },
            {
                "statistic": "Normalized excess overlap",
                "value": round(normalized_excess, 4),
                "ci_95_low": "",
                "ci_95_high": "",
                "p_value": "",
                "null_mean": "",
                "interpretation": "0 = random baseline   1 = max entanglement",
            },
            {
                "statistic": "CCA first canonical correlation (r1)",
                "value": round(obs_cca_r1, 4),
                "ci_95_low": round(ci_cca[0], 4),
                "ci_95_high": round(ci_cca[1], 4),
                "p_value": round(pval_cca, 4),
                "null_mean": round(float(null_cca_r1.mean()), 4),
                "interpretation": "0 = disentangled   1 = fully entangled",
            },
            {
                "statistic": "Mean principal angle (deg)",
                "value": f"{round(float(angles_deg.mean()), 2)} ± {round(float(angles_deg.std()), 2)}",
                "ci_95_low": "",
                "ci_95_high": "",
                "p_value": "",
                "null_mean": "",
                "interpretation": "90 = orthogonal   0 = same direction",
            },
            {
                "statistic": "W_cell rank",
                "value": rank_c,
                "ci_95_low": "",
                "ci_95_high": "",
                "p_value": "",
                "null_mean": "",
                "interpretation": f"max_rank cap = {max_rank}",
            },
            {
                "statistic": "W_loc rank",
                "value": rank_l,
                "ci_95_low": "",
                "ci_95_high": "",
                "p_value": "",
                "null_mean": "",
                "interpretation": f"max_rank cap = {max_rank}",
            },
        ]
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = _make_figure(
        obs_overlap,
        ci_overlap,
        pval_overlap,
        null_overlap,
        obs_cca_r1,
        ci_cca,
        pval_cca,
        null_cca_r1,
        angles_deg,
        sv_loc,
        rank_c,
        rank_l,
        max_rank,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, "subspace_entanglement.csv"), index=False)
    fig.savefig(
        os.path.join(output_dir, "subspace_entanglement.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\n  Saved → {output_dir}/subspace_entanglement.{{csv,pdf}}")
    print(
        f"  Total time: " f"{int(elapsed//60)}m {elapsed%60:.1f}s"
        if elapsed >= 60
        else f"  Total time: {elapsed:.1f}s"
    )
    print("─" * 60)
    print(
        summary[["statistic", "value", "ci_95_low", "ci_95_high", "p_value"]].to_string(
            index=False
        )
    )
    print("─" * 60)

    return dict(
        overlap=obs_overlap,
        ci_overlap=ci_overlap,
        pval_overlap=pval_overlap,
        cca_r1=obs_cca_r1,
        ci_cca=ci_cca,
        pval_cca=pval_cca,
        angles_deg=angles_deg,
        rank_cell=rank_c,
        rank_loc=rank_l,
        sv_loc=sv_loc,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Figure
# ─────────────────────────────────────────────────────────────────────────────


def _make_figure(
    obs_overlap,
    ci_overlap,
    pval_overlap,
    null_overlap,
    obs_cca,
    ci_cca,
    pval_cca,
    null_cca,
    angles_deg,
    sv_loc,
    rank_c,
    rank_l,
    max_rank,
):
    """
    Four-panel publication figure:
      (a) Subspace overlap — null distribution vs observed
      (b) CCA r₁         — null distribution vs observed
      (c) Principal angles between W_cell and W_loc
      (d) Localization subspace scree (SVD singular values)

    Design principles:
      • Stats live in the subplot title, not in floating text boxes
        (avoids clipping and legend collision)
      • CI shown as capped error bar on the observed line, not as
        axis-wide shading that swamps the histogram
      • Top/right spines removed for a cleaner look
      • Consistent muted colour palette throughout
    """
    # ── Colour palette ────────────────────────────────────────────────────────
    HIST_COLOUR = "#B0BEC5"  # muted blue-grey for null histogram
    OBS_COLOUR = "#1565C0"  # strong blue for observed value
    CI_COLOUR = "#90CAF9"  # light blue for CI error bar
    ANGLE_HIGH = "#1565C0"  # blue  for angles ≥ 45° (near-orthogonal)
    ANGLE_LOW = "#C62828"  # red   for angles  < 45° (near-collinear)
    SCREE_COLOUR = "#546E7A"  # slate for scree bars

    # ── Layout: 4 equal columns ───────────────────────────────────────────────
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 4.2),
        gridspec_kw={"wspace": 0.42},
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.16)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Shared helper: null-distribution panel ────────────────────────────────
    def _null_panel(ax, null, obs, ci, pval, xlabel, panel_letter, short_name):
        """
        Histogram of the random-subspace null with observed value marked.
        CI shown as a capped error bar centred on the observed line.
        All statistics placed in the panel title — nothing floats inside.
        """
        ax.hist(
            null,
            bins=40,
            color=HIST_COLOUR,
            edgecolor="white",
            density=True,
            zorder=1,
            label="Random-subspace null",
        )

        # Observed value: solid vertical line
        ax.axvline(obs, color=OBS_COLOUR, lw=2.0, zorder=3)

        # 95% bootstrap CI: a horizontal bracket at 93% of the axes height (x in data
        # coordinates, y in axes coordinates)
        ax.plot(
            [ci[0], ci[1]],
            [0.93, 0.93],
            transform=ax.get_xaxis_transform(),  # x in data, y in axes
            color=CI_COLOUR,
            lw=3,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            [ci[0], ci[0]],
            [0.90, 0.96],
            transform=ax.get_xaxis_transform(),
            color=CI_COLOUR,
            lw=1.5,
            zorder=2,
        )
        ax.plot(
            [ci[1], ci[1]],
            [0.90, 0.96],
            transform=ax.get_xaxis_transform(),
            color=CI_COLOUR,
            lw=1.5,
            zorder=2,
        )
        ax.text(
            obs,
            0.965,
            transform=ax.get_xaxis_transform(),
            s="95% CI",
            ha="center",
            va="bottom",
            fontsize=7,
            color=CI_COLOUR,
        )

        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel("Density", fontsize=9.5)

        # Legend: only two entries, placed where there is space
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, fc=HIST_COLOUR, ec="white", label="Random-subspace null"
            ),
            Line2D([0], [0], color=OBS_COLOUR, lw=2, label="Observed"),
        ]
        ax.legend(
            handles=handles,
            fontsize=8,
            loc="upper left",
            framealpha=0.0,
            handlelength=1.5,
        )

        # Title carries all key numbers — no floating box needed
        sig_star = stars(pval) or "n.s."
        ax.set_title(
            (
                f"({panel_letter})  {short_name}\n"
                f"obs = {obs:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  "
                f"p = {pval:.4f}  {sig_star}"
            ),
            fontsize=9,
            loc="left",
            pad=6,
        )

    # ── Panel (a): subspace overlap ───────────────────────────────────────────
    _null_panel(
        axes[0],
        null_overlap,
        obs_overlap,
        ci_overlap,
        pval_overlap,
        xlabel="Subspace overlap coefficient",
        panel_letter="a",
        short_name="Subspace overlap  [0 = disent., 1 = entangled]",
    )

    # ── Panel (b): CCA r₁ ─────────────────────────────────────────────────────
    _null_panel(
        axes[1],
        null_cca,
        obs_cca,
        ci_cca,
        pval_cca,
        xlabel="First canonical correlation  r₁",
        panel_letter="b",
        short_name="CCA r₁  [0 = disent., 1 = entangled]",
    )

    # ── Panel (c): principal angles ───────────────────────────────────────────
    ax = axes[2]
    n_angles = len(angles_deg)
    bar_colors = [ANGLE_LOW if a < 45 else ANGLE_HIGH for a in angles_deg]
    ax.bar(
        range(n_angles),
        angles_deg,
        color=bar_colors,
        edgecolor="white",
        width=0.75,
        zorder=2,
    )
    ax.axhline(
        90, color="black", linestyle="--", lw=1.2, zorder=3, label="90° — orthogonal"
    )
    ax.axhline(
        45,
        color=ANGLE_LOW,
        linestyle=":",
        lw=1.2,
        zorder=3,
        label="45° — collinearity threshold",
    )
    ax.set_ylim(0, 105)
    ax.set_xlabel("Principal angle index", fontsize=9.5)
    ax.set_ylabel("Angle (degrees)", fontsize=9.5)
    ax.set_xticks(range(n_angles))
    ax.legend(fontsize=8, loc="lower left", framealpha=0.0)
    ax.set_title(
        f"(c)  Principal angles  θ₁ … θ_{n_angles}\n"
        f"W_cell rank = {rank_c}   W_loc rank = {rank_l}   "
        f"mean = {angles_deg.mean():.1f}°",
        fontsize=9,
        loc="left",
        pad=6,
    )
    # Colour legend as text
    ax.text(
        n_angles - 0.5,
        48,
        "< 45°: collinear",
        ha="right",
        fontsize=7.5,
        color=ANGLE_LOW,
    )
    ax.text(
        n_angles - 0.5,
        93,
        "≥ 45°: near-orth.",
        ha="right",
        fontsize=7.5,
        color=ANGLE_HIGH,
    )

    # ── Panel (d): localization subspace scree ────────────────────────────────
    ax = axes[3]
    ax.bar(
        range(len(sv_loc)),
        sv_loc,
        color=SCREE_COLOUR,
        edgecolor="white",
        width=0.65,
        zorder=2,
    )
    ax.set_xlabel("SVD component", fontsize=9.5)
    ax.set_ylabel("Singular value", fontsize=9.5)
    ax.set_xticks(range(len(sv_loc)))
    ax.set_title(
        f"(d)  Localization subspace — scree\n"
        f"{len(sv_loc)} pooled Fisher discriminants, top {max_rank} kept",
        fontsize=9,
        loc="left",
        pad=6,
    )
    # Cumulative variance annotation
    sv_sq = sv_loc**2
    cum_var = sv_sq.cumsum() / sv_sq.sum()
    ax2 = ax.twinx()
    ax2.plot(
        range(len(sv_loc)),
        cum_var,
        color="#FF8F00",
        marker="o",
        ms=4,
        lw=1.5,
        label="Cumul. var.",
    )
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Cumulative variance", fontsize=8.5, color="#FF8F00")
    ax2.tick_params(axis="y", colors="#FF8F00", labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#FF8F00")
    ax2.legend(fontsize=8, loc="center right", framealpha=0.0)

    return fig
