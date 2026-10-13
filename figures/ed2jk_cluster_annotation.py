"""Extended Data Fig. 2j,k -- the 13 k-means clusters of the tissue cells.

j  Mean z-scored nuclear and cell features per cluster, with the manual quality
   assessment (fraction of correct cells and the tiling, out-of-focus and
   outside-tissue flags). Features come from ``intensity_metrics.csv`` over all
   cells, log(1 + x)-transformed when heavy-tailed and z-scored within acquisition
   block, which removes the staining and exposure offset while keeping core-to-core
   differences. DAPI and nuclear shape use the nucleus mask; vimentin and
   microtubules use the whole-cell mask (cytoplasmic structures).
k  Cluster prevalence per donor (% of each donor's cells), with a Ward dendrogram
   of the retained clusters and bootstrap support (%) over resampled donors.

A cluster is retained for the dendrogram, and drawn above the rule, when at least
``naming.TISSUE_MIN_CORRECT`` (60%) of its cells are scored as correct ("yes" counts
as 100%, a percentage at face value, blank as 0). Prevalence profiles are z-scored
within cluster and clustered with Ward linkage (``subcell.utils.dendrogram``).

Rows carry canonical ids c1..c13 numbered by the dendrogram order of the per-core
prevalence table, whichever unit is drawn, so a cluster has one name in every
variant of the figure. The crosswalk to the clustering's own ids (gapped, inherited
from the original k = 19 labelling) is written to ``config.TISSUE_CLUSTER_IDS`` and
is what ``naming.load_cluster_ids`` reads. The clustering is matched to the reference
(k = 13) labelling on which the manual quality assessment was done
(``config.TISSUE_REF_LABELS``) before the annotation is applied; the ids are
identical, so the annotation applies verbatim.

    python -m figures.ed2jk_cluster_annotation [--unit patient|core] [--out-dir DIR]

Inputs   ``config.TISSUE_CLUSTERS``/cell_clusters.csv, ``config.TISSUE_ANNOT_K13``,
         ``config.TISSUE_REF_LABELS``, ``config.TISSUE_INTENSITY``
Outputs  ``config.figure_dir("ed2")``/ed2jk_cluster_annotation (pdf, png, svg), and
         ``config.TISSUE_CLUSTER_IDS``. Also written next to the clustering (an
         input directory): cluster_annotation_measured.csv, the figure's source data.
"""

from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from subcell import config, naming, style
from subcell.runlog import log
from subcell.style.palettes import (CONDITION_PAL as COND, GRID, INK, INK2, MUTED,
                                    QUALITY_SCALE as QCOL, SURFACE, TISSUE_PAL as TISSUE)
from subcell.tissue import prevalence_support
from subcell.tissue.prevalence_support import UNIT_COLUMN, prevalence, unit_order
from subcell.tissue.clustering import MEASURED, aligned_intensity
from subcell.utils import dendrogram, zscore

B_BOOT = 2000
SEED = 0
NORM_KEY = "slide"                        # acquisition block
MIN_CORRECT = naming.TISSUE_MIN_CORRECT   # minimum share of correct cells to enter the tree
UNIT_NAME = {"patient": "donor", "core": "core"}

# ends and midpoint of the diverging scale, as hex for the hand-drawn heatmap cells
DIV_LO, DIV_MID, DIV_HI = "#2166ac", "#f2f2f2", "#b2182b"
ZLIM = 2.0

QUAL = [("correct", "Correct\ncells"), ("tiling", "Tiling\nartefact"),
        ("focus", "Out of\nfocus"), ("outside", "Outside\ntissue")]

# Layout, in inches: the 183 mm Nature double-column width with the two panels
# stacked, so the 29 prevalence columns and the 12 feature columns each get the
# full width. Type sizes are at or above the 5 pt floor.
FIG_W = 183.0 / 25.4
MARGIN = 0.26                  # clear space left and right
GAPB, GAPG = 0.14, 0.075       # between column blocks / between tissue-condition groups
PANEL_GAP = 0.24
CELL_H = 0.185
SPL = 0.30                     # gap between retained and excluded rows
ID_W, N_W, GUT = 0.06, 0.52, 0.34
DNW = 0.86                     # dendrogram width
HEAD_A, HEAD_B = 0.74, 1.55    # header heights of the two panels
LEG_H, LEG_A = 0.14, 0.44      # legend strips
FS = dict(cell=5.5, head=5.0, ident=6.5, name=5.5, title=6.5, leg=5.5, core=5.0,
          band=5.5, panel=8.0, note=5.0, boot=5.5)
DARK_FILLS = (COND["Cancer"],)   # fills that need white text


# ------------------------------------------------------------------- inputs


def load():
    meta = pd.read_csv(config.TISSUE_CLUSTERS / "cell_clusters.csv",
                       usecols=["id", "cluster", "core_id", "patient", "slide",
                                "tissue", "cancer"])
    return meta, prevalence_support.load_annotation()


def check_labels(meta):
    """Match the clustering to the reference labelling the annotation was scored on."""
    ref = pd.read_csv(config.TISSUE_REF_LABELS, usecols=["id", "cluster"]).rename(
        columns={"cluster": "ref"})
    d = meta[["id", "cluster"]].merge(ref, on="id", how="inner")
    C = pd.crosstab(d["cluster"], d["ref"])
    r, c = linear_sum_assignment(-C.to_numpy())
    mp = {C.index[i]: C.columns[j] for i, j in zip(r, c)}
    agree = sum(int(C.loc[k, mp[k]]) for k in C.index)
    renamed = {k: v for k, v in mp.items() if k != v}
    log(f"label transfer from the reference labelling: {len(d):,} shared cells, "
        f"{100 * agree / len(d):.2f}% agree, "
        f"ARI {adjusted_rand_score(d['cluster'], d['ref']):.4f}")
    if renamed:
        log(f"  renames applied: {renamed}")
        meta["cluster"] = meta["cluster"].map(lambda k: mp.get(k, k))
    else:
        log("  all ids identical -- the annotation applies directly")
    return meta


def measured(meta):
    """Mean z-scored measurement per cluster (z within acquisition block)."""
    im = aligned_intensity(meta["id"])
    E = zscore.log_heavy_tailed(im[[c for c, _ in MEASURED]].to_numpy())
    Z = zscore.zscore_within(E, meta[NORM_KEY].to_numpy())
    return pd.DataFrame(Z, columns=[lbl for _, lbl in MEASURED]).groupby(
        meta["cluster"].to_numpy()).mean()


# ------------------------------------------------------------------ drawing


def sentence_case(header: str) -> str:
    """Two-line column header with the second line in lower case (acronyms such as CV kept)."""
    first, *rest = header.split("\n")
    return "\n".join([first] + [" ".join(w if w.isupper() else w.lower() for w in r.split())
                               for r in rest])


def cell(ax, x, y, w, h, face, txt="", fs=7.2, tc=None, alpha=1.0, ec=GRID, clip=True):
    # clip=False for anything drawn above the grid (the header bands sit at negative
    # y, outside the axes limits, and patches are clipped to the axes by default)
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=ec, lw=0.6,
                           zorder=2, alpha=alpha, clip_on=clip))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc or INK, zorder=3, clip_on=clip)


def blues(v, vmax):
    t = 0.0 if vmax <= 0 else float(np.clip(v / vmax, 0, 1))
    return plt.get_cmap("Blues")(0.06 + 0.84 * t)


def diverging(z):
    t = float(np.clip(z / ZLIM, -1, 1))
    pole = mpl.colors.to_rgb(DIV_LO if t < 0 else DIV_HI)
    mid = mpl.colors.to_rgb(DIV_MID)
    f = abs(t)
    return tuple(mid[i] + f * (pole[i] - mid[i]) for i in range(3))


def annotate_support(dax, L, lab, support, fsize):
    """Support % just inside each join, right-aligned with a white halo.

    The gap immediately left of a join, level with the join's midpoint, is empty
    because the child links sit at the children's own heights.
    """
    n = len(lab)
    mem = {i: frozenset([lab[i]]) for i in range(n)}
    pos = {li: 5.0 + 10.0 * r
           for r, li in enumerate(sch.dendrogram(L, no_plot=True)["leaves"])}
    dx = 0.035 * float(L[:, 2].max())
    for i, (x, y, h, _) in enumerate(L):
        x, y = int(x), int(y)
        mem[n + i] = mem[x] | mem[y]
        pos[n + i] = (pos[x] + pos[y]) / 2.0
        s = support.get(mem[n + i])
        if s is None:
            continue
        t = dax.text(h - dx, pos[n + i], f"{s:.0f}", fontsize=fsize,
                     color=INK if s >= 70 else MUTED, ha="right", va="center",
                     fontweight="bold" if s >= 70 else "normal", clip_on=False,
                     zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def render(a, M, P, units, spans, order, kept_order, drop, sizes, newid, L, lab,
           support, unit):
    """Draw panel j (features and quality) above panel k (prevalence)."""
    fs = FS
    lab_w = GUT + ID_W + N_W
    n_a = len(MEASURED) + len(QUAL)
    grid_h = len(order) * CELL_H + SPL
    ytop = len(kept_order) * CELL_H

    content = FIG_W - 2 * MARGIN
    CW = (content - lab_w - 2 * GAPB) / n_a
    PVW = (content - GUT - DNW - GAPB - (len(spans) - 1) * GAPG) / len(units)
    # height is derived from the content so no header or legend is clipped
    FIG_H = LEG_H + grid_h + HEAD_B + PANEL_GAP + LEG_A + grid_h + HEAD_A + 0.10
    ya = LEG_H + grid_h + HEAD_B + PANEL_GAP + LEG_A
    yb = LEG_H

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fx, fy = (lambda v: v / FIG_W), (lambda v: v / FIG_H)

    def panel(x_in, y_in, w_in, h_in):
        ax = fig.add_axes([fx(x_in), fy(y_in), fx(w_in), fy(h_in)])
        ax.set_xlim(0, w_in)
        ax.set_ylim(0, h_in)
        ax.invert_yaxis()
        ax.axis("off")
        return ax

    axa = panel(MARGIN, ya, content, grid_h)
    axb = panel(MARGIN, yb, content, grid_h)

    def rowy(i):
        return i * CELL_H + (SPL if i >= len(kept_order) else 0)

    def rule(ax, x1):
        ax.plot([0, x1], [ytop + SPL / 2] * 2, color=INK2, lw=0.8, clip_on=False, zorder=6)

    def head(ax, x, w, txt, size):
        ax.text(x + w / 2, -0.05, txt, ha="center", va="bottom", fontsize=size,
                color=INK, linespacing=1.28, clip_on=False)

    def title(ax, x0, x1, txt, y):
        ax.plot([x0, x1], [y, y], color=INK2, lw=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.035, txt, ha="center", va="bottom",
                fontsize=fs["title"], color=INK, clip_on=False)

    # ---------------------------------------------------------------- panel j
    x_n = GUT + ID_W
    x_me = lab_w + GAPB
    x_qu = x_me + CW * len(MEASURED) + GAPB
    for i, k in enumerate(order):
        y, r = rowy(i), a.loc[k]
        excl = k in drop
        al = 0.45 if excl else 1.0
        ink = MUTED if excl else INK
        axa.text(0.0, y + CELL_H / 2, f"c{newid[k]}", ha="left", va="center",
                 fontsize=fs["ident"], fontweight="bold", color=ink)
        axa.text(x_n + N_W - 0.03, y + CELL_H / 2, f"{sizes[k]:,}", ha="right",
                 va="center", fontsize=fs["name"], color=ink)
        for c, (_, lbl) in enumerate(MEASURED):
            z = float(M.loc[k, lbl])
            cell(axa, x_me + c * CW, y, CW, CELL_H, diverging(z), f"{z:+.2f}",
                 fs["cell"], "white" if abs(z) > 0.62 * ZLIM else INK, alpha=al)
        for c, (col, _) in enumerate(QUAL):
            v = str(r[col]).strip()
            face = (QCOL["yes"] if v.lower() == "yes"
                    else QCOL["pct"] if v.endswith("%") else QCOL[""])
            cell(axa, x_qu + c * CW, y, CW, CELL_H, face, v, fs["cell"],
                 "white" if face == QCOL["yes"] else INK2, alpha=al)
    rule(axa, x_qu + CW * len(QUAL))
    axa.text(0.0, ytop + SPL / 2 + 0.030,
             f"Excluded from the prevalence hierarchy — under {MIN_CORRECT:.0%} correct cells",
             ha="left", va="top", fontsize=fs["note"], color=MUTED)
    head(axa, x_n, N_W, "\nCells", fs["head"])
    for c, (_, lbl) in enumerate(MEASURED):
        head(axa, x_me + c * CW, CW, sentence_case(lbl), fs["head"])
    for c, (_, lbl) in enumerate(QUAL):
        head(axa, x_qu + c * CW, CW, lbl, fs["head"])
    ty = -0.42
    title(axa, x_me, x_me + CW * len(MEASURED),
          "Measured  (z within acquisition block  ·  CV = SD ÷ mean)", ty)
    title(axa, x_qu, x_qu + CW * len(QUAL), "Quality", ty)
    axa.text(0.0, ty, "j", ha="left", va="bottom", fontsize=fs["panel"],
             fontweight="bold", color=INK, clip_on=False)

    # ---------------------------------------------------------------- panel k
    x_pv = GUT
    xcol, off = [], 0.0
    for i0, i1, _, _ in spans:
        for i in range(i0, i1):
            xcol.append(x_pv + i * PVW + off)
        off += GAPG
    pv_w = len(units) * PVW + (len(spans) - 1) * GAPG
    pvmax = float(P.to_numpy().max())
    for i, k in enumerate(order):
        y = rowy(i)
        al = 0.62 if k in drop else 1.0
        axb.text(0.0, y + CELL_H / 2, f"c{newid[k]}", ha="left", va="center",
                 fontsize=fs["ident"], fontweight="bold", color=MUTED if k in drop else INK)
        for c, u in enumerate(units):
            cell(axb, xcol[c], y, PVW, CELL_H, blues(float(P.loc[k, u]), pvmax),
                 alpha=al, ec="#ececec")
    rule(axb, x_pv + pv_w)
    for c, u in enumerate(units):
        axb.text(xcol[c] + PVW / 2, -0.04, u.replace("_", " ") if unit == "core" else str(u),
                 ha="left", va="bottom", fontsize=fs["core"], color=INK, rotation=90,
                 clip_on=False)
    bh = 0.145
    y_cond = -0.60
    y_tis = y_cond - bh - 0.055
    for i0, i1, _, cond in spans:
        x0, x1 = xcol[i0], xcol[i1 - 1] + PVW
        cell(axb, x0, y_cond, x1 - x0, bh, COND[cond], cond, fs["band"],
             "white" if COND[cond] in DARK_FILLS else INK, ec="none", clip=False)
    for tis in ("Pancreas", "Liver"):
        sp = [q for q in spans if q[2] == tis]
        x0, x1 = xcol[sp[0][0]], xcol[sp[-1][1] - 1] + PVW
        cell(axb, x0, y_tis, x1 - x0, bh, TISSUE[tis], tis, fs["band"] + 0.4,
             "white", ec="none", clip=False)
    tyb = y_tis - 0.10
    name = UNIT_NAME[unit]
    title(axb, x_pv, x_pv + pv_w,
          f"Prevalence per {name}  (% of that {name}'s cells)", tyb)
    axb.text(0.0, tyb, "k", ha="left", va="bottom", fontsize=fs["panel"],
             fontweight="bold", color=INK, clip_on=False)

    dax = fig.add_axes([fx(MARGIN + x_pv + pv_w + GAPB), fy(yb + grid_h - ytop),
                        fx(DNW - 0.10), fy(ytop)])
    with plt.rc_context({"lines.linewidth": 1.0}):
        sch.dendrogram(L, orientation="right", no_labels=True,
                       link_color_func=lambda _: INK2, ax=dax)
    annotate_support(dax, L, lab, support, fs["boot"])
    dax.invert_yaxis()
    dax.set_facecolor(SURFACE)
    dax.set_xticks([])
    dax.set_yticks([])
    for s in dax.spines.values():
        s.set_visible(False)
    dax.set_title("Ward linkage\nbootstrap %", fontsize=fs["head"], color=INK,
                  pad=2, linespacing=1.28)

    # ---------------------------------------------------------------- legends
    sw_w, sw_step = 0.34, 0.375
    fsz = fs["leg"]
    fsz_sw = max(fsz - 0.8, style.FS_MIN)

    def lab_width(t):
        # 0.52 em per character is the average advance width of Arial/Helvetica
        # lower case; it only has to be right enough to space the blocks
        return len(t) * 0.52 * fsz / 72.0 + 0.04

    def legend_row(blocks, x0, x1, y):
        """Lay the legend blocks out evenly between x0 and x1 on one row."""
        widths = [lab_width(t) + len(items) * sw_step for t, items in blocks]
        gap = max(0.10, (x1 - x0 - sum(widths)) / max(len(blocks) - 1, 1))
        cx = x0
        for (ttl, items), w in zip(blocks, widths):
            fig.text(fx(cx), fy(y), ttl, fontsize=fsz, color=INK2, ha="left", va="center")
            sx = cx + lab_width(ttl)
            for lbl, col in items:
                fig.patches.append(Rectangle(
                    (fx(sx), fy(y - 0.060)), fx(sw_w), fy(0.120), facecolor=col,
                    edgecolor=GRID, lw=0.5, transform=fig.transFigure, figure=fig))
                fig.text(fx(sx + sw_w / 2), fy(y), lbl, fontsize=fsz_sw, ha="center",
                         va="center", color="white" if col in DARK_FILLS else INK)
                sx += sw_step
            cx += w + gap

    def vstrip(items, x, y_top, title_txt):
        """Vertical key drawn downward from y_top (the prevalence ramp), in the space
        beside the excluded rows, where the dendrogram does not reach."""
        w, h = 0.30 * (fsz / 6.2), 0.155 * (fsz / 6.2)
        fig.text(fx(x), fy(y_top), title_txt, fontsize=fsz, color=INK2, ha="left", va="bottom")
        cy = y_top - 0.09 - h
        for lbl, col in items:
            fig.patches.append(Rectangle(
                (fx(x), fy(cy)), fx(w), fy(h), facecolor=col, edgecolor=GRID, lw=0.5,
                transform=fig.transFigure, figure=fig))
            if lbl:
                fig.text(fx(x + w + 0.06), fy(cy + h / 2), lbl, fontsize=fsz_sw,
                         ha="left", va="center", color=INK)
            cy -= h

    legend_row([
        ("Measured z:", [(f"−{ZLIM:g}", diverging(-ZLIM)), ("", diverging(-1.0)),
                         ("0", diverging(0.0)), ("", diverging(1.0)),
                         (f"+{ZLIM:g}", diverging(ZLIM))]),
        ("Quality:", [("blank", QCOL[""]), ("%", QCOL["pct"]), ("yes", QCOL["yes"])]),
    ], MARGIN + GUT, MARGIN + content, ya - LEG_A * 0.55)
    vstrip([("max", blues(100, 100)), ("", blues(70, 100)), ("", blues(35, 100)),
            ("0", blues(0, 100))],
           MARGIN + x_pv + pv_w + GAPB, yb + (len(order) - len(kept_order)) * CELL_H,
           "Prevalence")

    return fig


def main(out_dir=None, unit="patient"):
    style.use_agg()
    rng = np.random.default_rng(SEED)
    meta, a = load()
    meta = check_labels(meta)
    assert set(meta["cluster"].unique()) == set(a.index), "annotation/cluster mismatch"
    M = measured(meta)
    units, spans = unit_order(meta, unit)
    P = prevalence(meta, unit, units)
    sizes = meta["cluster"].value_counts()

    keep = prevalence_support.retained_clusters(a)
    kept_order, L, lab, support = dendrogram.ward_bootstrap(P.loc[keep], B_BOOT, rng)
    drop = [k for k in a.index if k not in keep]
    order = kept_order + drop

    # Canonical ids c1..c13: always from the per-core dendrogram (kept clusters in
    # its leaf order, then the excluded ones), so a cluster has the same number in
    # every unit's figure. Source ids are gapped because they were inherited from
    # the original k = 19 labelling; the crosswalk keeps the renumbering reversible.
    core_units, _ = unit_order(meta, "core")
    core_order, *_ = dendrogram.ward_bootstrap(
        prevalence(meta, "core", core_units).loc[keep], B_BOOT, np.random.default_rng(SEED))
    canon_order = core_order + drop
    newid = {k: i + 1 for i, k in enumerate(canon_order)}
    ids_path = config.TISSUE_CLUSTER_IDS
    config.ensure(ids_path.parent)
    pd.DataFrame({"cluster": [f"c{newid[k]}" for k in canon_order],
                  "source_cluster": [f"c{k}" for k in canon_order],
                  "in_clustering": [k in keep for k in canon_order]}).to_csv(
        ids_path, index=False)

    log("canonical id -> source label: " + ", ".join(f"c{v}={k}" for k, v in newid.items()))
    log(f"in the linkage ({len(keep)}): " + ", ".join(f"c{newid[k]}" for k in kept_order))
    for c, v in sorted(support.items(), key=lambda kv: -kv[1]):
        log(f"  bootstrap {' '.join(f'c{newid[x]}' for x in sorted(c, key=lambda z: newid[z])):18s} {v:5.1f}%")

    stem = "ed2jk_cluster_annotation" + ("" if unit == "patient" else "_by_core")
    with style.nature():
        fig = render(a, M, P, units, spans, order, kept_order, drop, sizes, newid, L,
                     lab, support, unit)
        style.save(fig, stem, out_dir or config.figure_dir("ed2"),
                   formats=("pdf", "png", "svg"))

    out = a.loc[order].copy()
    out.insert(0, "cluster", [f"c{newid[k]}" for k in order])
    out.insert(1, "source_label", order)
    out["n_cells"] = [sizes[k] for k in order]
    out["in_clustering"] = [k in keep for k in order]
    for _, lbl in MEASURED:
        out["z_" + lbl.replace("\n", " ")] = M.loc[order, lbl].values
    for u in units:
        out[f"prev_{u}"] = P.loc[order, u].values
    out.to_csv(config.TISSUE_CLUSTERS / "cluster_annotation_measured.csv", index=False)
    log(f"  -> {config.TISSUE_CLUSTERS / 'cluster_annotation_measured.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unit", default="patient", choices=sorted(UNIT_COLUMN),
                    help="prevalence unit: donor (paper, default) or core")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.out_dir, a.unit)
