"""Extended Data Fig. 1: robustness of SubCell embeddings to a change of microscope (Supplementary Note 1).

Each function draws one plot at a position in millimetres on a figure sized for Nature's Extended Data page
(180 x 170 mm; 5-7 pt sans-serif type, strokes 0.25-1 pt):
  images      a: one cell of each line on the widefield 40x and confocal 63x microscope (colour composite + the four
              channels in their colours)
  retrieval   b: normalised mAP@1 / @5 / @10: cell, cell-line and phase identity across the microscopes
  transfer    c: readouts within one microscope (x) against across microscopes (y): phase, cell line, GMNN / CDT1
  geometry    d: linear CKA and Mantel correlation between the 40x and 63x embeddings
  direction   e: (63x -> 40x) - (40x -> 63x) for each readout
  gallery     f-m: each model's five most similar cells for four 40x and four 63x queries, with similarity profiles
              (single channels in their colours; the channel names on the first query of each row)
  model_key   the representation colours
Inputs: the tables of ``subcell.microscope_robustness`` (config.MICROSCOPE_DIR) and the crops (config.MICROSCOPE_CROPS).

    python -m figures.ed1_microscopes [--out-dir DIR]

Each plot is written on its own (SVG and PDF) to ``config.figure_dir("ed1")``: ed1a_images, ed1b_retrieval,
ed1c_transfer, ed1d_geometry, ed1e_direction, ed1f-m_gallery and ed1_key.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from figures import _microscope_images as img
from subcell import config, style
from subcell.runlog import log
from subcell.microscope_robustness import data
from subcell.microscope_robustness.examples import DIRS, K
from subcell.style import layout as L, palettes as pal

FS_TICK, FS_LAB = style.FS_TINY, style.FS_SMALL                      # 5 pt ticks / numbers / names, 6 pt labels / titles
CI_LW, MS = 0.6, 3.2
ORDER, NAME = data.ORDER, data.NAME
# Colour-blind-safe representation colours, from the published Okabe-Ito and Tol palettes: every pair, the black
# reference included, stays apart under protanopia and deuteranopia (OKLab delta E >= 10.6) and with normal vision
# (>= 17.4), so identity holds in the scatter plots too. Orange and sky blue are below 3:1 contrast on white, so the
# plots name the representations (axis labels, key) and keep numbers in ink.
COLOR = dict(zip(data.MODELS, ["#0072B2", "#CC3311", "#E69F00", "#56B4E9", "#009E73", "#882255"])) | {data.PIXELS: "#000000"}
KS = [("map1", "mAP@1", 1.0), ("map5", "mAP@5", 0.6), ("map10", "mAP@10", 0.3)]
TASKS = [("cell", "Cell identity"), ("cell_line", "Cell-line identity"), ("phase", "Phase identity")]
READOUTS = [("phase", "Phase", "balanced accuracy"), ("cell_line", "Cell line", "balanced accuracy"), ("gmnn_cdt1", "GMNN / CDT1", "Spearman ρ")]
MARK = {"phase": "o", "cell_line": "^", "gmnn_cdt1": "D"}
ARROW = "→"                                                           # the font's own glyph: every label stays in Arial
HEAD = 4.4                                                            # gallery block header: line name, then the rank numbers


def tables():
    d = config.MICROSCOPE_DIR
    return dict(R=pd.read_csv(d / "identity.csv"), T=pd.read_csv(d / "transfer.csv"), G=pd.read_csv(d / "geometry.csv"),
                D=pd.read_csv(d / "transfer_direction_delta.csv"), Gal=pd.read_csv(d / "gallery.csv"),
                sample=pd.read_csv(d / "panel_a_cells.csv").set_index("cell_line").loc[data.LINES])


def _yticks(ax, ticks):
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0" if t == 0 else f"{t:.1f}".replace("-", "−") for t in ticks])


def _model_axis(ax, lo, hi, ticks, ylabel=None):
    """the representations on x (ORDER), named vertically; zero line; y ticks"""
    n = len(ORDER)
    ax.axhline(0, color=pal.INK, lw=0.4, zorder=3)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(n))
    ax.set_xticklabels([img.wrap(NAME[r]) for r in ORDER], rotation=90, ha="right", va="center", rotation_mode="anchor", fontsize=FS_TICK,
                       linespacing=0.9, multialignment="right")
    ax.tick_params(axis="x", length=0, pad=1.0)
    _yticks(ax, ticks)
    ax.tick_params(axis="y", labelsize=FS_TICK, length=1.5, width=0.4, pad=1.0)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LAB, labelpad=1.0)


def _shade_key(fig, x, y, entries, box=1.8):
    for lab, alpha in entries:
        ax = L.axes(fig, x, y - box / 2, box, box)
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=to_rgba(pal.INK2, alpha), edgecolor=pal.INK2, lw=0.3))
        L.bare(ax)
        tx = L.text(fig, x + box + 0.8, y, lab, fontsize=FS_TICK, va="center")
        x += box + 0.8 + tx.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4 + 2.2


def model_key(fig, xc, y, box=2.2, gap=3.5):
    """one colour key for the representations, centred on xc"""
    rend = fig.canvas.get_renderer()
    tw = []
    for r in ORDER:
        tx = L.text(fig, 0, y, NAME[r], fontsize=FS_LAB, va="center")
        tw.append(tx.get_window_extent(rend).width / fig.dpi * 25.4)
        tx.remove()
    sep = 0.9
    x = xc - (sum(tw) + len(ORDER) * (box + sep) + (len(ORDER) - 1) * gap) / 2
    for r, w in zip(ORDER, tw):
        ax = L.axes(fig, x, y - box / 2, box, box)
        ax.add_patch(Rectangle((0, 0), 1, 1, color=COLOR[r], lw=0))
        L.bare(ax)
        L.text(fig, x + box + sep, y, NAME[r], fontsize=FS_LAB, va="center")
        x += box + sep + w + gap


def images(fig, x0, y0, sample, c=10.4, g=0.25, gx=1.2, lab_w=4.6):
    """columns = cell lines; per microscope the composite and its four channels in a column beside it"""
    crops = {(l, m): img.window(sample.loc[l, "id"], m) for l in data.LINES for m in data.MAGS}
    lim = {m: img.limits([crops[(l, m)] for l in data.LINES], "zero" if m == "63x" else "percrop") for m in data.MAGS}
    q = (c - 3 * g) / 4
    colw = c + g + q
    y = y0 + 2.6
    for j, l in enumerate(data.LINES):
        L.text(fig, x0 + lab_w + j * (colw + gx) + colw / 2, y0 + 2.2, l, fontsize=FS_TICK, ha="center", va="bottom")
    for i, (m, rowlab) in enumerate((("40x", "Widefield\n40×"), ("63x", "Confocal\n63×"))):
        L.text(fig, x0 + lab_w - 0.5, y + c / 2, rowlab, fontsize=FS_TICK, ha="right", va="center", rotation=90, linespacing=0.95,
               multialignment="center")
        for j, l in enumerate(data.LINES):
            xx = x0 + lab_w + j * (colw + gx)
            cr = crops[(l, m)]
            ax = L.axes(fig, xx, y, c, c)
            img.show(ax, img.render(cr, lim[m], size=240))
            if j == len(data.LINES) - 1 and i == 1:                    # 10-um scale bar
                n = 10.0 / img.WINDOW_UM * 240
                ax.plot([240 * 0.94 - n, 240 * 0.94], [240 * 0.93] * 2, color="white", lw=0.8, solid_capstyle="butt")
                ax.text(240 * 0.94, 240 * 0.88, "10 µm", color="white", fontsize=FS_TICK, ha="right", va="bottom")
            for k, (ch, _, _) in enumerate(img.CHANNELS):
                ax = L.axes(fig, xx + c + g, y + k * (q + g), q, q)
                img.show(ax, img.render(cr, lim[m], size=64, only=ch))
        y += c + 0.8
    img.channel_key(fig, x0 + lab_w, y + 0.4, fontsize=FS_TICK, names={**img.KEY_NAME, "red": "Microtubules"}, gap=2.0)
    return lab_w + 4 * colw + 3 * gx, y + 2.6 - y0


def retrieval(fig, x0, y0, w, h, R, ylab_w=6.0, gap=5.0, bar_w=0.27):
    """normalised mAP@1 / @5 / @10 as adjacent bars per representation (shades of its colour), 95% CIs"""
    pw = (w - ylab_w - 2 * gap) / 3
    his = {t: 1.0 if float(R[R.task == t].ci_hi.max()) > 0.62 else 0.6 for t, _ in TASKS}
    below = max(-min(0.0, np.floor(float(R[R.task == t].ci_lo.min()) * 20) / 20) / his[t] for t, _ in TASKS)   # the same share of
    for pi, (t, title) in enumerate(TASKS):                                                                   # each panel below 0:
        px = x0 + ylab_w + pi * (pw + gap)                                                                    # the zero lines align
        d = R[R.task == t].set_index(["run", "metric"])
        hi = his[t]
        ax = L.axes(fig, px, y0, pw, h)
        for i, r in enumerate(ORDER):
            for k, (met, _, alpha) in enumerate(KS):
                v, lo_, hi_ = d.loc[(r, met), ["value", "ci_lo", "ci_hi"]]
                xx = i + (k - 1) * bar_w
                ax.bar(xx, v, width=bar_w * 0.92, color=to_rgba(COLOR[r], alpha), edgecolor=COLOR[r], lw=0.3, zorder=1)
                ax.plot([xx, xx], [lo_, hi_], color=pal.INK, lw=0.4, solid_capstyle="butt", zorder=2)
        _model_axis(ax, -below * hi, hi, (0, 0.5, 1.0) if hi == 1.0 else (0, 0.2, 0.4, 0.6),
                    "Normalised mAP" if pi == 0 else None)
        L.text(fig, px + pw / 2, y0 - 0.8, f"{title} (n = {int(d.n_cells.iloc[0]):,})", fontsize=FS_LAB, ha="center", va="bottom")
    _shade_key(fig, x0 + ylab_w, y0 - 4.6, [(lab, a) for _, lab, a in KS])


def _scatter(fig, px, py, s, lab, pts, lo, hi, chance, title, ylabel):
    ax = L.axes(fig, px, py, s, s)
    ax.plot([lo, hi], [lo, hi], color=pal.GRID, lw=0.5, zorder=1)
    if chance is not None:
        ax.axhline(chance, color=pal.INK2, lw=0.4, ls=(0, (3, 2)), zorder=1)
        ax.axvline(chance, color=pal.INK2, lw=0.4, ls=(0, (3, 2)), zorder=1)
    for r in ORDER:
        x, xl, xh, y, yl, yh = pts[r]
        ax.plot([xl, xh], [y, y], color=COLOR[r], lw=CI_LW, zorder=2, clip_on=False)
        ax.plot([x, x], [yl, yh], color=COLOR[r], lw=CI_LW, zorder=2, clip_on=False)
        ax.scatter(x, y, s=10, color=COLOR[r], edgecolor=pal.INK, lw=0.3, zorder=3, clip_on=False)
    step = 0.085 * (hi - lo)
    last = lo + 0.03 * (hi - lo) - step
    ly = {}
    for r in sorted(ORDER, key=lambda r: pts[r][3]):                     # labels without overlap, in the order of y
        ly[r] = max(pts[r][3], last + step)
        last = ly[r]
    over = max(ly.values()) - (hi - 0.02 * (hi - lo))
    if over > 0:
        ly = {r: v - over for r, v in ly.items()}
    lx = hi + 0.05 * (hi - lo)
    for r in ORDER:
        ax.plot([pts[r][0], lx - 0.01 * (hi - lo)], [pts[r][3], ly[r]], color="#d9d9d9", lw=0.25, zorder=1, clip_on=False)
        ax.scatter(lx + 0.02 * (hi - lo), ly[r], s=5, color=COLOR[r], edgecolor=pal.INK, lw=0.25, zorder=3, clip_on=False)
        ax.text(lx + 0.06 * (hi - lo), ly[r], f"{pts[r][3] - pts[r][0]:+.2f}".replace("-", "−"), fontsize=FS_TICK, ha="left", va="center",
                color=pal.INK, clip_on=False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0, 0.5, 1.0) if lo <= t <= hi] if chance is None else [t for t in (0.25, 0.5, 0.75, 1.0) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=FS_TICK, length=1.5, width=0.4, pad=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.4)
    ax.set_xlabel("Within microscope", fontsize=FS_LAB, labelpad=1.0)
    if ylabel:
        ax.set_ylabel("Across microscopes", fontsize=FS_LAB, labelpad=1.0)
    L.text(fig, px + (s + lab) / 2, py - 0.8, title, fontsize=FS_LAB, ha="center", va="bottom")


def transfer(fig, x0, y0, s, T, lab=7.5, gap=5.0, ylab_w=6.0):
    """within one microscope (x) against across microscopes (y) per readout; 95% CIs on both axes; numbers, y - x"""
    for pi, (t, title, unit) in enumerate(READOUTS):
        d = T[T.target == t].set_index(["kind", "run"])
        acc = unit != "Spearman ρ"
        lo, hi = (0.2, 1.0) if acc else (-0.05, 1.0)
        pts = {r: (*d.loc[("within", r), ["value", "ci_lo", "ci_hi"]], *d.loc[("across", r), ["value", "ci_lo", "ci_hi"]]) for r in ORDER}
        _scatter(fig, x0 + ylab_w + pi * (s + lab + gap), y0, s, lab, pts, lo, hi, float(d.chance.iloc[0]) if acc else None,
                 f"{title} ({unit})", pi == 0)
    return ylab_w + 3 * (s + lab) + 2 * gap


def geometry(fig, x0, y0, w, h, G, ylab_w=5.0):
    """linear CKA and Mantel correlation side by side per representation (full and light shade of its colour)"""
    g = G.set_index("run")
    ax = L.axes(fig, x0 + ylab_w, y0, w - ylab_w, h)
    for i, r in enumerate(ORDER):
        for k, (col, alpha) in enumerate((("cka_linear", 1.0), ("mantel", 0.45))):
            ax.bar(i + (k - 0.5) * 0.38, float(g.loc[r, col]), width=0.36, color=to_rgba(COLOR[r], alpha), edgecolor=COLOR[r], lw=0.3, zorder=1)
    _model_axis(ax, 0, 1.0, (0, 0.5, 1.0), "40× vs 63×")
    L.text(fig, x0 + ylab_w + (w - ylab_w) / 2, y0 - 0.8, "Embedding geometry", fontsize=FS_LAB, ha="center", va="bottom")
    _shade_key(fig, x0 + ylab_w, y0 - 4.6, [("Linear CKA", 1.0), ("Mantel", 0.45)])


def direction(fig, x0, y0, w, h, D, ylab_w=5.0, dodge=0.27):
    """(63x -> 40x) - (40x -> 63x) per representation and readout (marker), 95% paired CI; filled, Holm P < 0.05"""
    d = D[D.comparison == "across"].set_index(["target", "run"])
    ax = L.axes(fig, x0 + ylab_w, y0, w - ylab_w, h)
    for i, r in enumerate(ORDER):
        for k, (t, _, _) in enumerate(READOUTS):
            v, lo, hi, p = d.loc[(t, r), ["delta", "ci_lo", "ci_hi", "p_holm"]]
            xx = i + (k - 1) * dodge
            ax.plot([xx, xx], [lo, hi], color=COLOR[r], lw=CI_LW, zorder=2)
            ax.plot(xx, v, marker=MARK[t], ms=2.6, color=COLOR[r], mfc=COLOR[r] if p < 0.05 else "white", mec=COLOR[r], mew=0.5, ls="none",
                    zorder=3)
    _model_axis(ax, -0.5, 0.6, (-0.4, 0, 0.4), "Δ")
    for j, (fill, lab) in enumerate(((True, "P < 0.05"), (False, "P ≥ 0.05"))):  # the fill, in the empty top left (Holm P, legend)
        yk = 0.6 - (0.05 + 0.085 * j) * 1.1
        ax.plot(-0.35, yk, marker="s", ms=2.2, color=pal.INK2, mfc=pal.INK2 if fill else "white", mec=pal.INK2, mew=0.5, ls="none",
                zorder=4)
        ax.text(-0.05, yk, lab, fontsize=FS_TICK, ha="left", va="center", color=pal.INK, zorder=4)
    L.text(fig, x0 + ylab_w + (w - ylab_w) / 2, y0 - 0.8, f"(63×{ARROW}40×) − (40×{ARROW}63×)", fontsize=FS_LAB, ha="center", va="bottom")
    rend = fig.canvas.get_renderer()                                    # the readout key, on two lines
    for row, items in ((7.4, READOUTS[:2]), (4.6, READOUTS[2:])):
        kx = x0 + 3.0
        for t, lab, _ in items:
            a = L.axes(fig, kx, y0 - row - 1.0, 2.0, 2.0)
            a.plot(0.5, 0.5, marker=MARK[t], ms=2.6, color=pal.INK2, ls="none")
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            L.bare(a)
            tx = L.text(fig, kx + 1.8, y0 - row, lab, fontsize=FS_TICK, va="center")
            kx += 1.8 + tx.get_window_extent(rend).width / fig.dpi * 25.4 + 1.8


def gallery_images(Gal, size=120, ch_size=48):
    """the rendered windows of the gallery: {mag: {id: RGB}}, the queries' single channels, with one gain per microscope"""
    need = {"40x": set(), "63x": set()}
    for dname, (qm, gm) in DIRS.items():
        Gd = Gal[Gal.direction == dname]
        need[qm] |= set(Gd["query"])
        need[gm] |= set(Gd["query"]) | set(Gd[[f"cand{k}" for k in range(1, K + 1)]].values.ravel())
    crops = {m: {c: img.window(c, m) for c in sorted(need[m])} for m in need}
    lim = {m: img.limits(list(crops[m].values()), "zero" if m == "63x" else "percrop") for m in crops}
    ims = {m: {c: img.render(v, lim[m], size) for c, v in crops[m].items()} for m in crops}
    qs = set(Gal["query"])
    chs = {m: {c: {ch: img.render(crops[m][c], lim[m], ch_size, ch) for ch, _, _ in img.CHANNELS} for c in crops[m] if c in qs}
           for m in crops}
    return ims, chs


def _block(fig, x0, y0, qid, dname, Gq, ims, chs, geo, line, letter=None, headers=False, names=False):
    t, g, s, hw = geo["t"], geo["g"], geo["s"], geo["hw"]
    qm, gm = DIRS[dname]
    if letter:
        L.panel_label(fig, letter, x0 - 0.3, y0 - 0.6)
    L.text(fig, x0 + (3.2 if letter else 0.0), y0, line, fontsize=FS_TICK, va="top")
    yb = y0 + HEAD
    q = (s - 3 * 0.2) / 4
    yy = yb
    for m, lab, frame in ((qm, f"{qm.replace('x', '×')} query", None), (gm, f"{gm.replace('x', '×')} match", img.CORRECT)):
        ax = L.axes(fig, x0, yy, s, s)
        img.show(ax, ims[m][qid], frame=frame, lw=1.0)
        if names and frame is None:                                     # the channel names on the query, in their colours
            n, e = ims[m][qid].shape[0], 0.2 / s                               # 0.2 mm in from the edges
            corner = {"DNA": (e, e, "left", "top"), "CDT1": (1 - e, e, "right", "top"),     # one name per corner, so the centre stays clear;
                      "GMNN": (e, 1 - e, "left", "bottom"), "MT": (1 - e, 1 - e, "right", "bottom")}   # a long and a short name on each edge
            for _, lut, nm in img.CHANNELS:
                cx, cy, ha, va = corner[nm]
                ax.text(cx * n, cy * n, nm, color=img.LUT[lut][-1], fontsize=FS_TICK, ha=ha, va=va, zorder=7)
        for k, (ch, _, _) in enumerate(img.CHANNELS):
            ax = L.axes(fig, x0 + k * (q + 0.2), yy + s + 0.2, q, q)
            img.show(ax, chs[m][qid][ch])
        L.text(fig, x0 + s / 2, yy + s + q + 0.3, lab, fontsize=FS_TICK, ha="center", va="top", color=pal.INK2)
        yy += s + q + 2.4
    xc = x0 + s + 0.9
    xr = xc + K * (t + g) + 0.5
    if headers:
        for k in range(K):
            L.text(fig, xc + k * (t + g) + t / 2, yb - 0.2, f"{k + 1}", fontsize=FS_TICK, ha="center", va="bottom", color=pal.INK2)
        L.text(fig, xr + hw, yb - 0.2, "Similarity", fontsize=FS_TICK, ha="right", va="bottom", color=pal.INK2)
    for mi, r in enumerate(data.MODELS):
        rr = Gq[Gq.run == r].iloc[0]
        y = yb + mi * (t + g)
        for k in range(K):
            cid = rr[f"cand{k + 1}"]
            ok = cid == qid
            ax = L.axes(fig, xc + k * (t + g), y, t, t)
            img.show(ax, ims[gm][cid], frame=img.CORRECT if ok else img.WRONG, lw=1.0 if ok else 0.4)
        ax = L.axes(fig, xr, y + 0.3, hw, t - 0.6)
        h = np.asarray(json.loads(rr["hist"]), float)
        ax.stairs(np.log1p(h) / np.log1p(h.max()), np.linspace(rr.sim_lo, rr.sim_hi, len(h) + 1), fill=True, color="#bdbdbd", lw=0, zorder=1)
        ax.axvline(rr.sim_kth, color=pal.INK2, lw=0.4, ls=(0, (1.5, 1.0)), zorder=2)
        ax.axvline(rr.sim_true, color=img.CORRECT if float(rr.rank_true) <= K else img.WRONG, lw=0.9, zorder=3)
        ax.set_xlim(rr.sim_lo, max(rr.sim_hi, rr.sim_true) + 0.02 * (rr.sim_hi - rr.sim_lo))
        ax.set_ylim(0, 1.05)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.3)
        ax.spines["bottom"].set_color(pal.INK2)
        ax.set_xticks([])
        ax.set_yticks([])


def gallery_geometry(w, name_w=10.5, gap=2.2, s=9.8, hw=5.4, g=0.25):
    """tile sizes of the gallery for a width w (4 blocks per row)"""
    bw = (w - name_w - 3 * gap) / 4
    t = (bw - s - 0.9 - 0.5 - hw) / K - g
    q = (s - 3 * 0.2) / 4
    bh = HEAD + max(len(data.MODELS) * (t + g), 2 * (s + q + 2.4))
    return dict(t=t, g=g, s=s, hw=hw, bw=bw, bh=bh, name_w=name_w, gap=gap)


def gallery(fig, x0, y0, w, Gal, ims, chs, letters="fghijklm", row_gap=2.0, title_h=4.0):
    """two rows of four queries: 40x queries (top), 63x queries (bottom), one per outcome; model names at the left"""
    geo = gallery_geometry(w)
    titles = {"ab": "Widefield 40× queries, confocal 63× gallery", "ba": "Confocal 63× queries, widefield 40× gallery"}
    y = y0
    li = iter(letters or "")
    for ri, dname in enumerate(DIRS):
        L.text(fig, x0 + geo["name_w"] + (w - geo["name_w"]) / 2, y, titles[dname], fontsize=FS_LAB, ha="center", va="top")
        y += title_h
        Gd = Gal[Gal.direction == dname]
        picked = Gd.drop_duplicates("query")[["query", "cell_line"]].values.tolist()
        for bi, (qid, line) in enumerate(picked):
            _block(fig, x0 + geo["name_w"] + bi * (geo["bw"] + geo["gap"]), y, qid, dname, Gd[Gd["query"] == qid], ims, chs, geo, line,
                   letter=next(li) if letters else None, headers=True, names=(bi == 0))
        for mi, r in enumerate(data.MODELS):
            L.text(fig, x0, y + HEAD + mi * (geo["t"] + geo["g"]) + geo["t"] / 2, img.wrap(NAME[r]), fontsize=FS_TICK, va="center", ha="left",
                   linespacing=0.9)
        y += geo["bh"] + row_gap
    return y - y0


def main(out_dir=None):
    """each plot on its own, at its size on the Extended Data page"""
    tb = tables()
    out = config.ensure(out_dir or config.figure_dir("ed1"))
    ims, chs = gallery_images(tb["Gal"])
    plots = {"ed1a_images": (66, 32, lambda f: images(f, 1, 1, tb["sample"])),
             "ed1b_retrieval": (118, 36, lambda f: retrieval(f, 1, 7, 116, 20, tb["R"])),
             "ed1c_transfer": (108, 36, lambda f: transfer(f, 1, 4, 22, tb["T"])),
             "ed1d_geometry": (36, 36, lambda f: geometry(f, 1, 7, 34, 18, tb["G"])),
             "ed1e_direction": (40, 36, lambda f: direction(f, 1, 7, 38, 18, tb["D"])),
             "ed1f-m_gallery": (180, 92, lambda f: gallery(f, 0, 1, 180, tb["Gal"], ims, chs, letters=None)),
             "ed1_key": (180, 5, lambda f: model_key(f, 90, 2.5))}
    style.use_agg()
    with style.nature():
        for name, (w, h, draw) in plots.items():
            fig = L.figure(w, height=h, strict=False)
            draw(fig)
            for p in style.check(fig, dpi=300):
                log(f"  {name}: {p}")
            for ext in ("svg", "pdf"):
                fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            log(f"wrote {out / name}.{{svg,pdf}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", help="output directory (default: the ed1 figure directory)")
    main(parser.parse_args().out_dir)
