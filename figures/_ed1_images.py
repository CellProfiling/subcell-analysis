"""Display of the cross-microscope benchmark's cell windows (Extended Data Fig. 1a and its gallery, f-m).

The images are the models' inputs before the resize: the 63x crop (1,024 px, 71.68 um) and the 40x crop aligned onto
the same window, no mask. Additive LUT composite and single channels: DNA blue, microtubules red, GMNN green, CDT1
orange (light blue, the usual FUCCI colour for GMNN, would vanish on the DNA blue). Display rule, linear: background
none at 63x (photon-counting confocal detector, background ~0: a per-image percentile would remove real signal in
confluent fields); at 40x each image's own 1st percentile per channel (the widefield camera offset and out-of-focus
light vary between fields). Gain: one per microscope and channel for the whole figure, the 99.8th percentile of the
pooled background-subtracted pixels. Frames: the match thick teal, other cells thin dark red (the figure palette's
colour-blind-checked pair)."""

import cv2
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from subcell import config, style
from subcell.io.microscope_crops import load_window
from subcell.style import layout as L, luts

CHANNELS = [("blue", "blue", "DNA"), ("red", "red", "MT"), ("gmnn", "green", "GMNN"), ("cdt1", "orange", "CDT1")]
LUT = {lut: luts.load_lut(lut).astype(np.float32) / 255.0 for _, lut, _ in CHANNELS}
KEY_NAME = {"blue": "DNA", "red": "Microtubules (MT)", "green": "GMNN", "orange": "CDT1"}
CORRECT, WRONG = "#2a9d8f", "#b2182b"
WINDOW_UM = 71.68
_ALIGN = []


def window(cid, mag):
    """one cell's window at one magnification (40x aligned onto the 63x window)"""
    if not _ALIGN:
        _ALIGN.append(pd.read_parquet(config.MICROSCOPE_ALIGN).set_index("id"))
    return load_window(str(config.MICROSCOPE_CROPS), cid, mag, _ALIGN[0] if mag == "40x" else None,
                       channels=tuple(ch for ch, _, _ in CHANNELS))


def background(c):
    return {ch: float(np.percentile(c[ch][~c["pad"]], 1)) for ch, _, _ in CHANNELS}


def limits(crops, mode):
    """gains of one microscope over the figure's crops; mode "zero" (63x) or "percrop" (40x) background"""
    bgs = [({ch: 0.0 for ch, _, _ in CHANNELS} if mode == "zero" else background(c)) for c in crops]
    lim = {ch: float(np.percentile(np.concatenate([(c[ch][~c["pad"]] - bg[ch]).ravel() for c, bg in zip(crops, bgs)]), 99.8))
           for ch, _, _ in CHANNELS}
    lim["_mode"] = mode
    return lim


def render(c, lim, size=160, only=None, grey=False):
    """additive LUT composite (only=None) or one channel, in its LUT colour or in greyscale (grey=True: readable
    without colour vision; the channel is identified by its position)"""
    rgb = np.zeros(c["blue"].shape + (3,), np.float32)
    bg = {ch: 0.0 for ch, _, _ in CHANNELS} if lim.get("_mode") == "zero" else background(c)
    for ch, lut, _ in CHANNELS:
        if only is None or ch == only:
            x = np.clip((c[ch] - bg[ch]) / max(lim[ch], 1e-6), 0, 1) * ~c["pad"]
            rgb += np.repeat(x[..., None], 3, -1) if grey else LUT[lut][np.round(x * 255).astype(np.int32)]
    return np.clip(cv2.resize(np.clip(rgb, 0, 1), (size, size), interpolation=cv2.INTER_AREA), 0, 1)


def show(ax, img, frame=None, lw=0.9):
    ax.imshow(img, interpolation="none", extent=(0, img.shape[1], img.shape[0], 0))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    L.bare(ax)
    if frame:
        ax.add_patch(Rectangle((0, 0), img.shape[1], img.shape[0], fill=False, edgecolor=frame, lw=lw, clip_on=False, zorder=9,
                               joinstyle="miter"))


def channel_key(fig, x, y, fontsize=None, names=KEY_NAME, gap=2.2):
    """channel names in their LUT colours; returns the x after the last name"""
    rend = fig.canvas.get_renderer()
    for _, lut, _ in CHANNELS:
        tx = L.text(fig, x, y, names[lut], fontsize=fontsize or style.FS_TINY, va="top", color=LUT[lut][-1] * 0.85, fontweight="bold")
        x += tx.get_window_extent(rend).width / fig.dpi * 25.4 + gap
    return x


def wrap(label, n=11):
    """long model names over two lines at the last hyphen"""
    return label if len(label) <= n else label[:label.rindex("-") + 1] + "\n" + label[label.rindex("-") + 1:]
