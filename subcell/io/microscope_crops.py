"""Single-cell crops of the widefield / confocal benchmark (Supplementary Note 1, Extended Data Fig. 1).

An image pipeline outside this repository (shading correction, projection, stitching, segmentation, matching across
microscopes) writes, per cell and microscope, the raw uint16 crops of one physical window (``MICROSCOPE_DATA``):

    <crops>/<mag>/<well>/<id><channel>.png       channels red (microtubules), blue (DAPI), gmnn, cdt1
    <crops>/crops_index.parquet                   one row per id x magnification; the same id is the same cell

The 63x crop is 1,024 px at 0.07 um (71.68 um). The 40x crop (448 px at 0.1616971666 um, 72.44 um) is zoomed about
its centre by 448 x 0.1616971666 / (1024 x 0.07) = 1.010607 and rotated by its 63x field's registered rotation
(``align``: id, zoom, rotation_deg), so that it spans the 63x window. No mask is applied: a mask derived from either
microscope would differ between the two images, and a shared mask would itself identify the cell.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CHANNELS = ("red", "blue", "gmnn", "cdt1")


def align_matrix(shape, zoom, rotation_deg):
    """similarity about the crop centre (cv2 convention), as applied to every 40x plane"""
    h, w = shape[:2]
    return cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), float(rotation_deg), float(zoom))


class MicroscopeCropDataset(Dataset):
    """One item = one cell at one magnification: for every protein in ``crop_params["proteins"]`` an image stacked in
    ``crop_params["channels"]`` order ("protein" = that protein; channels in ``blank_channels`` are zeros, e.g.
    yellow = ER, not stained), aligned (``crop_params["align"]`` = {mag: parquet}), resized to ``resize_to`` px.
    The images of one cell are consecutive in the batch (``collate_dict``).

    ``preprocess`` is either a dict {input name: (channel indices, [functions])} -- several models in one pass; the
    item is (dict of tensors, metadata row) -- or one function of the list of images, whose result is returned with
    the metadata row."""

    def __init__(self, crop_folder, index_file, mag, crop_params, preprocess, ids=None):
        self.crop_folder = crop_folder
        df = pd.read_parquet(index_file)
        df = df[df["mag"] == mag]
        if ids is not None:
            df = df[df["id"].isin(ids)]
        self.df = df.sort_values("id").reset_index(drop=True)
        self.mag = mag
        self.proteins = crop_params["proteins"]
        self.channels = crop_params["channels"]
        self.blank = set(crop_params.get("blank_channels", []))
        self.resize_to = crop_params["resize_to"]
        self.meta_cols = [c for c in crop_params.get("meta_cols", ["id", "wtag", "mag"]) if c in self.df.columns]
        align = crop_params.get("align", {}).get(mag)
        self.align = pd.read_parquet(align).set_index("id")[["zoom", "rotation_deg"]] if align else None
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def load(self, row, name):
        return cv2.imread(f"{self.crop_folder}/{self.mag}/{row['wtag']}/{row['id']}{name}.png", cv2.IMREAD_UNCHANGED)

    def images(self, row):
        """the model inputs of one cell before normalisation: one float32 (resize_to, resize_to, C) per protein"""
        planes = {c: self.load(row, c) for c in self.channels if c != "protein" and c not in self.blank}
        out, M = [], None
        for protein in self.proteins:
            planes["protein"] = self.load(row, protein)
            img = np.stack([np.zeros_like(planes["protein"]) if c in self.blank else planes[c] for c in self.channels],
                           axis=-1).astype(np.float32)
            if self.align is not None:
                if M is None:
                    M = align_matrix(img.shape, *self.align.loc[row["id"], ["zoom", "rotation_deg"]])
                h, w = img.shape[:2]
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
                img = img if img.ndim == 3 else img[..., None]
            if img.shape[0] != self.resize_to:
                img = cv2.resize(img, (self.resize_to, self.resize_to),
                                 interpolation=cv2.INTER_AREA if self.resize_to < img.shape[0] else cv2.INTER_LINEAR)
                img = img if img.ndim == 3 else img[..., None]
            out.append(img)
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = self.images(row)
        meta = row[self.meta_cols].to_frame().T
        if isinstance(self.preprocess, dict):
            return {k: torch.cat([f([im[..., ch] for im in images]) for f in fns], dim=0)
                    for k, (ch, fns) in self.preprocess.items()}, meta
        return self.preprocess(images), meta


def collate_dict(batch):
    """batch of (dict of tensors, metadata) -> (dict of tensors concatenated along dim 0, metadata DataFrame)"""
    return ({k: torch.cat([b[0][k] for b in batch], dim=0) for k in batch[0][0]},
            pd.concat([b[1] for b in batch], ignore_index=True))


def load_window(crop_folder, cid, mag, align=None, channels=CHANNELS):
    """the raw planes of one cell's window at one magnification (float32, native pixels, before the resize to the
    model size), aligned as the model input; ``pad`` marks pixels outside the acquired field (zero in every channel)"""
    well = cid.rsplit("_cell", 1)[0]
    d = {}
    for ch in channels:
        v = cv2.imread(f"{crop_folder}/{mag}/{well}/{cid}{ch}.png", cv2.IMREAD_UNCHANGED)
        assert v is not None, (cid, mag, ch)
        d[ch] = v.astype(np.float32)
    if align is not None:
        M = align_matrix(d[channels[0]].shape, *align.loc[cid, ["zoom", "rotation_deg"]])
        h, w = d[channels[0]].shape
        d = {ch: cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR) for ch, x in d.items()}
    d["pad"] = np.all([d[ch] == 0 for ch in channels], axis=0)
    return d
