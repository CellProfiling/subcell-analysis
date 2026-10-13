"""Cropping and per-model input normalization of single-cell images.

``safe_crop`` cuts a window, possibly extending past the image, from an image; the
``preprocess_input_*`` functions turn a list of ``H x W x C`` uint8/uint16 crops
into the float tensor each encoder expects: ``bestfitting`` (divide by the dtype
maximum), ``dino`` (min-max, then the DINO4Cells channel mean/std), ``self_norm``
(per-channel max, then z-score) and ``subcell`` (min-max over the image, or per
channel). Used by the ``subcell.inference`` and ``subcell.classification`` modules.
"""
import numpy as np
import torch

HPA_COLORS = ["red", "yellow", "blue", "green"]

DINO_MEAN = torch.tensor([0.1450534, 0.11360057, 0.1231717, 0.14919987])
DINO_STD = torch.tensor([0.18122554, 0.14004277, 0.18840286, 0.17790672])


def safe_crop(image, bbox):
    """Window ``bbox = (x1, y1, x2, y2)`` of *image* (``x`` along axis 0), zero-padded
    where it extends past the image. Returns the patch and the clipped box."""
    x1, y1, x2, y2 = bbox
    img_w, img_h = image.shape[:2]
    is_single_channel = len(image.shape) == 2
    if x1 < 0:
        pad_x1 = 0 - x1
        new_x1 = 0
    else:
        pad_x1 = 0
        new_x1 = x1
    if y1 < 0:
        pad_y1 = 0 - y1
        new_y1 = 0
    else:
        pad_y1 = 0
        new_y1 = y1
    if x2 > img_w:
        pad_x2 = x2 - img_w
        new_x2 = img_w
    else:
        pad_x2 = 0
        new_x2 = x2
    if y2 > img_h:
        pad_y2 = y2 - img_h
        new_y2 = img_h
    else:
        pad_y2 = 0
        new_y2 = y2

    patch = image[new_x1:new_x2, new_y1:new_y2]
    patch = (
        np.pad(
            patch,
            ((pad_x1, pad_x2), (pad_y1, pad_y2)),
            mode="constant",
            constant_values=0,
        )
        if is_single_channel
        else np.pad(
            patch,
            ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    )
    return patch, (new_x1, new_y1, new_x2, new_y2)


def preprocess_input_bestfitting(images):
    images_np = np.concatenate([image[np.newaxis, ...] for image in images], axis=0)

    images_np = images_np / np.iinfo(images_np.dtype).max
    image_tensors = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
    return image_tensors


def preprocess_input_dino(images):
    images_np = np.concatenate([image[np.newaxis, ...] for image in images], axis=0)

    img_min = np.min(images_np, axis=(1, 2, 3), keepdims=True)
    img_max = np.max(images_np, axis=(1, 2, 3), keepdims=True)
    images_np = (images_np - img_min) / (img_max - img_min + 1e-8)
    images_np = np.clip(images_np, 0, 1)

    image_tensors = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
    image_tensors = (image_tensors - DINO_MEAN.view(1, 4, 1, 1)) / DINO_STD.view(
        1, 4, 1, 1
    )
    return image_tensors


def preprocess_input_self_norm(images, to_8bit=True):
    images_np = np.concatenate([image[np.newaxis, ...] for image in images], axis=0)
    images_np = images_np.astype(np.float32)

    images_np = images_np / np.maximum(images_np.max(axis=(1, 2), keepdims=True), 1)
    if to_8bit:
        images_np = np.floor(images_np * 255) / 255
    img_mean = images_np.mean(axis=(1, 2), keepdims=True)
    img_std = images_np.std(axis=(1, 2), keepdims=True)
    images_np = (images_np - img_mean) / (img_std + 1e-7)

    image_tensors = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
    return image_tensors


def preprocess_input_subcell(images, per_channel=False):
    images_np = np.concatenate([image[np.newaxis, ...] for image in images], axis=0)

    dims = (1, 2) if per_channel else (1, 2, 3)

    img_min = np.min(images_np, axis=dims, keepdims=True)
    img_max = np.max(images_np, axis=dims, keepdims=True)
    images_np = (images_np - img_min) / (img_max - img_min + 1e-8)
    images_np = np.clip(images_np, 0, 1)

    image_tensors = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
    return image_tensors
