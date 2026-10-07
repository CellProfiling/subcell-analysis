"""
Loader for cpg0000 JUMP Pilot crops.

Crops are stored as 192x1728 uint8 PNGs with 9 channels stacked horizontally.
The stacking/unstacking logic mirrors JUMP-Pilot-Processing/process_utils.py:
    stack_to_png:  (C, H, W) -> transpose(1,0,2) -> reshape(H, C*W)
    png_to_stack:  (H, C*W) -> reshape(H, C, W) -> transpose(1,0,2) -> (C, H, W)

Channel order (from JUMP-Pilot-Processing configs):
    0: Mito
    1: AGP
    2: RNA
    3: ER
    4: DNA
    5: HighZBF
    6: LowZBF
    7: Brightfield
    8: Mask (cell segmentation mask, appended during processing)
"""

import numpy as np
from PIL import Image

# Channel indices in the stacked crop
CHANNEL_IDX = {
    "Mito": 0,
    "AGP": 1,
    "RNA": 2,
    "ER": 3,
    "DNA": 4,
    "HighZBF": 5,
    "LowZBF": 6,
    "Brightfield": 7,
    "Mask": 8,
}

N_CHANNELS = 9
NATIVE_CROP_SIZE = 192


def png_to_stack(arr: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """
    Unstack a horizontally-concatenated PNG into a channel stack.

    Input:  (H, C*W) e.g. (192, 1728)
    Output: (C, H, W) e.g. (9, 192, 192)

    Mirrors JUMP-Pilot-Processing/process_utils.py::png_to_stack exactly.
    """
    h = arr.shape[0]
    arr = arr.reshape(h, n_channels, h)   # (H, C, W)
    arr = arr.transpose(1, 0, 2)          # (C, H, W)
    return arr


def center_crop(arr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Center-crop a (C, H, W) array to (C, target_size, target_size).
    """
    h, w = arr.shape[-2:]
    top = (h - target_size) // 2
    left = (w - target_size) // 2
    return arr[..., top:top + target_size, left:left + target_size]


def load_crop(path: str, crop_size: int = 128, as_float: bool = True) -> np.ndarray:
    """
    Load a JUMP crop PNG, unstack channels, and center-crop.

    Args:
        path: Path to the stacked PNG (192 x 1728 uint8).
        crop_size: Target spatial size after center-cropping. Default 128.
        as_float: If True, convert to float32 in [0, 1]. If False, keep uint8.

    Returns:
        np.ndarray of shape (9, crop_size, crop_size).
        dtype float32 in [0,1] if as_float else uint8.
    """
    arr = np.array(Image.open(path))                # (192, 1728) uint8
    arr = png_to_stack(arr, n_channels=N_CHANNELS)  # (9, 192, 192)
    if crop_size < NATIVE_CROP_SIZE:
        arr = center_crop(arr, crop_size)           # (9, crop_size, crop_size)
    if as_float:
        arr = arr.astype(np.float32) / 255.0
    return arr


def select_channels(arr: np.ndarray, channels: list[str]) -> np.ndarray:
    """
    Select and reorder channels from a (9, H, W) stack.

    Args:
        arr: (9, H, W) array from load_crop.
        channels: List of channel names, e.g. ["ER", "DNA", "Mito"].

    Returns:
        (len(channels), H, W) array.
    """
    idx = [CHANNEL_IDX[c] for c in channels]
    return arr[idx]


def get_mask(arr: np.ndarray) -> np.ndarray:
    """
    Extract a binary float mask from the Mask channel (idx 8).

    The stored mask is 0/255 uint8 -> 0.0/1.0 after as_float division, but we
    threshold at 0.5 to be robust to interpolation artifacts if the caller ever
    rescales before extracting. Returned as (1, H, W) so it broadcasts cleanly
    against (C, H, W) image stacks.

    Args:
        arr: (9, H, W) array from load_crop.

    Returns:
        (1, H, W) float32 array with values in {0.0, 1.0}.
    """
    return (arr[CHANNEL_IDX["Mask"]] > 0.5).astype(np.float32)[None, :, :]
