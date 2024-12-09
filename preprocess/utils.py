import numpy as np
import pandas as pd
from skimage.measure import regionprops
from scipy.ndimage.morphology import grey_dilation


def safe_crop(image, bbox):
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
    if x2 > img_w - 1:
        pad_x2 = x2 - (img_w - 1)
        new_x2 = img_w - 1
    else:
        pad_x2 = 0
        new_x2 = x2
    if y2 > img_h - 1:
        pad_y2 = y2 - (img_h - 1)
        new_y2 = img_h - 1
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


def create_crops(image, cell_mask, crop_size=1024, mask_cell=True):
    regions = regionprops(cell_mask)

    all_crops = {}
    all_masks = {}
    cell_bboxes = []
    for region in regions:
        image_cp = image.copy()
        if mask_cell:
            # image_cp[cell_mask != region.label] = 0
            this_cell_mask = cell_mask == region.label
            this_cell_mask = grey_dilation(this_cell_mask, size=7)
            image_cp[this_cell_mask == 0] = 0

        bbox = region.bbox
        bbox_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        fixed_bbox = (
            bbox_center[0] - crop_size // 2,
            bbox_center[1] - crop_size // 2,
            bbox_center[0] + crop_size // 2,
            bbox_center[1] + crop_size // 2,
        )
        cell_crop, _ = safe_crop(image_cp, fixed_bbox)
        all_crops[region.label] = cell_crop

        cell_mask_crop, _ = safe_crop(cell_mask == region.label, fixed_bbox)
        all_masks[region.label] = cell_mask_crop

        new_center = (crop_size // 2, crop_size // 2)
        new_bbox = (
            new_center[0] - w // 2,
            new_center[1] - h // 2,
            new_center[0] + w // 2,
            new_center[1] + h // 2,
        )
        cell_bboxes.append(
            {
                "cell_id": region.label,
                "x1": new_bbox[0],
                "y1": new_bbox[1],
                "x2": new_bbox[2],
                "y2": new_bbox[3],
            }
        )

    cell_bbox_df = pd.DataFrame(cell_bboxes)
    return all_crops, all_masks, cell_bbox_df
