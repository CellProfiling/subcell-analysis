import cv2
import numpy as np
import torch

HPA_COLORS = ["red", "yellow", "blue", "green"]

DINO_MEAN = torch.tensor([0.1450534, 0.11360057, 0.1231717, 0.14919987])
DINO_STD = torch.tensor([0.18122554, 0.14004277, 0.18840286, 0.17790672])


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


def load_crops(
    bboxes_df,
    pad=-1,
    crop=-1,
    resize_to=-1,
    channels=["red", "green", "blue", "yellow"],
):
    processed_crops = []
    for _, bbox_row in bboxes_df.iterrows():
        cell_img = cv2.imread(bbox_row["cell_path"], -1)
        if pad > 0:
            bbox = bbox_row[["x1", "y1", "x2", "y2"]].values + np.array(
                [-pad, -pad, pad, pad]
            )
        if crop > 0:
            bbox = bbox_row[["x1", "y1", "x2", "y2"]].values
            crop_center = (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2
            bbox = (
                crop_center[0] - crop // 2,
                crop_center[1] - crop // 2,
                crop_center[0] + crop // 2,
                crop_center[1] + crop // 2,
            )
        else:
            bbox = bbox_row[["x1", "y1", "x2", "y2"]].values
        cell_img, _ = safe_crop(cell_img, bbox)

        if resize_to > 0:
            cell_img = cv2.resize(
                cell_img, (resize_to, resize_to), interpolation=cv2.INTER_AREA
            )

        cell_img = cell_img[..., [HPA_COLORS.index(c) for c in channels]]

        processed_crops.append(cell_img)
    return processed_crops


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


def preprocess_input_subcell(images, per_channel=False):
    images_np = np.concatenate([image[np.newaxis, ...] for image in images], axis=0)

    dims = (1, 2) if per_channel else (1, 2, 3)

    img_min = np.min(images_np, axis=dims, keepdims=True)
    img_max = np.max(images_np, axis=dims, keepdims=True)
    images_np = (images_np - img_min) / (img_max - img_min + 1e-8)
    images_np = np.clip(images_np, 0, 1)

    image_tensors = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
    return image_tensors
