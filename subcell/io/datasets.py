"""torch Datasets for the imaging datasets the encoders are run on.

Every Dataset returns ``(model_input, original_image, metadata_row)`` for one
cell, and :func:`collate_fn` concatenates these into a batch. They are used by the
``subcell.inference`` modules, which look the HPA Datasets up by class name from the
``dataset:`` key of the files in ``configs/``.

HPA-style crops (``HPATestDataset``, ``HPATestDataset2``, ``HPATestDataset3``)
share the cropping logic below; they differ in how the crop files are named and in
the metadata table that locates them. Allen Cell, FUCCI, yeast and OpenCell have
one Dataset each.
"""

import copy
import os
import re

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from subcell import config
from subcell.io.preprocess import HPA_COLORS, safe_crop

pixel_micron_HPA = 0.0800885
pixel_micron_FUCCI = 0.1625


def collate_fn(batch):
    images, images_orig, rows = zip(*batch)
    batch_images = torch.cat(images, dim=0)
    batch_images_orig = torch.cat(images_orig, dim=0)
    batch_df = pd.concat(rows, ignore_index=True)
    return batch_images, batch_images_orig, batch_df


class _HPACropDataset(Dataset):
    """Shared window selection, resizing and batching of single-cell HPA crops.

    Subclasses set ``bbox_columns`` (the four metadata columns of the cell's
    bounding box) and ``meta_columns`` (the metadata returned with each cell) and
    implement ``load_crop``, which reads the channels in ``crop_params["channels"]``.
    """

    bbox_columns = ("x1", "y1", "x2", "y2")
    meta_columns = ()

    def __init__(self, crop_folder, crop_params, preprocess_algo):
        self.crop_folder = crop_folder
        self.pad = crop_params["pad"]
        self.crop = crop_params["crop"]
        self.resize_to = crop_params["resize_to"]
        self.channels = crop_params["channels"]
        self.preprocess_algo = preprocess_algo

    def __len__(self):
        return len(self.df)

    def _window(self, image, row):
        """Crop *image* to the cell's box (grown by ``pad`` px; or a fixed
        ``crop`` x ``crop`` window on its centre), then resize to ``resize_to``."""
        bbox = row[list(self.bbox_columns)].astype(int).values

        if self.pad > 0:
            bbox = bbox + np.array([-self.pad, -self.pad, self.pad, self.pad])
        if self.crop > 0:
            crop_center = (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2
            bbox = (
                crop_center[0] - self.crop // 2,
                crop_center[1] - self.crop // 2,
                crop_center[0] + self.crop // 2,
                crop_center[1] + self.crop // 2,
            )

        image, _ = safe_crop(image, bbox)

        if self.resize_to > 0:
            image = cv2.resize(
                image, (self.resize_to, self.resize_to), interpolation=cv2.INTER_AREA
            )

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = self.load_crop(row)
        img_orig_tensor = (
            torch.from_numpy(
                np.concatenate(
                    [image[np.newaxis, ...] for image in img], axis=0
                ).astype(np.float32)
            )
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )
        img_tensor = self.preprocess_algo([img])

        return img_tensor, img_orig_tensor, row[list(self.meta_columns)].to_frame().T


class HPATestDataset(_HPACropDataset):
    """Pre-masked crops of the CM4AI drug-perturbation images.

    One ``<crop_folder>/<id>cell_image.png`` per cell; the metadata CSV carries
    ``bbox_x1..bbox_y2`` and per-cell ``id`` strings written as
    ``cell<N>_crop_...`` (rewritten here to the ``crop_masked_<N>_...`` file stem).
    Columns containing ``feat`` or ``prob`` are dropped.
    """

    bbox_columns = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    meta_columns = ("id", "protein", "gene_name", "fov", "z_plane", "treatment")

    def __init__(self, crop_folder, metadata_file, crop_params, preprocess_algo):
        super().__init__(crop_folder, crop_params, preprocess_algo)

        self.df = pd.read_csv(metadata_file)
        self.df = self.df.drop(columns=[x for x in self.df.columns if "feat" in x])
        self.df = self.df.drop(columns=[x for x in self.df.columns if "prob" in x])

        self.df["id"] = self.df["id"].apply(
            lambda x: re.sub(r"cell(\d+)_crop_", r"crop_masked_\1_", x)
        )

    def load_crop(self, row):
        image = cv2.imread(f"{self.crop_folder}/{row['id']}cell_image.png", -1)
        image = image[..., [HPA_COLORS.index(c) for c in self.channels]]
        return self._window(image, row)


class HPATestDataset2(_HPACropDataset):
    """HPA single-cell crops stored as one multi-channel image per cell.

    ``<crop_folder>/<if_plate_id>/<plate>_<position>_<sample>_<cell_id>_cell_image.png``
    with an optional ``_cell_mask.png`` beside it (``mask_cell``); the metadata
    CSV (see ``subcell/inference/hpa_metadata.py``) carries ``x1..y2`` and the HPA
    annotation columns. ``no_prot`` blanks the protein channel (the last one).
    """

    meta_columns = ("id", "plate_position_sample", "atlas_name", "antibody",
                    "ensembl_ids", "gene_names", "locations")

    def __init__(self, crop_folder, metadata_file, crop_params, preprocess_algo):
        super().__init__(crop_folder, crop_params, preprocess_algo)

        self.df = pd.read_csv(metadata_file, index_col=0)
        self.df["plate_position_sample"] = self.df.apply(
            lambda x: f"{x['if_plate_id']}_{x['position']}_{x['sample']}",
            axis=1,
        )
        self.df["id"] = self.df.apply(
            lambda x: f"{x['if_plate_id']}_{x['position']}_{x['sample']}_{int(x['cell_id'])}",
            axis=1,
        )

        self.mask_cell = crop_params["mask_cell"]
        self.no_prot = crop_params.get("no_prot", False)

    def load_crop(self, row):
        image = cv2.imread(
            f"{self.crop_folder}/{row['if_plate_id']}/{row['id']}_cell_image.png", -1
        )
        if self.no_prot:
            image[:, :, -1] = 0
        image = image[..., [HPA_COLORS.index(c) for c in self.channels]]

        if self.mask_cell:
            cell_mask = cv2.imread(
                f"{self.crop_folder}/{row['if_plate_id']}/{row['id']}_cell_mask.png", -1
            )
            image = image * (cell_mask[:, :, None] > 1).astype(image.dtype)

        return self._window(image, row)


class HPATestDataset3(_HPACropDataset):
    """HPA-style crops stored as one single-channel PNG per channel and cell.

    ``<crop_folder>/<if_plate_id>/<plate_position_sample>_cell<cell_id>_<channel>.png``
    (and ``_cell_mask.png`` for ``mask_cell``), as written for the Kaggle HPA
    single-cell test set (``subcell/inference/kaggle_metadata.py``). The metadata's
    ``id`` and ``cell`` columns are renamed ``plate_position_sample`` and
    ``cell_id``.
    """

    meta_columns = ("plate_position_sample", "cell_id", "atlas_name", "antibody",
                    "ensembl_ids", "gene_names", "locations")

    def __init__(self, crop_folder, metadata_file, crop_params, preprocess_algo):
        super().__init__(crop_folder, crop_params, preprocess_algo)

        self.df = pd.read_csv(metadata_file)
        self.df = self.df.rename(
            columns={"id": "plate_position_sample", "cell": "cell_id"}
        )

        self.mask_cell = crop_params["mask_cell"]

    def load_crop(self, row):
        stem = (f"{self.crop_folder}/{row['if_plate_id']}/"
                f"{row['plate_position_sample']}_cell{row['cell_id']}")
        image = np.stack(
            [cv2.imread(f"{stem}_{color}.png", -1) for color in self.channels],
            axis=-1,
        )
        if self.mask_cell:
            mask = cv2.imread(f"{stem}_cell_mask.png", -1)
            mask = (mask > 0).astype(image.dtype)
            image = image * mask[:, :, np.newaxis]

        return self._window(image, row)


class AllenCellDataset(Dataset):
    """Allen Cell hiPSC projections (nucleus, plasma membrane, structure channels)
    arranged by ``channel_combination`` and min-max normalized per channel or image."""

    def __init__(self, crop_folder, metadata_file, channel_combination, norm):
        self.crop_folder = crop_folder
        self.df = pd.read_csv(metadata_file, low_memory=False)
        self.channel_combination = channel_combination
        self.norm = norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.crop_folder,
            row["crop_raw"]
            .replace("crop_raw/", "")
            .replace(".ome.tif", "_resized_proj.png"),
        )
        img_orig = cv2.imread(img_path, -1)
        img_orig_tensor = (
            torch.from_numpy(img_orig).float().unsqueeze(0).permute(0, 3, 1, 2)
        )

        if self.channel_combination == "concat":
            img = np.stack([img_orig[:, :, [0, 1]], img_orig[:, :, [0, 2]]])
        elif self.channel_combination == "nuc_prot_only":
            img = img_orig[:, :, [0, 2]]
        elif self.channel_combination == "plasma_nuc_prot":
            img = img_orig[:, :, [1, 0, 2]]
        else:
            raise ValueError(f"Unknown channel combination: {self.channel_combination}")

        if self.norm == "perchannel":
            min_max_dims = (0, 1) if len(img.shape) == 3 else (1, 2)
        elif self.norm == "perimage":
            min_max_dims = (0, 1, 2) if len(img.shape) == 3 else (0, 1, 2, 3)
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

        img_min = img.min(min_max_dims, keepdims=True)
        img_max = img.max(min_max_dims, keepdims=True)

        img = np.clip((img - img_min) / (img_max - img_min + 1e-6), 0, 1)

        permute_comb = (0, 3, 1, 2) if len(img.shape) == 3 else (0, 1, 4, 2, 3)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).permute(*permute_comb)

        return (
            img_tensor,
            img_orig_tensor,
            row[["CellId", "structure_name", "cell_stage"]].to_frame().T,
        )


class FUCCIDataset(Dataset):
    """FUCCI U2OS crops (``<ab_id>/<cell_id>.npy``). The data has no nuclear stain,
    so ``channel_combination`` selects the substitute for the DNA channel; with
    ``resize`` the crop is rescaled to the HPA pixel size and cut to 640 px."""

    def __init__(self, crop_folder, metadata_file, channel_combination, resize=False):
        self.crop_folder = crop_folder
        self.df = pd.read_csv(metadata_file)
        self.channel_combination = channel_combination
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.crop_folder, row["ab_id"], f"{row['cell_id']}.npy")
        cell_img = np.load(img_path)

        img = np.zeros((cell_img.shape[0], cell_img.shape[1], 3))
        img[:, :, [0, 2]] = cell_img[:, :, [0, 3]]

        img_orig = cell_img[:, :, [0, 1, 2, 3]]

        if self.channel_combination == "nuc_const":
            img[:, :, 1] = cell_img[:, :, [0, 3]].max() * cell_img[:, :, 4] / 2
        elif self.channel_combination == "nuc_cdt_gmmn":
            img[:, :, 1] = np.mean(cell_img[:, :, [1, 2]], axis=-1) * cell_img[:, :, 4]
        # any other combination leaves the nuclear channel empty

        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)

        if self.resize:
            img_hpa_resize = int(img.shape[0] * pixel_micron_FUCCI / pixel_micron_HPA)
            img = cv2.resize(
                img, (img_hpa_resize, img_hpa_resize), interpolation=cv2.INTER_AREA
            )
            bbox = (
                img_hpa_resize // 2 - 320,
                img_hpa_resize // 2 - 320,
                img_hpa_resize // 2 + 320,
                img_hpa_resize // 2 + 320,
            )
            img, _ = safe_crop(img, bbox)

            img_orig = cv2.resize(
                img_orig,
                (img_hpa_resize, img_hpa_resize),
                interpolation=cv2.INTER_AREA,
            )
            img_orig, _ = safe_crop(img_orig, bbox)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2)
        img_orig_tensor = (
            torch.from_numpy(img_orig).float().unsqueeze(0).permute(0, 3, 1, 2)
        )

        return (
            img_tensor,
            img_orig_tensor,
            row[
                [
                    "ab_id",
                    "cell_id",
                    "GMNN_nu_mean",
                    "CDT1_nu_mean",
                    "pseudotime",
                    "GMM_cc",
                    "GMM_cc_label",
                ]
            ]
            .to_frame()
            .T,
        )


class YeastDataset(Dataset):
    """Yeast localization or cell-cycle images from an HDF5 file; the task is read
    from the file name. Channels are reordered and min-max normalized per image."""

    def __init__(self, crop_file, resize=-1):
        self.data_type = "localization" if "localization" in crop_file else "cellcycle"

        data = h5py.File(crop_file, "r")
        img_data = data["data1"]
        splitByChannel = [
            img_data[:, (chan * 64**2) : ((chan + 1) * 64**2)].reshape((-1, 64, 64, 1))
            for chan in range(8 if self.data_type == "localization" else 5)
        ]
        self.data = np.concatenate(splitByChannel, 3)[:, :, :, [2, 0, 1]]
        self.labels = data["Index1"]

        self.resize = resize

        self.localization_classes = [
            "Actin",
            "Bud",
            "Bud Neck",
            "Bud Periphery",
            "Bud Site",
            "Cell Periphery",
            "Cytoplasm",
            "Cytoplasmic Foci",
            "Eisosomes",
            "Endoplasmic Reticulum",
            "Endosome",
            "Golgi",
            "Lipid Particles",
            "Mitochondria",
            "None",
            "Nuclear Periphery",
            "Nucleolus",
            "Nucleus",
            "Peroxisomes",
            "Punctate Nuclear",
            "Vacuole",
            "Vacuole Periphery",
        ]
        self.cellcycle_classes = [
            "Early G1",
            "Late G1",
            "S/G2",
            "Metaphase",
            "Anaphase",
            "Telophase",
            "Abberent",
            "Over_seg",
            "Anaphase_defect",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.resize > 0:
            img = cv2.resize(
                img.astype(np.float32),
                (self.resize, self.resize),
                interpolation=cv2.INTER_LINEAR,
            )

        img_min = np.min(img, axis=(0, 1), keepdims=True)
        img_max = np.max(img, axis=(0, 1), keepdims=True)

        img = (img - img_min) / (img_max - img_min + 1e-6)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2)

        img_orig_tensor = copy.deepcopy(img_tensor)

        if self.data_type == "localization":
            label_df = pd.DataFrame(
                {
                    "localization": [self.localization_classes[label.argmax()]],
                }
            )
        elif self.data_type == "cellcycle":
            label_df = pd.DataFrame(
                {
                    "cellcycle": [self.cellcycle_classes[label.argmax()]],
                }
            )
        return img_tensor, img_orig_tensor, label_df


class OpenCellDataset(Dataset):
    """OpenCell crops (two channels) listed by ``cell_save_path`` in the metadata."""

    def __init__(self, crop_folder, metadata_file, resize, norm):
        self.crop_folder = crop_folder
        self.df = pd.read_csv(metadata_file, low_memory=False)
        self.resize = resize

        self.norm = norm

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _resolve(path):
        """Locate a crop under ``config.ROOTS["OPENCELL_DATA"]``.

        The metadata stores the path at the machine that wrote it; everything after
        its ``OpenCell/`` component is taken relative to the local data root.
        Relative paths are resolved against the root, other absolute paths are used
        as given.
        """
        rel = path.split("/OpenCell/", 1)[-1] if "/OpenCell/" in path else path
        return os.path.join(config.ROOTS["OPENCELL_DATA"], rel)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self._resolve(row["cell_save_path"])

        img_orig = cv2.cvtColor(cv2.imread(img_path, -1), cv2.COLOR_RGB2BGR)[:, :, :2]
        img_orig_tensor = (
            torch.from_numpy(img_orig).float().unsqueeze(0).permute(0, 3, 1, 2)
        )

        if self.norm == "perchannel":
            min_max_dims = (0, 1)
        elif self.norm == "perimage":
            min_max_dims = (0, 1, 2)
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

        img_min = img_orig.min(min_max_dims, keepdims=True)
        img_max = img_orig.max(min_max_dims, keepdims=True)

        img = np.clip((img_orig - img_min) / (img_max - img_min + 1e-8), 0, 1)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2)

        return (
            img_tensor,
            img_orig_tensor,
            row[
                [
                    "fov_id",
                    "cell_id",
                    "locations",
                    "ensembl_ids",
                    "gene_id",
                ]
            ]
            .to_frame()
            .T,
        )
