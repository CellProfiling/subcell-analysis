import os
from glob import glob
import copy

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py
import torch.nn.functional as F

from utils.preprocess import safe_crop


pixel_micron_HPA = 0.0800885
pixel_micron_FUCCI = 0.1625


def collate_fn(batch):
    images, images_orig, rows = zip(*batch)
    batch_images = torch.cat(images, dim=0)
    batch_images_orig = torch.cat(images_orig, dim=0)
    batch_df = pd.concat(rows, ignore_index=True)
    return batch_images, batch_images_orig, batch_df


class HPATestDataset(Dataset):
    def __init__(self, crop_folder, metadata_file, crop_params, preprocess_algo):
        self.crop_folder = crop_folder

        files = sorted(glob(f"{self.crop_folder}/*.png"))

        self.df = pd.read_csv(metadata_file)
        self.df = self.df.drop(columns=[x for x in self.df.columns if "feat" in x])
        self.df = self.df.drop(columns=[x for x in self.df.columns if "prob" in x])

        self.pad = crop_params["pad"]
        self.crop = crop_params["crop"]
        self.resize_to = crop_params["resize_to"]
        self.channels = crop_params["channels"]

        self.preprocess_algo = preprocess_algo

    def __len__(self):
        return len(self.df)

    def load_crop(self, row):
        image = np.stack(
            [
                cv2.imread(f"{self.crop_folder}/{row['id']}masked_{color}.png", -1)
                for color in self.channels
            ],
            axis=-1,
        )

        bbox = row[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].astype(int).values

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

        return (
            img_tensor,
            img_orig_tensor,
            row[["id", "protein", "gene_name", "fov", "z_plane", "treatment"]]
            .to_frame()
            .T,
        )


class AllenCellDataset(Dataset):
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
            .replace("crop_raw/", "crop_subcell/")
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
        else:
            pass

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
    def __init__(self, crop_folder, metadata_file, resize, norm):
        self.crop_folder = (
            crop_folder
            if not resize
            else crop_folder.replace("intermediate", "resized")
        )
        self.df = pd.read_csv(metadata_file, low_memory=False)
        self.resize = resize

        self.norm = norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.crop_folder, row["resized_image_name"])
        img_path = img_path if self.resize else img_path.replace("_resized.png", ".png")

        img_orig = cv2.imread(img_path, -1)[:, :, :2]
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
                    "image_id",
                    "cell_id",
                    "locations",
                    "atlas_name",
                    "ensembl_ids",
                    "gene_names",
                ]
            ]
            .to_frame()
            .T,
        )
