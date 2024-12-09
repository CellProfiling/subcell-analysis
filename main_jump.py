import argparse
import os
import shutil
from ast import literal_eval
from glob import glob

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm

from models.vit_inference import ViTPoolClassifier
from utils.jump_utils import (
    elim_corr,
    interpolated_precision_recall_curve,
    match_matrix,
    nn_accuracy,
    normalize_step,
    sim_matrix,
)
from utils.load_model_util import get_dino_model

CHANNEL_DICT = {"nuc": 0, "rna": 1, "er": 2, "agp": 3, "mito": 4}
DINO_CHANNEL_ORDER = ["nuc", "rna", "er", "agp", "mito"]
# DINO_CHANNEL_ORDER = ["nuc", "er", "rna", "agp", "mito"]


def load_model_subcell(config):
    model = ViTPoolClassifier(config=config)
    encoder_path = config.get("encoder_path")
    classifier_paths = config.get("classifier_paths")
    model.load_model_dict(encoder_path, classifier_paths)
    return model


def load_image_tensor(input_path):
    img = cv2.imread(input_path, -1)
    img = np.reshape(img, (img.shape[0], img.shape[0], -1), order="F")
    img = (
        torch.from_numpy(img[:, :, :5].astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    )
    return img


def pc_min_max_standardize(im):
    min_val = torch.amin(im, dim=(2, 3), keepdim=True)
    max_val = torch.amax(im, dim=(2, 3), keepdim=True)

    im = (im - min_val) / (max_val - min_val + 1e-6)
    return im


def min_max_standardize(im):
    min_val = torch.amin(im, dim=(1, 2, 3), keepdim=True)
    max_val = torch.amax(im, dim=(1, 2, 3), keepdim=True)

    im = (im - min_val) / (max_val - min_val + 1e-6)
    return im


def self_normalize(im):
    mean = torch.mean(im, dim=(2, 3), keepdim=True)
    std = torch.std(im, dim=(2, 3), keepdim=True, unbiased=False)
    im = (im - mean) / (std + 1e-7)
    return im


def load_image_dino(input_path):
    img = load_image_tensor(input_path)
    img = img[:, [CHANNEL_DICT[c] for c in DINO_CHANNEL_ORDER], :, :]
    img = pc_min_max_standardize(img)
    img = self_normalize(img)
    return img


def load_image_subcell(input_path, rescale, crop_size=-1, normalize="all_chan"):
    img = load_image_tensor(input_path)

    imgs = []
    for channel in ["mito", "agp", "rna"]:
        i = CHANNEL_DICT[channel]
        imgs.append(img[:, [CHANNEL_DICT["er"], CHANNEL_DICT["nuc"], i], :, :])

    imgs = torch.cat(imgs, dim=0)

    new_size = int(imgs.shape[2] * rescale)
    imgs = F.interpolate(imgs, size=new_size, mode="bilinear", align_corners=False)
    if crop_size > 0:
        imgs = TF.center_crop(imgs, [crop_size, crop_size])

    if normalize == "per_chan":
        imgs = pc_min_max_standardize(imgs)
    elif normalize == "all_chan":
        imgs = min_max_standardize(imgs)
    else:
        imgs = imgs / 255.0
    return imgs


def extract_features(config, image_df, device):
    image_folder = config["image_folder"]

    rescale_ratio = config.get("rescale_ratio")
    crop_size = config.get("crop_size")
    normalize = config.get("normalize")

    with torch.no_grad():
        model_config = config["model_config"]
        if "dino" in config["model_name"]:
            model = get_dino_model(model_config)
        else:
            model = load_model_subcell(model_config)
        model.to(device)

        model.eval()
        for i, row in tqdm(image_df.iterrows(), total=len(image_df)):
            img_path = row.values[0]

            try:
                img_name = os.path.basename(img_path).split(".")[0]
                plate, well, fov, cell = img_name.split("_")
            except:
                img_name = img_path.split("single_cells_dataset/")[1]
                plate, well, fov, cell = img_name.split("/")

            well_save_folder = f"{output_folder}/{plate}/{well}"
            if not os.path.exists(well_save_folder):
                os.makedirs(well_save_folder)

            feat_save_path = f"{well_save_folder}/{fov}_{cell}.pt"
            try:
                torch.load(feat_save_path)
                continue
            except:
                if "dino" in config["model_name"]:
                    imgs = load_image_dino(f"{image_folder}{img_path}")
                else:
                    imgs = load_image_subcell(
                        f"{image_folder}{img_path}", rescale_ratio, crop_size, normalize
                    )

                output = model(imgs.to(device))
                features = output.pool_op.reshape(1, -1)

                torch.save(features, feat_save_path)


def gather_plate_embeddings(output_folder, plate):
    all_features = []
    all_wells = []
    all_fovs = []
    all_cell_ids = []

    all_well_folders = glob(f"{output_folder}/{plate}/*")
    wells = [os.path.basename(w) for w in all_well_folders]

    for well in tqdm(wells, total=len(wells)):
        all_fov_files = glob(f"{output_folder}/{plate}/{well}/*.pt")

        if len(all_fov_files) == 0:
            print(f"No files found for {plate}_{well}")
            continue

        all_fov_features = [torch.load(f, map_location="cpu") for f in all_fov_files]

        all_features.extend(all_fov_features)
        all_wells.extend([well] * len(all_fov_features))
        all_fovs.extend([os.path.basename(f).split("_")[0] for f in all_fov_files])
        all_cell_ids.extend(
            [os.path.basename(f).split("_")[1].split(".")[0] for f in all_fov_files]
        )

    plate_features = torch.cat(all_features, dim=0)
    plate_df = pd.DataFrame(
        {
            "plate": [plate] * plate_features.shape[0],
            "well": all_wells,
            "fov": all_fovs,
            "cell_id": all_cell_ids,
        }
    )
    assert plate_features.shape[0] == plate_df.shape[0]
    torch.save((plate_df, plate_features), f"{output_folder}/{plate}.pth")

    for well_folder in all_well_folders:
        shutil.rmtree(well_folder)
    shutil.rmtree(f"{output_folder}/{plate}/")


def save_plate_features(output_folder):
    all_plate_folders = glob(f"{output_folder}/*")
    all_plate_folders = [p for p in all_plate_folders if os.path.isdir(p)]
    plates = [os.path.basename(p) for p in all_plate_folders]
    for plate in plates:
        gather_plate_embeddings(output_folder, plate)


def get_well_embeddings(plate_paths, agg="mean"):
    plates = [os.path.basename(p).split(".")[0] for p in plate_paths]

    all_plates = []
    all_wells = []
    all_fovs = []
    all_features = []

    for i, plate in tqdm(enumerate(plates), total=len(plates)):
        plate_df, plate_features = torch.load(plate_paths[i], map_location="cpu")
        well_fov_groups = plate_df.groupby(["well", "fov"]).groups
        for well_fov, idxs in tqdm(well_fov_groups.items(), total=len(well_fov_groups)):
            well = well_fov[0]
            fov = well_fov[1]
            fov_features = plate_features[idxs, :]
            if agg == "mean":
                fov_agg_feature = torch.mean(fov_features, dim=0, keepdim=True)
            elif agg == "median":
                fov_agg_feature = torch.median(fov_features, dim=0, keepdim=True)[0]
            all_plates.append(plate)
            all_wells.append(well)
            all_fovs.append(fov)
            all_features.append(fov_agg_feature)

    all_features = torch.cat(all_features, dim=0)
    features_df = pd.DataFrame(
        {
            "plate": all_plates,
            "well": all_wells,
            "fov": all_fovs,
        }
    )
    feat_cols = [f"feature_{i}" for i in range(all_features.shape[1])]
    features_df.loc[:, feat_cols] = all_features.numpy()
    features_df = features_df.groupby(["plate", "well"])[feat_cols].mean().reset_index()
    features_df.to_csv(f"{output_folder}/all_features_agg_{agg}.csv", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("-c", "--config", help="path to configuration file")

    args = argparser.parse_args(["-c", "configs/config_jump.yaml"])

    config = yaml.safe_load(open(args.config))

    n_cores = 20

    image_df = pd.read_csv(config["csv_file"], header=None)
    split_dfs = np.array_split(image_df, n_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rescale_ratio = config.get("rescale_ratio")
    normalize = config.get("normalize")

    output_folder = f"{config['output_folder']}/{config['model_name']}_rescale_{rescale_ratio}_normalize_{normalize}"
    # os.makedirs(output_folder, exist_ok=True)

    agg = config["agg"]

    if not os.path.exists(f"{output_folder}/all_features_agg_{agg}.csv"):
        plate_paths = glob(f"{output_folder}/*.pth")
        if len(plate_paths) == 7:
            print("All plate features already saved, Aggregating well features")
        else:
            print("Extracting features")
            joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(extract_features)(config, split_df, device)
                for split_df in split_dfs
            )
            print("All features extracted, ")
            save_plate_features(output_folder)
            plate_paths = glob(f"{output_folder}/*.pth")

        print("All plate features saved, Aggregating well features")
        get_well_embeddings(plate_paths, agg)
        print("All well features saved")

    print("Features already extracted, skipping")
    df = pd.read_csv(f"{output_folder}/all_features_agg_{agg}.csv")
    df = df[df["plate"] != "BR00116995"].reset_index(drop=True)

    na_wells = df[df.isna().any(axis=1)].index
    if len(na_wells) > 0:
        print(f"Removing {len(na_wells)} wells with missing features")
        df = df.drop(na_wells).reset_index(drop=True)

    feature_cols = [c for c in df.columns if "feature" in c]

    metadata = pd.read_csv("annotations/jump_metadata.csv")
    metadata_cols = ["plate", "well", "compound", "usable moas"]
    metadata = metadata[metadata_cols]

    df = df.merge(metadata, on=["plate", "well"], how="left")

    # Define all normalization combinations
    spherize_method = ["PCA", "PCA-cor", "ZCA", "ZCA-cor"]
    standardize_methods = ["mad_robustize", "standardize"]
    sphere_then_stand = [(i, j) for i in spherize_method for j in standardize_methods]
    stand_then_sphere = [(i, j) for i in standardize_methods for j in spherize_method]
    just_stand = [(i, None) for i in standardize_methods]
    steps = sphere_then_stand + stand_then_sphere + just_stand

    feature_select_options = [False, True]

    results = []
    for reduce_features in feature_select_options:  # False needs to come first!!
        print(f"Number of features {len(feature_cols)}")

        if reduce_features:
            df1, feature_cols1 = elim_corr(df, metadata_cols, feature_cols)
            print(f"Number of features after feature-select: {len(feature_cols1)}")
        else:
            df1 = df
            feature_cols1 = feature_cols

        for step1, step2 in steps:
            print(f"Running {step1} then {step2}")

            df2, feature_cols2 = normalize_step(
                df1, metadata_cols, feature_cols1, method=step1
            )
            df3, feature_cols3 = normalize_step(
                df2, metadata_cols, feature_cols2, method=step2
            )

            # Compound Similarity matrices
            rep_agg_df = (
                df3.groupby(["plate", "compound", "usable moas"], dropna=False)[
                    feature_cols3
                ]
                .mean()
                .reset_index()
            )  # aggregate replicates per batch
            similarities = sim_matrix(rep_agg_df, feature_cols3)
            comp_match_matrix = match_matrix(rep_agg_df, "compound")
            plate_block_matrix = ~match_matrix(rep_agg_df, "plate")

            # MoA Similarity matrices
            consensus_df = (
                rep_agg_df.dropna()
                .groupby(["compound", "usable moas"])[feature_cols3]
                .mean()
                .reset_index()
            )
            consensus_df.loc[:, "usable moas"] = consensus_df["usable moas"].apply(
                literal_eval
            )
            moa_similarities = sim_matrix(consensus_df, feature_cols3)
            moa_match_matrix = match_matrix(consensus_df, "usable moas")

            # Interpolated Mean Average Precision
            __, comp_average_precision = interpolated_precision_recall_curve(
                comp_match_matrix, similarities
            )
            compound_map = np.mean(comp_average_precision)
            __, moa_average_precision = interpolated_precision_recall_curve(
                moa_match_matrix, moa_similarities
            )
            moa_map = np.mean(moa_average_precision)

            # Nearest Neighbor Accuracy
            compound_accuracy = nn_accuracy(
                comp_match_matrix, similarities, plate_block_matrix
            )
            moa_accuracy = nn_accuracy(moa_match_matrix, moa_similarities)

            results.append(
                [
                    agg,
                    reduce_features,
                    step1,
                    step2,
                    compound_map,
                    moa_map,
                    compound_accuracy,
                    moa_accuracy,
                ]
            )
            print(
                f"Compound MAP: {compound_map:.4f}, MoA MAP: {moa_map:.4f}, Compound NN Accuracy: {compound_accuracy:.4f}, MoA NN Accuracy: {moa_accuracy:.4f}"
            )
    results_df = pd.DataFrame(
        results,
        columns=[
            "fov_agg",
            "feature_select",
            "normalize step1",
            "normalize step2",
            "Compound mAP",
            "MoA mAP",
            "Compound NN-acc",
            "MoA NN-acc",
        ],
    )
    results_df.to_csv(f"{output_folder}/all_results_{agg}_results.csv", index=False)
    x = 1
