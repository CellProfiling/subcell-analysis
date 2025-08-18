import os
import pandas as pd
import torch
from harmony import harmonize
from utils.feature_analysis import get_per_gene_mean_features

if __name__ == "__main__":
    exp_name_dir_dict = {
        "ViT-ProtS-Pool": "/scratch/groups/emmalu/subcell_ankit/features/ViT-ProtS-Pool/hpa_features/all_features.pth",
        "MAE-CellS-ProtS-Pool": "/scratch/groups/emmalu/subcell_ankit/features/MAE-CellS-ProtS-Pool/hpa_features/all_features.pth",
        "DINO4Cells-HPA": "/scratch/groups/emmalu/subcell_ankit/features/DINO4Cells-HPA/hpa_features/all_features.pth",
        "ViT-Supervised": "/scratch/groups/emmalu/subcell_ankit/features/ViT-Supervised/hpa_features/all_features.pth",
        "bestfitting": "/scratch/groups/emmalu/subcell_ankit/features/bestfitting/hpa_features/all_features.pth",
    }

    harmonize_cats = [
        ["if_plate_id"],
        ["microscope"],
        ["atlas_name"],
        ["if_plate_id", "microscope"],
        ["if_plate_id", "microscope", "atlas_name"],
    ]
    cell_lines = ["U2OS", "all"]

    meta_df = (
        pd.read_csv("annotations/IF-image-orig-bboxes.csv")
        .groupby(["if_plate_id", "position", "sample", "image_width"])
        .first()
        .reset_index()
    )
    meta_df["microscope"] = meta_df["image_width"].apply(
        lambda x: "M1" if x >= 2048 else "M2"
    )

    all_path_mean_file_dict = {}

    for exp_name, dir_path in exp_name_dir_dict.items():
        method_save_folder = f"{os.path.dirname(dir_path)}"

        df, feature_data = torch.load(dir_path, map_location="cpu", weights_only=False)

        for cell_line in cell_lines:
            if os.path.exists(f"{method_save_folder}/mean_feat_{cell_line}.csv"):
                print(
                    f"Mean features for {exp_name} with cell line {cell_line} already exist."
                )
            else:
                feat_dict = get_per_gene_mean_features(
                    df,
                    feature_data,
                    cell_lines=cell_line if cell_line != "all" else None,
                )
                feat_df = pd.DataFrame(feat_dict).T
                feat_df.to_csv(f"{method_save_folder}/mean_feat_{cell_line}.csv")

            all_path_mean_file_dict[f"{exp_name}_{cell_line}"] = (
                f"{method_save_folder}/mean_feat_{cell_line}.csv"
            )

        df1 = pd.merge(
            df,
            meta_df[["if_plate_id", "position", "sample", "microscope"]],
            on=["if_plate_id", "position", "sample"],
            how="left",
        )
        df1["microscope"] = df1["microscope"].fillna("M1")

        if exp_name in ["bestfitting", "ViT-Supervised"]:
            continue

        for harmonize_cat in harmonize_cats:
            if os.path.exists(
                f"{method_save_folder}/harmonized_features_{'_'.join(harmonize_cat)}.pth"
            ):
                print(
                    f"Harmonized features for {exp_name} with categories {harmonize_cat} already exist."
                )
                df, harmonized_features = torch.load(
                    f"{method_save_folder}/harmonized_features_{'_'.join(harmonize_cat)}.pth",
                    map_location="cpu",
                    weights_only=False,
                )
            else:
                print(f"Harmonizing {exp_name} with categories {harmonize_cat}")
                harmonized_features = harmonize(
                    X=feature_data.numpy(),
                    batch_mat=df1,
                    batch_key=harmonize_cat,
                    use_gpu=True,
                    verbose=True,
                    random_state=42,
                )
                harmonized_features = torch.tensor(
                    harmonized_features, dtype=torch.float32
                )

                torch.save(
                    (df, harmonized_features),
                    f"{method_save_folder}/harmonized_features_{'_'.join(harmonize_cat)}.pth",
                )

            for cell_line in cell_lines:
                if os.path.exists(
                    f"{method_save_folder}/mean_feat_{cell_line}_harmonized_{'_'.join(harmonize_cat)}.csv"
                ):
                    print(
                        f"Mean features for {exp_name} with cell line {cell_line} and harmonization {harmonize_cat} already exist."
                    )
                else:
                    feat_dict = get_per_gene_mean_features(
                        df,
                        harmonized_features,
                        cell_lines=cell_line if cell_line != "all" else None,
                    )
                    feat_df = pd.DataFrame(feat_dict).T
                    feat_df.to_csv(
                        f"{method_save_folder}/mean_feat_{cell_line}_harmonized_{'_'.join(harmonize_cat)}.csv",
                    )
                all_path_mean_file_dict[
                    f"{exp_name}_{cell_line}_harmonized_{'_'.join(harmonize_cat)}"
                ] = f"{method_save_folder}/mean_feat_{cell_line}_harmonized_{'_'.join(harmonize_cat)}.csv"

    all_path_mean_file_df = pd.DataFrame.from_dict(
        all_path_mean_file_dict, orient="index", columns=["path"]
    )
    all_path_mean_file_df.index.name = "experiment_cell_line"
    all_path_mean_file_df.to_csv("annotations/mean_features_paths.csv", index=True)

