import os
import re


import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn import decomposition
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import colorcet as cc
from scipy.signal import correlate2d
from scipy.stats import pearsonr

cmap = cm.nipy_spectral

if __name__ == "__main__":

    exp_name_dir_dict = {
        "ViT-ProtS-Pool": "rybg_448_nc8_vitb16_pool_fp32_supcon",
        "MAE-CellS-ProtS-Pool": "rybg_448_nc8_contrast_mae_pool_vitb16_fp32_mr0.25_objmr0.0_supcon",
    }

    exp_folder = "/proj/hpa_subcell/allhpa_ablations_results/opencell"

    for exp_name, exp_dir in exp_name_dir_dict.items():
        res_df = pd.read_csv(os.path.join(exp_folder, exp_dir, "results.csv"))