import torch
from preprocessing.img import openImage
import pandas as pd
import os
import numpy as np
import pickle as pkl

class JUMP_Dataset(torch.utils.data.Dataset):
    

    def __init__(self, csv, outline=False):
        if not isinstance(csv, pd.DataFrame): #path to metadara
            self.metadata=pd.read_csv(csv)
        else: # csv=DataFrame
            self.metadata = csv 
        self.include_outline = outline


    def  __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        channels = ["Mito", "AGP", "NucRNA", "ER", "DAPI"]
        paths = [self.metadata.iloc[idx][chan] for chan in channels]
       
        #GET OUTLINES
        outlines=None
        if self.include_outline:
            assert "Nuc Outline" in self.metadata
            assert "Cell Outline" in self.metadata
            outlines = [self.metadata.iloc[idx]["Cell Outline"], self.metadata.iloc[idx]["Nuc Outline"]]
        
        #GET IMAGE
        image = openImage(paths, outlines)

        #GET ILLUMINATION CORRECTION
        if "Illum" in self.metadata.columns:
            illum_corr = np.load(self.metadata.iloc[idx]["Illum"])
        else:
            illum_corr = np.ones(image.shape[0:2])


        #GET ILLUMINATION STATS
        if "Illum Stats" in self.metadata.columns:
            with open(self.metadata.iloc[idx]["Illum Stats"], "rb") as f:
                illum_stats = pkl.load(f)
        else:
            illum_stats = None
        
        #SAMPLE IDENTIFICATION INFO
        plate = self.metadata.iloc[idx]["Plate"]
        well = self.metadata.iloc[idx]["Well"]
        fov = self.metadata.iloc[idx]["FOV"]
        name =  f"{plate}_{well}_{fov}"
        if "Source" in self.metadata.columns:
            source = self.metadata.iloc[idx]["Source"]
            name = f"{source}_{name}"

        sample = {"image": image, "illum_corr": illum_corr, "illum_stats": illum_stats, "name": name}

        return sample

        
