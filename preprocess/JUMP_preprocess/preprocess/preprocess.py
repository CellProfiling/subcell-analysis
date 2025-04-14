import click
import numpy as np
import pandas as pd
from preprocessing.preprocess_utils import *
from preprocessing.img import saveImage
from preprocessing.dataset import JUMP_Dataset


def preprocess(metadata, crop_size, save_dir, save_crops=True, center_type="cell", reduce=True):
    '''Purpose: find crops of cells with area at most 25% padding
       Input: path to metadata, side length of crop, directory to save crops to...
       Ouput: list of crop names, list of centers, optional save crops as pngs
    '''

    dataset = JUMP_Dataset(metadata, outline=True)

    names=[]
    all_centers=[]
    savenames = []

    for sample in dataset:
        img = sample["image"]
        illum_stats = sample["illum_stats"]
        illum_corr = illum_stats["illum_correction_function"]
        
        #ILLUMINATION CORRECTION (switched to Broad)
        #img[:, :, 0:5] = img[:, :, 0:5]/illum_corr
        plate = sample["name"].split("_")[0]
        broad_illum_corr = []
        for ch in ["Mito", "AGP", "RNA", "ER", "DNA"]:
            broad_illum_corr.append(np.load(f"{illum_corr_path}/{plate}_Illum{ch}.npy"))
        broad_illum_corr = np.stack(broad_illum_corr)
        broad_illum_corr = np.transpose(broad_illum_corr, (1,2,0))
        img[:, :, 0:5] = img[:, :, 0:5]/broad_illum_corr
        
        
        #NORMALIZE each channel to be in (0,1) range
        #This version is same as Broad's DeepProfiler:
        plate_upper = illum_stats["upper_percentiles"] #99.99% of plate
        plate_lower = illum_stats["lower_percentiles"] #0.01% of plate
        fov_lower, fov_upper = np.percentile(img[:, :, 0:5], [0.05, 99.95], axis=(0,1))
        upper = np.min(np.stack([plate_upper, fov_upper]), axis=0)
        img[:, :, 0:5] = ((img[:, :, 0:5] - fov_lower)/upper).clip(0,1)

        #PAD:
        pad_len = int(crop_size * 0.25) #will ensure that crops are no more than 25% padding
        img = np.transpose(img, (2,0,1))
        img = pad(img, pad_len)
        img = np.transpose(img, (1,2,0))

        try:
            #CROP
            cell_mask = img[:,:,-1].astype(np.int16)
            crops, centers = get_sc_crops(img, cell_mask, crop_size, center=center_type, reduce=reduce)
            if crops is not None and centers is not None:
                crops = (crops*255).astype(np.uint8)
                crops = unfold_stack(crops)

                assert len(crops) == len(centers)

                #SAVE CROPS and GET NAMES
                for i in range(len(centers)):
                    name = sample["name"]
                    savename = f"{save_dir}/{name}_{i+1}.png"
                    savenames.append(savename)
                    names.append(name.split("_"))
                    if save_crops: saveImage(crops[i], savename)
                    
                all_centers.append(centers)
            else:
                print("No crops after filtering centers: " + sample["name"])
        
        except Exception as e: 
            print("Problem with " + sample["name"])
            print(e)

    all_centers = np.concatenate(all_centers) - pad_len

    df = pd.DataFrame(np.concatenate([all_centers, names, np.array([savenames]).T], axis=1), columns = ["x", "y", "Plate", "Well", "FOV", "img_path"])
    return df


    
'''

nuc_paths = glob.glob("/scratch/groups/emmalu/JUMP/raw/*/outlines/*nuclei_outlines.png")

nuc_paths = random.sample(nuc_paths, 10 )

for nuc_path in nuc_paths:
    cell_path = nuc_path.replace("nuclei", "cell")
    if os.path.isfile(cell_path): #check that cell path exists
        nuc_outline = imread(nuc_path)
        cell_outline = imread(cell_path)

        if np.any(nuc_outline) and np.any(cell_outline): #make sure outlines are not empty
            well, fov = os.path.basename(nuc_path).split("-")[0].split("_")
            well = letter_to_rc(well)
            fov = fov.split("s")[1]
            fov = add_leading_zero(fov)

            path_head = nuc_path.split("/outlines")[0]
            chan_paths = [f"{path_head}/images/{well}f{fov}p01-ch{i}sk1fk1fl1.tiff" for i in range(1,6)]
            if all([os.path.isfile(path) for path in chan_paths]): #makes sure all channels exist
            
                img = openImg(img, outlines=[cell_outline, nuc_outline])
                img = min_max_norm(img)
                #img = illumCorrection(img)
                #get mask
                cell_mask = make_masks(cell_outline, nuc_outline)
                
                #get image + normalize
                img = np.stack([imread(path) for path in chan_paths]) #stack channels
                min = np.percentile(img, 0, axis=(1,2)).reshape((5,1,1))
                mito_min = np.percentile(img[0, :, :], )
                max = np.percentile(img, 99, axis=(1,2)).reshape((5,1,1))
                img = (img-min)/max
                assert np.all(np.min(img, axis=(1,2))==0)
                assert np.all(np.max(img,  axis=(1,2))>=1)
                img = np.clip(img,0,1)
                img = np.concatenate([img, cell_mask[None, :, :]], axis=0)


                #add padding:
                img = pad(img, 224)
                cell_mask = pad(cell_mask, 224)
                crops = get_sc_crops(img, cell_mask, SIZE, center=CENTER)
                crops = (crops*255).astype(np.uint8)
                
                plate = nuc_path.split("raw/")[1].split("/outlines")[0]
                savename = f"test_crops/{plate}_{well}f{fov}.npy"
                print(savename)
                np.save(savename, crops)

'''
