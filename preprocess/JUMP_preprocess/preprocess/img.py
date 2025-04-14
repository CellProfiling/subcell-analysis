#Adopted from https://github.com/cytomining/DeepProfiler/blob/84c388db144c9f2c8515d7b4bac7fb94c4608d01/deepprofiler/dataset/pixels.py

import numpy as np
from imageio import imread, imwrite
from preprocessing.preprocess_utils import make_masks
import os


# Main image reading function. Images are treated as 3D tensors: (height, width, channels)
def openImage(paths, outline_paths=None):
    channels = [imread(p) for p in paths]
    img = np.zeros((channels[0].shape[0], channels[0].shape[1], len(channels)))
    for c in range(len(channels)):
        img[:,:,c] = channels[c]
    if outline_paths is not None:
        cell_outline = imread(outline_paths[0])
        nuc_outline = imread(outline_paths[1])
        cell_mask = make_masks(cell_outline, nuc_outline)
        img = np.concatenate((img, cell_mask[:,:,np.newaxis]), axis=2)
    return img.astype(np.float32)

def saveImage(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, img)


# In this version the cell_mask is actually outlines
# But it seems to me like this does not separate outlines which are touching
'''
import skimage.measure
def openImageBroadVersion(paths, outlines=None):
    channels = [imread(p) for p in paths ]
    img = np.zeros((channels[0].shape[0], channels[0].shape[1], len(channels)))
    for c in range(len(channels)):
        img[:,:,c] = channels[c]
    if outlines is not None:
        boundaries = skimage.io.imread(outlines)
        labels = skimage.measure.label(boundaries, background=0)
        img = np.concatenate((img, cell_mask[:,:,np.newaxis]), axis=2)
    return img
'''
