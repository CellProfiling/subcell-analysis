#Adapted from: https://github.com/cytomining/DeepProfiler/blob/master/deepprofiler/dataset/illumination_statistics.py


#import deepprofiler.dataset.utils as utils
#import deepprofiler.dataset.image_dataset
import skimage.transform
import numpy as np
import os
import pickle as pickle
from preprocessing.illumination_correction import IlluminationCorrection
from preprocessing.dataset import JUMP_Dataset
#import torch


def percentile(prob, p):
    cum = np.cumsum(prob)
    pos = cum > p
    return np.argmax(pos)


#################################################
## COMPUTATION OF ILLUMINATION STATISTICS
#################################################

# Build pixel histogram for each channel
class IlluminationStatistics():
    def __init__(self, bits, channels, down_scale_factor, median_filter_size, name=""):
        self.depth = 2 ** bits
        self.channels = channels
        self.name = name
        self.down_scale_factor = down_scale_factor
        self.median_filter_size = median_filter_size
        self.hist = np.zeros((len(channels), self.depth), dtype=np.float64)
        self.count = 0
        self.expected = 1
        self.mean_image = None
        self.original_image_size = None

    def processImage(self, img):
        self.addToMean(img)
        self.count += 1
        #utils.logger.info("Plate {} Image {} of {} ({:4.2f}%)".format(self.name,
                                                                      #self.count, self.expected,
                                                                      #100 * float(self.count) / self.expected))
        for i in range(len(self.channels)):
            counts = np.histogram(img[:, :, i], bins=self.depth, range=(0, self.depth))[0]
            self.hist[i] += counts.astype(np.float64)

    # Accumulate the mean image. Useful for illumination correction purposes
    def addToMean(self, img):
        # Check image size (we assume all images have the same size)
        if self.original_image_size is None:
            self.original_image_size = img.shape
            self.scale = (img.shape[0] / self.down_scale_factor, img.shape[1] / self.down_scale_factor)
        else:
            if img.shape != self.original_image_size:
                raise ValueError("Images in this plate don't match: required=",
                                 self.original_image_size, " found=", img.shape)
        # Rescale original image to half
        thumb = skimage.transform.resize(img, self.scale, mode="reflect", anti_aliasing=True, preserve_range=True)
        if self.mean_image is None:
            self.mean_image = np.zeros_like(thumb, dtype=np.float64)
        # Add image to current mean values
        self.mean_image += thumb
        return

    # Compute global statistics on pixels. 
    def computeStats(self):
        # Initialize counters
        bins = np.linspace(0, self.depth - 1, self.depth)
        mean = np.zeros((len(self.channels)))
        lower = np.zeros((len(self.channels)))
        upper = np.zeros((len(self.channels)))
        self.mean_image /= self.count

        # Compute percentiles and histogram
        for i in range(len(self.channels)):
            probs = self.hist[i] / self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = percentile(probs, 0.0001)
            upper[i] = percentile(probs, 0.9999)
        stats = {"mean_values": mean, "upper_percentiles": upper, "lower_percentiles": lower, "histogram": self.hist,
                 "mean_image": self.mean_image, "channels": self.channels, "original_size": self.original_image_size}

        # Compute illumination correction function and add it to the dictionary
        correct = IlluminationCorrection(stats, self.channels, self.original_image_size)
        correct.compute_all(self.median_filter_size)
        stats["illum_correction_function"] = correct.illum_corr_func

        # Plate ready
        #utils.logger.info("Plate " + self.name + " done")
        return stats



def calculate_statistics(plate_csv, plate_name, outfile, bits=16, channels=["Mito", "AGP", "NucRNA", "ER", "DAPI"], down_scale_factor=4, median_filter_size=24):
    #arg default values copied from DeepProfiler configs (https://github.com/broadinstitute/DeepProfilerExperiments/tree/master/bbbc021)

    hist = IlluminationStatistics(bits, channels, down_scale_factor, median_filter_size, name=plate_name)
    
    dataset = JUMP_Dataset(plate_csv)
    #dataset = torch.utils.data.Subset(dataset, [0,1,2,3,4,5,6,7,8,9]) #for testing...

    for sample in dataset:
        #if sample["idx"] % 100==0: print(sample["idx"])
        img = sample["image"]
        #min-max normalize img first or else mean_img pix vals will be way too large ???? --> no float64 can be up to 10^308
        hist.processImage(sample["image"])

    stats = hist.computeStats()

    #SAVE STATS
    with open(outfile, "wb") as output:
        pickle.dump(stats, output)

    #SAVE ILLUM_CORRECTION FUNCTION SEPARATELY
    outfile = outfile.replace("illumstats.pkl", "illum_corr_fxn.npy")
    with open(outfile, 'wb') as f:
        np.save(f, stats["illum_correction_function"])


