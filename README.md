# SubCell: Vision foundation models representing cell biology in microscopy    

## Overview
The cell is the functional unit of life, capable of performing a wide range of biological functions underpinned by a myriad of molecular interactions carefully organized in an intricate subcellular architecture. Cellular architecture   can be studied with microscopy images  at scale, and the present era of machine learning has enabled data-driven modeling of these images to reveal the cellular organization and function beyond what humans can readily perceive. Here, we introduce SubCell, a suite of self-supervised deep learning models designed to accurately capture cellular morphology and protein localization in fluorescent microscope images. The models were trained using the metadata-rich, proteome-wide image collection from the Human Protein Atlas. SubCell outperforms state-of-the-art  methods in protein localization and cellular phenotyping. SubCell is generalizable to other fluorescent datasets  spanning different imaging devices and resolutions, including a dataset of perturbed cells, where SubCell succeeds in predicting perturbations and mechanisms of action without any fine-tuning. Finally, we use SubCell to construct the first proteome-wide hierarchical map of proteome organization directly learned from image data.   

![](resources/arch.png)

*Figure 1: Overview of the SubCell self-supervised learning framework. (A) Representative images of single-cell crops from the Human Protein Atlas data set demonstrating the diverse cell morphologies and protein subcellular localizations present in a variety of human cells (Blue: Nucleus; Red: Microtubules; Yellow: endoplasmic reticulum; Green: Protein of interest). (B) Figure 1Illustration depicting our multi-task learning approach to train a vision transformer model. We use three tasks to train our model: reconstruction, cell-specific, and protein-specific tasks.*


## This repository contains to code for the analysis and plotting of the results presented in the bioRxiv paper: https://www.biorxiv.org/content/10.1101/2024.12.06.627299v1 

## Installation
```bash
# Clone the repository
git clone https://github.com/username/subcell-embed.git
cd subcell-embed

# Create conda environment
conda create -n subcell python=3.8
conda activate subcell

# Install requirements
pip install -r requirements.txt
```

## Requirements
```
torch>=2.1.2
pytorch-lightning>=2.2
transformers>=4.41.0
torchmetrics==1.3.0
mosaicml-streaming==0.7.3
```

## Usage

```
python main_lightning.py --config configs/MAE-cellS-protS-contrast.yaml
```