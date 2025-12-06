# Depth-Only Inpainting of LiDAR Depth Maps for Autonomous Driving
This repository contains the code accompanying the paper “Depth-Only Inpainting of Depth Maps for Autonomous Driving”, which investigates whether large missing regions in outdoor KITTI depth maps can be reconstructed without RGB, using only geometric depth context.

All training, visualization, and evaluation is contained in:
`train_visualize_evaluate.ipynb`

Open the notebook and run all cells in order.
It includes:
- Loading KITTI data
- Generating synthetic holes
- Training U-Net 
- Saving checkpoints
- Plotting training loss
- Producing qualitative heatmaps & reconstructions
- Computing quantitative metrics

Note that many cells are self contained so long as the trained models are saved in the correct directories. For example, running the training cells are not necessary if you have downloaded the pretrained models.

## Installation
git clone https://github.com/jrai51/comp-imaging-project.git
cd comp-imaging-project
pip install -r requirements.txt

This project uses PyTorch and expects the KITTI Depth Completion dataset (semi-dense depth maps). 

You must download KITTI separately from: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
and select "Download annotated depth maps data set (14 GB)".

Once downloaded, place the depth maps inside:

`comp-imaging-project/data_depth_annotated/`

The data_utils.py file expects this directory structure and will automatically load the depth maps and generate synthetic rectangular occlusions (5–20% of image area

## File Structure

```
comp-imaging-project/
│
├── data_depth_annotated/       # KITTI depth dataset (user-supplied)
│                                # Must be placed in this folder.
│
├── figures/                    # Saved qualitative visualizations
│
├── loss-history/               # json logs for training loss curves
│
├── saved-models/               # Directory for pretrained checkpoints
│                                # (Download link provided below)
│
├── utils/
│   └── data_utils.py           # Dataloader, masking, preprocessing utilities
│
├── models.py                   # U-Net, Autoencoder, and variants
│
├── train_visualize_evaluate.ipynb
│                                # Main reproduction notebook:
│                                # - trains all models
│                                # - runs quantitative evaluations
│                                # - produces qualitative visualizations
│
├── requirements.txt
└── readme.md

```

# Pretrained Models
Pretrained U-Net, Autoencoder, and U-Net-no-TV checkpoints are available via Google Drive:
Download link here:
https://drive.google.com/drive/folders/1FYKc2KRopwvURJ8mZTvRWvKMtZuOMKDX?usp=drive_link

