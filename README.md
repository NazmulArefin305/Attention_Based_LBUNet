# LBUNet with AttentionSkipFusion (ASF)

This repository provides the official PyTorch implementation of two models:

- **Baseline LBUNet**: A lightweight boundary-assisted UNet with multi-stage deep supervision, boundary prediction, and gated fusion mechanisms.
- **Modified LBUNet**: A streamlined architecture using AttentionSkipFusion (ASF) modules in place of handcrafted fusion blocks, with simplified supervision for faster inference.

🧪 Both models are evaluated on dermoscopic skin lesion segmentation tasks using the ISIC dataset.

## Features
- Full training and evaluation pipeline
- Modular architecture for easy experimentation
- Implements custom loss functions (BCE + Dice)
- Attention-based skip fusion module for learnable encoder-decoder integration
- Reproducible results with sample configurations

## Results Summary
The modified LBUNet achieves approximately 40% faster inference with a minor trade-off in segmentation accuracy.

## Dataset
The ISIC 2018 Skin Lesion Segmentation Dataset was used for training and evaluation. Instructions for dataset preparation are provided in the README.

## Citation
If you use this code in your research, please consider citing our work or acknowledging this repository.

