# crowd-counting
# CCTrans: Vision Transformer-Based Crowd Counting with Density Maps

CCTrans is a deep learning pipeline for crowd counting using Vision Transformers (ViTs).  
It predicts a **density map** for each input crowd image and estimates total crowd counts by integrating over the map.  
This implementation uses a pre-trained ViT backbone with a custom decoder and is trained on custom datasets with provided image and ground-truth point annotation files.

---

## 📑 Features

- Vision Transformer (ViT-B/16) backbone pre-trained on ImageNet-21k.
- Custom multi-level feature fusion decoder for density map generation.
- Data preprocessing utility to convert point annotations (.mat files) to Gaussian-filtered density maps.
- Augmentation strategies: resizing, random horizontal flip, color jitter.
- Support for both training and evaluation modes.
- Evaluation metrics: **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.
- Optional visualization of predicted density maps.
  # CrowdDiff: Crowd Counting with U-Net Density Map Estimation

CrowdDiff is a PyTorch-based crowd counting model that predicts **density maps** for crowd images, enabling total count estimation through integration over these maps.  
It uses a U-Net style architecture for denoising and density map prediction with an optional pretrained model for transfer learning.

---

## 📑 Features

- U-Net-based **denoising diffusion-inspired architecture** for density map generation.
- Customizable dataset loader for image, point-annotation (.mat) and density map handling.
- Preprocessing utility to convert point annotations to Gaussian-blurred density maps.
- Training and evaluation pipeline with **MAE** and **RMSE** metrics.
- Visualization utility for qualitative inspection of predictions.
- Supports pretrained model loading for fine-tuning or evaluation.
