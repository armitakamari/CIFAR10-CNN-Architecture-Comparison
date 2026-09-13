# CIFAR-10 Classification: Architecture Comparison & Grad-CAM

A modular PyTorch implementation comparing custom CNN architectures, residual networks, and transfer learning on CIFAR-10, with model interpretability via Grad-CAM.

---

## Overview

This project trains and systematically compares **5 CNN configurations** on the CIFAR-10 dataset, analyzing performance, convergence, and spatial decision-making via Grad-CAM visual explanations.

### Models Compared
1. **Custom CNN (Baseline)**: Without data augmentation.
2. **Custom CNN (Augmented)**: With random crop, horizontal flip, and color jitter (Ablation study).
3. **Custom Deep CNN**: Deeper feature extractor with LeakyReLU activations.
4. **Residual CNN (ResCNN)**: Custom network with residual skip connections.
5. **Transfer Learning (ResNet-18)**: Pretrained on ImageNet with a frozen feature extractor and adapted classification head.

---

## Key Technical Highlights

* **Model Interpretability (Grad-CAM)**: Generates heatmaps for both **correct** and **misclassified** test predictions to inspect which features the model prioritized.
* **Modern PyTorch Hooks**: Uses `register_full_backward_hook` instead of deprecated backward hooks to ensure gradient integrity in modern PyTorch versions.
* **Input Resolution Alignment**: ResNet-18 inputs are resized to `224×224` during transfer learning to prevent spatial feature collapse and properly leverage ImageNet-pretrained weights.

---

## Project Structure
```text
cifar10-gradcam/
├── data/                  # Auto-downloaded CIFAR-10 dataset
├── notebooks/
│   └── demo.ipynb         # Interactive demo and visualization notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py         # Transforms and DataLoader builders
│   ├── models.py          # PyTorch model definitions
│   ├── interpretability.py# Grad-CAM class and heatmap overlays
│   └── utils.py           # Training loops, metrics, and plotting
├── main.py                # Pipeline execution entry point
├── requirements.txt       # Project dependencies
└── README.md
