# CIFAR-10 Classification: Architecture Comparison & Grad-CAM

Modular PyTorch project comparing 5 CNN architectures on CIFAR-10, with Grad-CAM for model interpretability.

## Models
1. **Custom CNN (Baseline)** — no augmentation
2. **Custom CNN (Augmented)** — random crop, flip, color jitter
3. **Custom Deep CNN** — deeper extractor, LeakyReLU
4. **ResCNN** — custom network with residual connections
5. **ResNet-18 (Transfer Learning)** — pretrained, frozen backbone, custom head

## Highlights
- **Grad-CAM** heatmaps for correct and misclassified predictions
- Uses `register_full_backward_hook` for gradient-safe hooks
- ResNet-18 inputs resized to 224×224 to preserve pretrained spatial features

## Structure
```
cifar10-gradcam/
├── data/                    # Auto-downloaded CIFAR-10
├── notebooks/demo.ipynb     # Interactive demo
├── src/
│   ├── dataset.py           # Transforms & DataLoaders
│   ├── models.py            # Model definitions
│   ├── interpretability.py  # Grad-CAM
│   └── utils.py             # Training/metrics/plots
├── main.py                  # Entry point
└── requirements.txt
```
