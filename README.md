# CIFAR-10: CNN Architecture Comparison & Grad-CAM Interpretability

A modular PyTorch project comparing five CNN architectures on CIFAR-10 — from a simple custom CNN to a fine-tuned ResNet-18 — with Grad-CAM interpretability and a data-augmentation ablation study.

## Models Compared

| # | Model | Description |
|---|-------|-------------|
| 1 | **CustomCNN (No Aug)** | Baseline custom CNN, no augmentation |
| 2 | **CustomCNN (Aug)** | Same architecture + random crop, flip, color jitter |
| 3 | **CustomCNN (Deep)** | Deeper feature extractor, LeakyReLU activations |
| 4 | **ResCNN** | Custom network with residual connections |
| 5 | **ResNet-18 (Transfer Learning)** | Pretrained backbone (frozen), custom classification head |

## Results

Results below are from a subset run (5,000 train / 1,000 val / 10,000 test images, 10 epochs, seed 42) — a quick, reproducible comparison rather than a full-data benchmark. Full config and metrics: [`results/local_run/config.json`](results/local_run/config.json), [`results/local_run/comparison.csv`](results/local_run/comparison.csv).

| Model | Test Accuracy | Training Time |
|-------|:---:|:---:|
| CustomCNN (No Aug) | 64.5% | 38s |
| CustomCNN (Aug) | 64.0% | 43s |
| CustomCNN (Deep) | 60.6% | 66s |
| ResCNN | 60.9% | 66s |
| **ResNet-18 (Transfer Learning)** | **84.1%** | 278s |


Transfer learning clearly wins on accuracy at this data scale, at roughly 4–7x the training cost of the custom CNNs. Test accuracy for CustomCNN with and without augmentation is nearly identical, but the training curves tell a more interesting story: the no-augmentation model overfits sharply (91.9% train vs. 65.6% val by epoch 10), while the augmented model shows almost no train/val gap (63.2% vs. 65.7%). Augmentation is doing its job as a regularizer — it just hasn't yet translated into higher test accuracy at only 10 epochs; a longer run would likely let the augmented model pull ahead.

Per-model folders under `results/local_run/` contain training curves, confusion matrices, and Grad-CAM visualizations (correct vs. misclassified predictions) for each architecture.

## Analysis

**Error patterns are consistent across all five architectures.** Reading the confusion matrices, *cat* and *dog* are confused with each other far more than any other pair, in both the weakest model and the strongest:

| Model | Top confusion pairs (count) |
|---|---|
| CustomCNN (No Aug) | dog→cat (225), cat→dog (176), ship→airplane (160), truck→automobile (152) |
| ResNet-18 (TL) | cat→dog (174), dog→cat (92), airplane→ship (73), deer→horse (62) |

This matches the well-known CIFAR-10 difficulty ranking: *cat* and *dog* share pose, texture, and low-resolution silhouette at 32×32, while *bird* is consistently the other weak class (confused mainly with *frog* and *deer*). Rigid, texturally-distinct vehicle classes — **truck, automobile, ship** — are the easiest across every model (all above 90% recall for ResNet-18). This says more about which classes are inherently hard at 32×32 resolution than about any one architecture.

**Per-class accuracy, ResNet-18 (best model):**

| Weakest classes | Strongest classes |
|---|---|
| cat — 68.0% | truck — 91.2% |
| bird — 75.2% | automobile — 92.1% |
| deer — 77.8% | ship — 92.9% |

**What this means for the Grad-CAM results:** the `gradcam_misclassified.png` outputs are most informative exactly on the cat/dog pairs above — worth calling out in the write-up whether the heatmaps focus on discriminative regions (ears, snout shape) even when the class prediction is wrong, versus latching onto background/texture cues. That distinction is the actual interpretability story, more than the raw accuracy numbers.

**Limitations of this run** (worth stating explicitly, since it's a subset experiment):
- Only 5,000/1,000/10,000 train/val/test images and 10 epochs — custom CNNs (esp. `CustomCNN_Deep` and `ResCNN`) were still improving at epoch 10 and had not converged.
- `CustomCNN_Aug` shows no overfitting gap (train ≈ val accuracy) but hasn't yet translated that into a higher test score than `CustomCNN_NoAug` — a longer run would likely change this ranking.
- ResNet-18 uses a frozen backbone (linear-probe style transfer learning); fine-tuning the later backbone layers would likely close more of the gap on the hard classes (cat, bird, deer).

## Highlights

- **Grad-CAM** heatmaps for correct and misclassified predictions, using `register_full_backward_hook` for gradient-safe hooks
- **Data augmentation ablation** — same architecture, with and without augmentation
- ResNet-18 inputs resized to 224×224 to preserve pretrained spatial features
- Deterministic, seeded train/val/test splits for fair comparison across models

## Setup

```bash
git clone https://github.com/armitakamari/CIFAR10-CNN-Architecture-Comparison.git
cd CIFAR10-CNN-Architecture-Comparison
pip install -r requirements.txt
```

## Usage

Run all five models and save metrics/plots:

```python
from main import main

rows = main(
    epochs=10,
    transfer_epochs=10,
    train_limit=5000,
    val_limit=1000,
    test_limit=10000,
    output="results/local_run",
)
```

Or open `notebooks/run_comparison.ipynb` for an end-to-end run with inline visualizations, or `notebooks/demo.ipynb` for a lighter interactive walkthrough.

