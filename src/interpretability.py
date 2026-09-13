import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.dataset import CIFAR_MEAN, CIFAR_STD

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def heatmap(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def denorm(img_tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)

def show_gradcam(model, test_set, y_true, y_pred, device, correct=True, n=4, title="Grad-CAM", save_path=None):
    model.eval()
    idxs = np.where((y_true == y_pred) if correct else (y_true != y_pred))[0]
    if len(idxs) == 0:
        print("No samples found for this condition.")
        return
    idxs = idxs[:n]

    if hasattr(model, 'layer4'):
        cam_engine = GradCAM(model, model.layer4)
    else:
        target_layer = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                target_layer = m
        cam_engine = GradCAM(model, target_layer)

    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(idxs, 1):
        img_t, label = test_set[idx]
        x = img_t.unsqueeze(0).to(device)
        out = model(x)
        pred = out.argmax(1).item()

        cam = cam_engine.heatmap(x, pred)
        img = denorm(img_t)

        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.65 * img + 0.35 * heat, 0, 1)

        ax = plt.subplot(1, n, i)
        ax.imshow(overlay)
        ax.set_title(f"T:{label} P:{pred}")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
