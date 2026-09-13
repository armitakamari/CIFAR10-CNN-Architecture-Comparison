import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_one_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / len(loader), correct / total

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=30):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_one_epoch(model, val_loader, optimizer, criterion, device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch+1:02d} | Train Acc: {tr_acc:.3f} | Val Acc: {va_acc:.3f} | Train Loss: {tr_loss:.3f} | Val Loss: {va_loss:.3f}")
    return history

def evaluate_test(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm, np.array(y_true), np.array(y_pred)

def plot_curves(history, title_prefix, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title(f"{title_prefix} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title(f"{title_prefix} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
