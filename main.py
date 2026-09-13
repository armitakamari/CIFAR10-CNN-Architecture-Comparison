import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import make_loaders
from src.models import CustomCNN, CustomCNN_Deep, ResCNN, get_resnet18_transfer
from src.interpretability import show_gradcam
from src.utils import set_seed, train_model, evaluate_test, plot_curves, plot_confusion_matrix

def run_experiment(model, train_loader, val_loader, test_loader, test_set, name, device, epochs=30, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs)
    plot_curves(history, name)

    test_acc, cm, y_true, y_pred = evaluate_test(model, test_loader, device)
    print(f"[{name}] Test Accuracy: {test_acc*100:.2f}%\n")
    
 
    plot_confusion_matrix(cm, test_set.classes, f"{name} - Confusion Matrix")
    show_gradcam(model, test_set, y_true, y_pred, device, correct=True, n=4, title=f"{name} - GradCAM (Correct)")
    show_gradcam(model, test_set, y_true, y_pred, device, correct=False, n=4, title=f"{name} - GradCAM (Misclassified)")

    return model, history, test_acc

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # 1. CustomCNN
    tr_loader, val_loader, test_loader, test_set = make_loaders(use_augmentation=False)
    run_experiment(CustomCNN(), tr_loader, val_loader, test_loader, test_set, "CustomCNN_NoAug", device)

    # 2. CustomCNN
    tr_loader_aug, val_loader_aug, test_loader_aug, test_set = make_loaders(use_augmentation=True)
    run_experiment(CustomCNN(), tr_loader_aug, val_loader_aug, test_loader_aug, test_set, "CustomCNN_Aug", device)

    # 3. CustomCNN_Deep
    run_experiment(CustomCNN_Deep(), tr_loader_aug, val_loader_aug, test_loader_aug, test_set, "CustomCNN_Deep", device)

    # 4. ResCNN
    run_experiment(ResCNN(), tr_loader_aug, val_loader_aug, test_loader_aug, test_set, "ResCNN", device)

    # 5. Transfer Learning ResNet18
    tr_tl, val_tl, test_tl, test_set_tl = make_loaders(use_augmentation=True, batch_size=64, resize_to=224)
    run_experiment(get_resnet18_transfer(), tr_tl, val_tl, test_tl, test_set_tl, "ResNet18_TL", device, epochs=15)

if __name__ == "__main__":
    main()
