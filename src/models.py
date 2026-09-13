import torch
import torch.nn as nn
from torchvision import models

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

class CustomCNN_Deep(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(0.1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), act,
            nn.Conv2d(64, 64, 3, padding=1), act,
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), act,
            nn.Conv2d(128, 128, 3, padding=1), act,
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), act,
            nn.Conv2d(256, 256, 3, padding=1), act,
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            act,
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1)
        )

    def forward(self, x):
        return torch.relu(self.block(x) + x)

class ResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.fc(x.view(x.size(0), -1))

def get_resnet18_transfer():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    return resnet
