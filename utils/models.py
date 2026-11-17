"""CNN backbones of FedCCFA (Chen et al., NeurIPS 2024), feature dimension 128.

Every model exposes ``hidden_layers`` (the encoder) and ``fc`` (the linear
classifier); ``model(x, return_features=True)`` returns ``(logits, features)``.
"""

import torch.nn as nn


class CifarCNN(nn.Module):
    """CIFAR-10 CNN, 140,458 parameters."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        feature = self.hidden_layers(x)
        out = self.fc(feature)
        return (out, feature) if return_features else out


class FashionMNISTCNN(nn.Module):
    """Fashion-MNIST CNN, 117,066 parameters."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        feature = self.hidden_layers(x)
        out = self.fc(feature)
        return (out, feature) if return_features else out


MODELS = {"cifar10": CifarCNN, "fashion_mnist": FashionMNISTCNN}


def build_model(dataset, num_classes=10):
    return MODELS[dataset](num_classes)
