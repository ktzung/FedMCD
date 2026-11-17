"""Small stand-in for flamby.datasets.fed_isic2019 with the same interface, for tests without the ISIC data."""
import numpy as np
import torch
import torch.nn as nn

BATCH_SIZE = 8
LR = 5e-4
_MISSING = {0: [], 1: [3], 2: [0, 5], 3: [6], 4: [1, 2, 7], 5: [4]}


class FedIsic2019(torch.utils.data.Dataset):
    def __init__(self, center=0, train=True, pooled=False):
        centers = range(6) if pooled else [center]
        images, targets = [], []
        for c in centers:
            rng = np.random.RandomState(100 * c + int(train))
            n = 37 if train else 13
            allowed = [k for k in range(8) if k not in _MISSING[c]]
            targets.append(rng.choice(allowed, n))
            images.append(rng.rand(n, 3, 12, 12).astype(np.float32))
        self.targets = np.concatenate(targets)
        self.images = np.concatenate(images)
        self.train = train

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy())
        if self.train and np.random.rand() < 0.5:   # augmentation drawn from the global NumPy generator
            image = torch.flip(image, dims=[2])
        return image, int(self.targets[index])


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.head = nn.Conv2d(16, 1280, 1)
        self.bn = nn.BatchNorm2d(1280)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(1280, 8)

    def extract_features(self, x):
        return torch.relu(self.bn(self.head(torch.relu(self.conv(x)))))

    def forward(self, x):
        v = self._dropout(self._avg_pooling(self.extract_features(x)).flatten(start_dim=1))
        return self._fc(v)


class Baseline(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = _Base()

    def forward(self, x):
        return self.base_model(x)


class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)
