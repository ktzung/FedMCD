"""Shared parts of the Fed-ISIC2019 case study (FLamby data, EfficientNet-B0, evaluation)."""

import copy
import csv
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

NUM_CLIENTS = 6
NUM_CLASSES = 8
FEATURE_DIM = 1280
LOCAL_UPDATES = 100
INIT_STATE_NAME = "efficientnet_b0_pretrained_seed0.pt"


def flamby():
    """Import FLamby's Fed-ISIC2019 module (installed separately, see README)."""
    from flamby.datasets import fed_isic2019
    return fed_isic2019


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_initialization(path):
    """Save the ImageNet-pretrained EfficientNet-B0 once, so every method starts from the same weights."""
    if os.path.exists(path):
        return
    seed_all(0)
    model = flamby().Baseline(pretrained=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def make_model(init_path, device):
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"run with --prepare-only first: {init_path}")
    model = flamby().Baseline(pretrained=False)
    model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
    return model.to(device)


def classifier(model):
    return model.base_model._fc


def features(model, inputs):
    """EfficientNet-B0 features before the classifier (d = 1280)."""
    base = model.base_model
    value = base.extract_features(inputs)
    value = base._avg_pooling(value)
    value = value.flatten(start_dim=1)
    return base._dropout(value)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def optimizer_to_cpu(state):
    copied = copy.deepcopy(state)
    for slot in copied.get("state", {}).values():
        for key, value in slot.items():
            if torch.is_tensor(value):
                slot[key] = value.cpu()
    return copied


def optimizer_to_device(optimizer, device):
    for slot in optimizer.state.values():
        for key, value in slot.items():
            if torch.is_tensor(value):
                slot[key] = value.to(device)


def warmup_factor(round_index, total_rounds):
    """Warm-up of 20 rounds and ramp of 20 rounds; runs shorter than 40 rounds adapt from round 1."""
    if total_rounds >= 40:
        if round_index < 20:
            return 0.0
        return min(1.0, (round_index - 20 + 1) / 20.0)
    return 0.0 if round_index < 1 else 1.0


def metric_triplet(truth, pred):
    return (
        float(balanced_accuracy_score(truth, pred)),
        float(f1_score(truth, pred, average="macro", zero_division=0)),
        float(accuracy_score(truth, pred)),
    )


class MetricsWriter:
    FIELDS = ["round", "pooled_balanced_accuracy", "pooled_macro_f1", "pooled_accuracy",
              "mean_center_balanced_accuracy", "std_center_balanced_accuracy",
              "worst_center_balanced_accuracy", "center_balanced_accuracies"]

    def __init__(self, path, extra_fields=()):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.handle = open(path, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS + list(extra_fields))
        self.writer.writeheader()

    def write(self, round_index, pooled, center_balanced, **extra):
        balanced, f1, acc = pooled
        self.writer.writerow({
            "round": round_index,
            "pooled_balanced_accuracy": balanced,
            "pooled_macro_f1": f1,
            "pooled_accuracy": acc,
            "mean_center_balanced_accuracy": float(np.mean(center_balanced)),
            "std_center_balanced_accuracy": float(np.std(center_balanced)),
            "worst_center_balanced_accuracy": min(center_balanced),
            "center_balanced_accuracies": json.dumps(center_balanced),
            **extra,
        })

    def close(self):
        self.handle.close()
