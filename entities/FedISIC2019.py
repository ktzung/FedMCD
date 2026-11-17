"""Fed-ISIC2019 case study: client updates and server steps of the baselines and of FedMCD
(Additional file 1, Section S7)."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from utils.anchors import update_anchors
from utils.isic import FEATURE_DIM, NUM_CLASSES, features


FEDPROX_MU = 0.001
FEDGELA_TEMPERATURE = 0.1


def class_balance_weights(counts, num_classes, device):
    """FedGELA phi_c = C * n_c / n from the local class counts."""
    counts = torch.as_tensor(counts, device=device, dtype=torch.float32).flatten()
    total = counts.sum()
    if total.item() <= 0.0:
        return torch.zeros_like(counts)
    return counts * (float(num_classes) / total)


def aggregate(states, sizes):
    total = float(sum(sizes))
    result = {}
    for key in states[0]:
        first = states[0][key]
        if not first.is_floating_point():
            result[key] = first.clone()
            continue
        value = torch.zeros_like(first)
        for state, size in zip(states, sizes):
            value.add_(state[key], alpha=float(size) / total)
        result[key] = value
    return result


def predict_global(model, loader, device, method, anchors):
    truth, predictions = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            if method == "fedgela":
                logits = features(model, inputs) @ anchors.T / FEDGELA_TEMPERATURE
            else:
                logits = model(inputs)
            truth.append(labels.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(truth), np.concatenate(predictions)


RESIDUAL_PENALTY = 0.01
MAX_CONE_RADIUS = 0.25
PROTOTYPE_TEMPERATURE = 0.1
PROTOTYPE_GAMMA = 20.0
PROTOTYPE_START_ROUND = 20
CLASSIFIER_UPDATE_FRACTION = 0.2
ANCHOR_WARMUP_ROUNDS = 20
ANCHOR_UPDATE_EVERY = 5
DBSCAN_EPS = 0.1


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def prototype_weight(counts):
    probabilities = counts / counts.sum().clamp_min(1.0)
    positive = probabilities[probabilities > 0]
    entropy = float((-(positive * positive.log()).sum()).item()) if positive.numel() else 0.0
    return entropy / PROTOTYPE_GAMMA


def local_prototypes(model, loader, device):
    sums = torch.zeros(NUM_CLASSES, FEATURE_DIM, device=device)
    counts = torch.zeros(NUM_CLASSES, device=device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sums.index_add_(0, labels, features(model, inputs))
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
    return {label: (sums[label] / counts[label]).detach() for label in range(NUM_CLASSES)
            if counts[label].item() > 0}


def predict_with_head(model, loader, device, weight, bias):
    truth, predictions = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = F.linear(features(model, inputs), weight, bias)
            truth.append(labels.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(truth), np.concatenate(predictions)


def finite_state(state):
    return all(not value.is_floating_point() or torch.isfinite(value).all() for value in state.values())


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return 0.0 if denominator < 1e-8 else float(left.dot(right) / denominator)


def madd(vectors):
    """MADD (Step 5) computed pair by pair."""
    count = len(vectors)
    result = np.zeros((count, count), dtype=np.float64)
    if count < 3:
        for i in range(count):
            for j in range(i + 1, count):
                result[i, j] = result[j, i] = 1.0 - cosine(vectors[i], vectors[j])
        return result
    for i in range(count):
        for j in range(i + 1, count):
            value = 0.0
            for z in range(count):
                if z in (i, j):
                    continue
                value += abs(cosine(vectors[i], vectors[z]) - cosine(vectors[j], vectors[z]))
            result[i, j] = result[j, i] = value / (count - 2)
    return result


def average_encoders(states, sizes, classifier_prefix="base_model._fc."):
    """Sample-size weighted average of the encoder; classifier keys are copied from the first state."""
    total = float(sum(sizes))
    result = {}
    for key, first in states[0].items():
        if key.startswith(classifier_prefix) or not first.is_floating_point():
            result[key] = first.clone()
            continue
        value = torch.zeros_like(first)
        for state, size in zip(states, sizes):
            value.add_(state[key], alpha=float(size) / total)
        result[key] = value
    return result


def anchor_update(initial_anchors, anchors, prototypes_per_client):
    """Step 4 from the uploaded prototypes of the clients holding each class."""
    classes = anchors.shape[0]
    device, dtype = anchors.device, anchors.dtype
    observed = anchors.clone()
    supports = torch.zeros(classes, dtype=torch.long, device=device)
    for label in range(classes):
        vectors = []
        for prototypes in prototypes_per_client:
            if label not in prototypes:
                continue
            vector = prototypes[label].to(device=device, dtype=dtype)
            if torch.isfinite(vector).all() and vector.norm().item() > 1e-8:
                vectors.append(vector / vector.norm().clamp_min(1e-8))
        supports[label] = len(vectors)
        if vectors:
            mean = torch.stack(vectors).mean(dim=0)
            observed[label] = mean / mean.norm().clamp_min(1e-8)
    new, updated = update_anchors(initial_anchors, anchors, observed, supports, gram_weight=0.25, ema=0.9,
                                  iterations=5, min_clients=2, max_residual=1.0, max_condition=1e6)
    return new, updated


def average_global_prototypes(previous, prototypes_per_client):
    result = previous.clone()
    for label in range(previous.shape[0]):
        vectors = [p[label].to(device=previous.device, dtype=previous.dtype) for p in prototypes_per_client
                   if label in p and torch.isfinite(p[label]).all()]
        if vectors:
            result[label] = torch.stack(vectors).mean(dim=0)
    return result


@dataclass
class GroupAverages:
    weights: list
    biases: list
    global_weight: torch.Tensor
    global_bias: torch.Tensor
    num_groups: int


def group_and_average(weights, biases, presences, anchors, eps=DBSCAN_EPS):
    """Steps 5-6: group anchor-subtracted rows by MADD + DBSCAN, average over holders of the class."""
    num_clients = len(weights)
    classes = anchors.shape[0]
    personal_w = [anchors.clone() for _ in range(num_clients)]
    personal_b = [torch.zeros(classes, device=anchors.device, dtype=anchors.dtype) for _ in range(num_clients)]
    global_w = anchors.clone()
    global_b = torch.zeros(classes, device=anchors.device, dtype=anchors.dtype)
    num_groups = 0
    for label in range(classes):
        residuals = []
        for k in range(num_clients):
            if bool(presences[k][label]):
                residuals.append((weights[k][label] - anchors[label]).detach().cpu().numpy())
            else:
                residuals.append(np.zeros(anchors.shape[1], dtype=np.float32))
        distances = np.nan_to_num(madd(np.asarray(residuals)), nan=0.0, posinf=1.0, neginf=0.0)
        labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit(distances).labels_
        for cluster in sorted(set(labels.tolist())):
            members = [k for k, value in enumerate(labels) if value == cluster]
            holders = [k for k in members if bool(presences[k][label])]
            if holders:
                row = torch.stack([weights[k][label] for k in holders]).mean(dim=0)
                bias = torch.stack([biases[k][label] for k in holders]).mean(dim=0)
            else:
                row = anchors[label]
                bias = torch.zeros((), device=anchors.device, dtype=anchors.dtype)
            for k in members:
                personal_w[k][label] = row
                personal_b[k][label] = bias
            num_groups += 1
        holders = [k for k in range(num_clients) if bool(presences[k][label])]
        if holders:
            global_w[label] = torch.stack([weights[k][label] for k in holders]).mean(dim=0)
            global_b[label] = torch.stack([biases[k][label] for k in holders]).mean(dim=0)
    return GroupAverages(personal_w, personal_b, global_w, global_b, num_groups)
