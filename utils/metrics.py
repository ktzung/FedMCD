"""Accuracy evaluation and per-round result files."""

import csv
import math
import os

import torch
from torch.utils.data import DataLoader

EVAL_BATCH_SIZE = 4096


def accuracy(model, dataset, device):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += inputs.data.size()[0]
            correct += (pred == labels.data).sum().item()
    return correct / float(total)


def average_loss(model, dataset, criterion, device):
    """Mean loss of ``model`` over ``dataset``."""
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            total_loss += criterion(model(inputs), labels).item() * inputs.shape[0]
    return total_loss / len(dataset.targets)


class CurveWriter:
    """Write one CSV row per round; the file is renamed from ``.partial`` when complete."""

    FIELDS = ("round", "local_test_accuracy", "test_accuracy")

    def __init__(self, path, rounds):
        self.path = path
        self.partial_path = path + ".partial"
        self.rounds = rounds
        self.next_round = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.handle = open(self.partial_path, "w", newline="", buffering=1)
        self.writer = csv.writer(self.handle)
        self.writer.writerow(self.FIELDS)

    def append(self, round_index, local_test_accuracy, test_accuracy):
        if round_index != self.next_round:
            raise ValueError(f"expected round {self.next_round}, got {round_index}")
        values = (float(local_test_accuracy), float(test_accuracy))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"non-finite accuracy at round {round_index}: {values}")
        self.writer.writerow((round_index, repr(values[0]), repr(values[1])))
        self.next_round += 1

    def close(self):
        self.handle.close()
        if self.next_round == self.rounds:
            os.replace(self.partial_path, self.path)


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
