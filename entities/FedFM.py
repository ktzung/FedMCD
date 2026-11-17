"""FedFM (Ye et al., IEEE TSP 2023): feature matching to global class anchors."""

from collections import Counter

import numpy as np
import torch
from torch.nn import CosineSimilarity, CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, Server, class_mean_features


class FedFMClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.anchor_criterion = CrossEntropyLoss().to(cfg.device)
        self.local_anchors = None
        self.label_distribution = torch.zeros(cfg.num_classes, dtype=torch.int)

    def update_label_distribution(self):
        """Counts of the present classes; counts of classes no longer present are kept."""
        for label, size in Counter(self.train_set.targets).items():
            self.label_distribution[label] = size
        self.label_distribution = np.array(self.label_distribution)

    def train_with_anchors(self, global_anchors, anchor_weight):
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        momentum=self.cfg.momentum)
        cos = CosineSimilarity(dim=2).to(self.device)
        for _ in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)
                loss_anchor = 0.0
                if len(global_anchors) > 0:
                    logits = cos(features.unsqueeze(1), torch.stack(global_anchors).repeat(len(labels), 1, 1))
                    loss_anchor = self.anchor_criterion(logits / self.cfg.anchor_temperature, labels)
                loss = loss_sup + anchor_weight * loss_anchor
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            self.local_anchors = class_mean_features(self.model, self.train_set, 2048, self.device)


class FedFMServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_anchors = []

    def aggregate_anchors(self, clients):
        """Global anchor of each class: mean of the local anchors weighted by class counts."""
        totals, sizes = {}, {}
        for client in clients:
            for label in client.local_anchors.keys():
                weighted = client.local_anchors[label] * client.label_distribution[label]
                if label in totals:
                    totals[label] += weighted
                    sizes[label] += client.label_distribution[label]
                else:
                    totals[label] = weighted
                    sizes[label] = client.label_distribution[label]
        averaged = {label: total / sizes[label] for label, total in totals.items()}
        self.global_anchors = [averaged[label] for label in range(self.cfg.num_classes)]
