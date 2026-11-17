"""FedCCFA (Chen et al., NeurIPS 2024): classifier clustering and feature alignment.

At ``missing_prob = 0`` this is the published method. With missing classes it is
the adapted version FedCCFA-PCS (present-class sampling): the balanced subset is
drawn from present classes only, a client keeps the last received prototype of a
class it does not hold, and prototype averaging within a group skips members
without the class. ``class_support_mask`` gives FedCCFA-PCS+mask: MADD, DBSCAN,
classifier averaging and global prototype averaging for class c use only the
clients that hold class c, and a class or group with fewer than two such clients
is not merged.
"""

from collections import Counter

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch.nn import CosineSimilarity, CrossEntropyLoss
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, Server, clip_gradients
from utils.gen_dataset import ClientDataset


class FedCCFAClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.prototype_criterion = CrossEntropyLoss().to(cfg.device)
        self.local_protos = {}
        self.global_protos = []
        self.local_classifier = []
        self.label_distribution = torch.zeros(cfg.num_classes, dtype=torch.int)
        self.prototype_weight = 0.0

    def classifier_parameters(self):
        return [self.model.fc.weight, self.model.fc.bias]

    def encoder_parameters(self):
        return list(self.model.hidden_layers.parameters())

    def copy_classifier(self):
        return [p.detach().clone() for p in self.classifier_parameters()]

    def set_classifier(self, params):
        for new, local in zip(params, self.classifier_parameters()):
            local.data = new.data.clone()

    def set_encoder(self, params):
        for new, local in zip(params, self.encoder_parameters()):
            local.data = new.data.clone()

    def freeze(self, train_classifier):
        for p in self.encoder_parameters():
            p.requires_grad = not train_classifier
        for p in self.classifier_parameters():
            p.requires_grad = train_classifier

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def update_label_distribution(self):
        num_classes = self.cfg.num_classes
        targets = np.asarray(self.train_set.targets, dtype=np.int64).reshape(-1)
        self.label_distribution = np.bincount(targets, minlength=num_classes)[:num_classes]
        prob = self.label_distribution / max(1, len(targets))
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(prob * np.where(prob > 0, np.log(prob), 0.0))
        gamma = self.cfg.prototype_gamma
        self.prototype_weight = entropy / gamma if gamma != 0 else 0.0

    def balanced_subset(self):
        """min(5, smallest class) samples of every present class, drawn with the global NumPy generator."""
        labels = np.array(self.train_set.targets)
        per_class = min(Counter(labels).most_common()[-1][1], self.cfg.balanced_samples_per_class)
        indices = []
        for c in sorted(set(labels.tolist())):
            class_indices = np.where(labels == c)[0]
            indices.extend(np.random.choice(class_indices, min(per_class, len(class_indices)), replace=False))
        np.random.shuffle(indices)
        return ClientDataset(self.train_set, indices, self.train_set.transform)

    def train_balanced_classifier(self):
        self.model.train()
        loader = DataLoader(self.balanced_subset(), batch_size=self.cfg.batch_size, shuffle=True)
        self.freeze(train_classifier=True)
        optimizer = SGD(self.classifier_parameters(), lr=self.cfg.balanced_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self._fit(loader, optimizer, self.cfg.balanced_epochs)

    def _fit(self, loader, optimizer, epochs):
        for _ in range(epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(self.trainable_parameters(), self.cfg.max_grad_norm)
                optimizer.step()

    def train(self, round_index):
        """Classifier training, then encoder training with prototype alignment."""
        self.set_classifier(self.local_classifier)
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)

        self.freeze(train_classifier=True)
        optimizer = SGD(self.classifier_parameters(), lr=self.cfg.classifier_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self._fit(loader, optimizer, self.cfg.classifier_epochs)

        self.freeze(train_classifier=False)
        optimizer = SGD(self.encoder_parameters(), lr=self.cfg.encoder_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        cos = CosineSimilarity(dim=2).to(self.device)
        for _ in range(self.cfg.encoder_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)
                loss_proto = 0.0
                if (len(self.global_protos) > 0 and round_index >= self.cfg.prototype_start_round
                        and self.prototype_weight != 0):
                    batch_protos = torch.stack(self.global_protos).repeat(len(labels), 1, 1)
                    logits = cos(features.unsqueeze(1), batch_protos)
                    loss_proto = self.prototype_criterion(logits / self.cfg.prototype_temperature, labels)
                loss = loss_sup + self.prototype_weight * loss_proto
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(self.trainable_parameters(), self.cfg.max_grad_norm)
                optimizer.step()

        with torch.no_grad():
            self.local_protos = self.compute_prototypes()
            previous = list(self.global_protos) if len(self.global_protos) > 0 else None
            any_proto = next(iter(self.local_protos.values()))
            self.global_protos = [
                self.local_protos[label] if label in self.local_protos
                else (previous[label] if previous is not None else torch.zeros_like(any_proto))
                for label in range(self.cfg.num_classes)
            ]

    def compute_prototypes(self):
        """Mean feature of every class present in the local data."""
        proto_lists = {}
        loader = DataLoader(self.train_set, batch_size=2048, shuffle=True)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                _, features = self.model(inputs, True)
                features = features.clone().detach()
                for i in range(len(labels)):
                    proto_lists.setdefault(labels[i].item(), []).append(features[i, :])
        return {label: torch.mean(torch.stack(protos), dim=0) for label, protos in proto_lists.items()}


def madd(vectors):
    """FedCCFA's MADD: the cosine of two zero rows is 1, of one zero row 0."""
    def cos_sim(a, b):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise FloatingPointError("non-finite classifier parameters")
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        eps = np.finfo(np.float64).eps
        if norm_a <= eps and norm_b <= eps:
            return 1.0
        if norm_a <= eps or norm_b <= eps:
            return 0.0
        return float(np.clip(a.dot(b) / (norm_a * norm_b), -1.0, 1.0))

    num = len(vectors)
    result = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            dist = 0.0
            for z in range(num):
                if z == i or z == j:
                    continue
                dist += np.abs(cos_sim(vectors[i], vectors[z]) - cos_sim(vectors[j], vectors[z]))
            result[i][j] = result[j][i] = dist / (num - 2)
    return result


class FedCCFAServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_protos = []

    def encoder_parameters(self):
        return list(self.model.hidden_layers.parameters())

    def send_encoder(self, clients):
        params = self.encoder_parameters()
        for client in clients:
            client.set_encoder(params)

    def aggregate_encoders(self, clients):
        params = self.encoder_parameters()
        new_params = torch.zeros_like(parameters_to_vector(params))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            new_params += client_size * parameters_to_vector(client.encoder_parameters())
        new_params /= total_size
        vector_to_parameters(new_params, params)

    def aggregate_prototypes(self, clients):
        """Global prototypes (uniform average); classes absent this round keep their previous value.

        As in the published code, the sum is accumulated in place into the first
        contributing client's prototype tensor.
        """
        totals = {}
        for client in clients:
            for label in client.local_protos.keys():
                if label in totals:
                    totals[label] += client.local_protos[label]
                else:
                    totals[label] = client.local_protos[label]
        if self.cfg.class_support_mask:
            contributors = {}
            for client in clients:
                for label in client.local_protos.keys():
                    contributors[label] = contributors.get(label, 0) + 1
            averaged = {label: proto / contributors[label] for label, proto in totals.items()}
        else:
            averaged = {label: proto / len(clients) for label, proto in totals.items()}
        previous = list(self.global_protos) if len(self.global_protos) > 0 else None
        any_proto = next(iter(averaged.values()))
        self.global_protos = [
            averaged[label] if label in averaged
            else (previous[label] if previous is not None else torch.zeros_like(any_proto))
            for label in range(self.cfg.num_classes)
        ]

    def group_clients(self, classifiers, support):
        """Per-class MADD and DBSCAN on (weight row, bias); optional class-support mask."""
        all_ids = np.array(list(classifiers.keys()))
        all_params = list(classifiers.values())
        groups = {}
        for label in range(self.cfg.num_classes):
            if self.cfg.class_support_mask:
                keep = [i for i, cid in enumerate(all_ids) if label in support.get(int(cid), set())]
            else:
                keep = list(range(len(all_ids)))
            if len(keep) < 2:
                groups[label] = []
                continue
            client_ids = all_ids[keep]
            vectors = np.array([np.hstack([p[label].detach().cpu().numpy() for p in all_params[i]])
                                for i in keep])
            try:
                distances = madd(vectors)
            except FloatingPointError:
                print(f"[fedccfa] non-finite classifier rows for class {label}; not merged this round")
                groups[label] = []
                continue
            labels = DBSCAN(eps=self.cfg.dbscan_eps, min_samples=1, metric="precomputed").fit(distances).labels_
            merged = []
            for cluster in set(labels):
                members = np.where(labels == cluster)[0]
                if len(members) > 1:
                    ids = client_ids[members]
                    ids.sort()
                    merged.append(list(ids))
            groups[label] = merged
        return groups

    def average_rows(self, label, clients):
        server_row = [self.model.fc.weight[label], self.model.fc.bias[label]]
        averaged = torch.zeros_like(parameters_to_vector(server_row))
        for client in clients:
            averaged += parameters_to_vector([client.model.fc.weight[label], client.model.fc.bias[label]])
        averaged /= len(clients)
        return averaged

    def average_group_prototypes(self, label, clients):
        """Mean prototype over group members holding the class, else the global prototype."""
        total, count = None, 0
        for client in clients:
            if label not in client.local_protos:
                continue
            if total is None:
                total = client.local_protos[label].clone()
            else:
                total = total + client.local_protos[label]
            count += 1
        if total is None:
            return self.global_protos[label].clone()
        return total / count
