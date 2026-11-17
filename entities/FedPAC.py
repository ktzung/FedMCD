"""FedPAC (Xu et al., ICLR 2023): feature alignment to global prototypes and classifier collaboration.

The classifier combination weights are found by a quadratic program (cvxpy).
"""

from collections import Counter

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, EncoderServer, class_mean_features, classifier_parameters, encoder_parameters, freeze


class FedPACClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.mse_criterion = MSELoss().to(cfg.device)
        self.label_distribution = torch.zeros(cfg.num_classes)
        self.label_prob = torch.zeros(cfg.num_classes)
        self.local_protos = None

    def update_label_distribution(self):
        """Counts and frequencies of the present classes; entries of absent classes are kept."""
        for label, size in Counter(self.train_set.targets).items():
            self.label_distribution[label] = size
            self.label_prob[label] = size / len(self.train_set)

    def train_with_protos(self, global_protos):
        """Classifier epochs, then encoder epochs with an L2 pull towards the global prototypes."""
        self.model.train()
        local_protos = class_mean_features(self.model, self.train_set, self.cfg.batch_size, self.device)
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        freeze(self.model, train_classifier=True)
        optimizer = SGD(classifier_parameters(self.model), lr=self.cfg.classifier_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.classifier_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        freeze(self.model, train_classifier=False)
        optimizer = SGD(encoder_parameters(self.model), lr=self.cfg.encoder_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.encoder_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)
                loss_proto = 0.0
                if len(global_protos) > 0:
                    targets = features.clone().detach()
                    for i in range(len(labels)):
                        label = labels[i].item()
                        source = global_protos if label in global_protos else local_protos
                        targets[i] = source[label].detach()
                    loss_proto = self.mse_criterion(features, targets)
                loss = loss_sup + self.cfg.prototype_weight * loss_proto
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            self.local_protos = class_mean_features(self.model, self.train_set, self.cfg.batch_size, self.device)

    def feature_statistics(self):
        """Feature variance term v and class-weighted mean features h used for classifier collaboration."""
        dim = self.model.fc.weight.shape[1]
        features_by_class = {}
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                _, features = self.model(inputs, True)
                features = features.clone().detach().cpu()
                for i in range(len(labels)):
                    features_by_class.setdefault(labels[i].item(), []).append(features[i, :])
        for label, values in features_by_class.items():
            features_by_class[label] = torch.stack(values)
        py = self.label_prob
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.cfg.num_classes, dim))
        for label in range(self.cfg.num_classes):
            if label in features_by_class:
                feats = features_by_class[label]
                count = feats.shape[0]
                mu = feats.mean(dim=0)
                h_ref[label] = py[label] * mu
                v += (py[label] * torch.trace(torch.mm(torch.t(feats), feats) / count)).item()
                v -= (py2[label] * torch.mul(mu, mu)).sum().item()
        v /= len(self.train_set)
        return v, h_ref


class FedPACServer(EncoderServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_protos = {}

    def aggregate_protos(self, clients):
        """Global prototypes weighted by class counts."""
        totals, sizes = {}, {}
        for client in clients:
            for label in client.local_protos.keys():
                weighted = client.local_protos[label] * client.label_distribution[label]
                if label in totals:
                    totals[label] += weighted
                    sizes[label] += client.label_distribution[label]
                else:
                    totals[label] = weighted
                    sizes[label] = client.label_distribution[label]
        self.global_protos = {label: total / sizes[label] for label, total in totals.items()}

    @staticmethod
    def combination_weights(variances, means, num_clients):
        """For each client, the convex weights over clients that minimize the estimated classifier risk."""
        import cvxpy as cvx

        num_classes, dim = means[0].shape
        v = torch.tensor(variances)
        weights = []
        for index in range(num_clients):
            h_ref = means[index]
            dist = torch.zeros((num_clients, num_clients))
            for j1 in range(num_clients):
                for j2 in range(j1, num_clients):
                    h = torch.zeros((dim, dim))
                    for k in range(num_classes):
                        h += torch.mm((h_ref[k] - means[j1][k]).reshape(dim, 1),
                                      (h_ref[k] - means[j2][k]).reshape(1, dim))
                    dist[j1][j2] = dist[j2][j1] = torch.trace(h)
            p_matrix = (torch.diag(v) + dist).numpy()
            eigenvalues, eigenvectors = torch.linalg.eig(torch.tensor(p_matrix))
            p_matrix_new = 0
            for ii in range(num_clients):
                real_value = torch.view_as_real(eigenvalues[ii])[0]
                if real_value >= 0.01:
                    p_matrix_new += real_value * torch.mm(eigenvectors[:, ii].reshape(num_clients, 1),
                                                          eigenvectors[:, ii].reshape(1, num_clients))
            p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix) >= 0.0) else p_matrix
            if np.all(np.linalg.eigvals(p_matrix) >= 0):
                alpha_var = cvx.Variable(num_clients)
                problem = cvx.Problem(cvx.Minimize(cvx.quad_form(alpha_var, p_matrix)),
                                      [cvx.sum(alpha_var) == 1.0, alpha_var >= 0])
                problem.solve()
                alpha = [a * (a > 1e-3) for a in alpha_var.value]
            else:
                alpha = None
            weights.append(alpha)
        return weights

    @staticmethod
    def combine_classifiers(all_params, weights):
        new_params = torch.zeros_like(all_params[0])
        for i in range(len(all_params)):
            new_params += weights[i] * all_params[i]
        return new_params
