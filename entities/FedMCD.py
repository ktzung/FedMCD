"""FedMCD client and server.

Client: Pre-step and Steps 1-3 of each round (Sections 4.2 and 4.3 of the paper).

Pre-step   cone radius rho_{k,c} from the local class counts (Eqs. 5-7).
Step 1     balanced classifier W^bal_k: encoder frozen, class-balanced subset,
           cross-entropy, one cone projection at the end (Section 4.3.1).
Step 2     local classifier W^pred_k: encoder frozen, all local data, classifier
           loss (Eq. 8), cone projection (Eqs. 9-11) after every SGD step (Section 4.3.2).
Step 3     encoder training: local classifier frozen, cross-entropy plus the
           prototype loss towards the group prototypes (Eq. 12, Section 4.3.3).

Server: encoder averaging, Step 4 (Section 4.4) and Steps 5-6 (Section 4.5).
"""

import numpy as np
import torch
from torch.nn import CosineSimilarity, CrossEntropyLoss
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD

from entities.base import Client, Server, clip_gradients
from utils.anchors import mean_prototypes, simplex_etf, update_anchors
from utils.averaging import average_prototypes, class_aware_average, plain_average, set_classifier_row
from utils.cone import cone_radius, project_to_cone, warmup_factor
from utils.gen_dataset import ClientDataset
from utils.grouping import group_clients


PROTOTYPE_BATCH_SIZE = 2048


class FedMCDClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.prototype_criterion = CrossEntropyLoss().to(cfg.device)
        self.anchors = None
        self.present_classes = set()
        self.missing_classes = set()
        self.label_distribution = None
        self.prototype_weight = 0.0
        self.local_protos = {}
        self.group_protos = []
        self.local_classifier = []

    # ----- shared helpers -------------------------------------------------

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

    def set_anchors(self, anchors):
        self.anchors = anchors.clone().to(self.device)

    def update_label_distribution(self):
        """n_{k,c}, the present classes S_k, the missing classes, and lambda_proto = H(nu_k)/gamma (Eq. 12)."""
        num_classes = self.cfg.num_classes
        targets = np.asarray(self.train_set.targets, dtype=np.int64).reshape(-1)
        self.label_distribution = np.bincount(targets, minlength=num_classes)[:num_classes]
        n_total = len(targets)
        if n_total > 0:
            prob = self.label_distribution / n_total
            with np.errstate(divide="ignore", invalid="ignore"):
                log_prob = np.where(prob > 0, np.log(prob), 0.0)
            entropy = -np.sum(prob * log_prob)
        else:
            entropy = 0.0
        gamma = self.cfg.prototype_gamma
        self.prototype_weight = entropy / gamma if gamma != 0 else 0.0
        self.present_classes = set(np.unique(targets).tolist())
        self.missing_classes = set(range(num_classes)) - self.present_classes

    def reset_missing_rows(self):
        """Rows of missing classes (c not in S_k) are set to their anchors a_c, bias 0."""
        with torch.no_grad():
            for c in self.missing_classes:
                self.model.fc.weight.data[c] = self.anchors[c].clone()
                self.model.fc.bias.data[c] = 0.0

    def mask_missing_gradients(self):
        """Gradients of missing classes are set to zero (Section 4.3, after Eq. 8)."""
        for p in self.classifier_parameters():
            if p.grad is not None:
                for c in self.missing_classes:
                    p.grad[c] = 0.0

    # ----- Pre-step: cone radius (Section 4.2) and cone projection (Eqs. 9-11) -----

    def cone_radii(self, round_index):
        """Pre-step: rho_{k,c} = rho_max * coverage factor * warm-up factor * frequency factor (Eq. 7)."""
        warmup = warmup_factor(int(round_index), self.cfg.warmup_rounds, self.cfg.ramp_rounds)
        radii, _ = cone_radius(
            self.label_distribution, self.cfg.num_classes,
            max_radius=self.cfg.max_cone_radius, kappa=self.cfg.frequency_kappa,
            coverage_classes=self.cfg.coverage_classes, warmup=warmup,
            device=self.device, dtype=self.model.fc.weight.dtype,
        )
        return radii

    def project_classifier(self, round_index):
        """Cone projection Pi_B(a_c, rho_{k,c}) of every classifier row and bias (Eqs. 9-11).

        Missing classes have rho = 0 and are set to their anchors. Without cone
        projection (ablation "unconstrained residual") only missing-class rows are reset.
        """
        radii = self.cone_radii(round_index)
        if self.cfg.cone_projection:
            project_to_cone(self.model.fc.weight, self.anchors, radii, bias=self.model.fc.bias)
        else:
            with torch.no_grad():
                for label in range(self.cfg.num_classes):
                    if label not in self.present_classes:
                        self.model.fc.weight[label].copy_(self.anchors[label])
                        self.model.fc.bias[label].zero_()
        return radii

    # ----- Step 1: balanced classifier W^bal_k (Section 4.3.1) ------------------

    def balanced_subset(self, rng):
        """D^bal_k: min(h_max, min_j n_{k,j}) samples of every present class (Step 1)."""
        labels = np.array(self.train_set.targets)
        present = sorted(self.present_classes)
        by_class = {c: np.where(labels == c)[0] for c in present}
        per_class = min(min(len(v) for v in by_class.values()), self.cfg.balanced_samples_per_class)
        indices = []
        for c in present:
            indices.extend(rng.choice(by_class[c], per_class, replace=False))
        rng.shuffle(indices)
        return ClientDataset(self.train_set, indices, self.train_set.transform)

    def train_balanced_classifier(self, round_index):
        """Step 1: W^bal_k on D^bal_k with the encoder frozen, then one cone projection (Eq. 11).

        The classifier starts from the server classifier (rows of missing classes set
        to a_c) and is trained with the cross-entropy term of Eq. 8 only.
        """
        self._train_balanced_classifier(round_index)
        self.project_classifier(round_index)

    def _train_balanced_classifier(self, round_index):
        if len(self.present_classes) == 0:
            return
        self.reset_missing_rows()
        self.model.train()
        seed = self.loader_seed(round_index, "balanced")
        subset = self.balanced_subset(np.random.RandomState(seed))
        if len(subset) == 0:
            return
        loader = self.seeded_loader("balanced", subset, self.cfg.batch_size, round_index)
        self.freeze(train_classifier=True)
        optimizer = SGD(self.classifier_parameters(), lr=self.cfg.balanced_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.balanced_epochs):
            for inputs, labels in loader:
                inputs, labels = self.to_device(inputs, labels)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(self.model.parameters(), self.cfg.max_grad_norm)
                self.mask_missing_gradients()
                optimizer.step()
        self.reset_missing_rows()

    # ----- Steps 2-3: local classifier (Section 4.3.2) and encoder (Section 4.3.3) -----

    def train(self, round_index):
        """Steps 2-3. Step 2 starts from the group classifier received from the server,
        projected onto this client's cones; Step 3 then computes the local prototypes Z_k."""
        self.set_classifier(self.local_classifier)
        self.model.train()
        loader = self.seeded_loader("train", self.train_set, self.cfg.batch_size, round_index)
        self.project_classifier(round_index)
        self.train_local_classifier(loader, round_index)
        self.train_encoder(loader, round_index)
        self.compute_prototypes(round_index)

    def train_local_classifier(self, loader, round_index):
        """Step 2: local classifier W^pred_k with the encoder frozen.

        Loss of Eq. 8 (cross-entropy plus (mu_res/2) * sum_c (|w_c - a_c|^2 + b_c^2) over
        present classes); after every SGD step the cone projection (Eqs. 9-11) is applied.
        """
        self.freeze(train_classifier=True)
        optimizer = SGD(self.classifier_parameters(), lr=self.cfg.classifier_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        mu = max(0.0, float(self.cfg.residual_penalty))
        weight, bias = self.model.fc.weight, self.model.fc.bias
        for _ in range(self.cfg.classifier_epochs):
            for inputs, labels in loader:
                inputs, labels = self.to_device(inputs, labels)
                loss = self.criterion(self.model(inputs), labels)
                if mu > 0.0:
                    penalty = 0.0
                    for label in self.present_classes:
                        penalty = penalty + (weight[label] - self.anchors[label]).pow(2).sum()
                    penalty = penalty + sum(bias[label].pow(2) for label in self.present_classes)
                    loss = loss + 0.5 * mu * penalty
                optimizer.zero_grad()
                loss.backward()
                self.mask_missing_gradients()
                clip_gradients(self.classifier_parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                self.project_classifier(round_index)

    def prototype_loss(self, features, labels, round_index, cos):
        """L_proto of Eq. 12: cross-entropy over cosine similarities to the group prototypes, temperature tau."""
        if (not self.cfg.prototype_alignment or len(self.group_protos) == 0
                or round_index < self.cfg.prototype_start_round or self.prototype_weight == 0):
            return None
        logits = cos(features.unsqueeze(1), torch.stack(self.group_protos).repeat(len(labels), 1, 1))
        return self.prototype_criterion(logits / self.cfg.prototype_temperature, labels)

    def train_encoder(self, loader, round_index):
        """Step 3: encoder theta_k with W^pred_k frozen; loss of Eq. 12 (cross-entropy + lambda_proto * L_proto)."""
        self.freeze(train_classifier=False)
        optimizer = SGD(self.encoder_parameters(), lr=self.cfg.encoder_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        cos = CosineSimilarity(dim=2).to(self.device)
        for _ in range(self.cfg.encoder_epochs):
            for inputs, labels in loader:
                inputs, labels = self.to_device(inputs, labels)
                outputs, features = self.model(inputs, True)
                loss = self.criterion(outputs, labels)
                proto_loss = self.prototype_loss(features, labels, round_index, cos)
                if proto_loss is not None:
                    loss = loss + self.prototype_weight * proto_loss
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(self.encoder_parameters(), self.cfg.max_grad_norm)
                optimizer.step()

    def compute_prototypes(self, round_index):
        """Local prototypes zeta_{k,c} (mean feature per present class), uploaded for Steps 4 and 6."""
        with torch.no_grad():
            loader = self.seeded_loader("prototype", self.train_set, PROTOTYPE_BATCH_SIZE, round_index)
            num_classes = self.cfg.num_classes
            sums = torch.zeros(num_classes, self.model.fc.in_features, device=self.device)
            counts = torch.zeros(num_classes, device=self.device)
            self.model.eval()
            for inputs, labels in loader:
                inputs, labels = self.to_device(inputs, labels)
                _, features = self.model(inputs, True)
                sums.index_add_(0, labels, features)
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
            self.local_protos = {label: sums[label] / counts[label]
                                 for label in range(num_classes) if counts[label].item() > 0}
            if len(self.group_protos) != num_classes:
                self.group_protos = [self.anchors[label].clone() for label in range(num_classes)]
            for label in self.present_classes:
                if label in self.local_protos:
                    self.group_protos[label] = self.local_protos[label]


class FedMCDServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.anchors = simplex_etf(cfg.num_classes, self.model.fc.in_features, cfg.device)
        self.initial_anchors = self.anchors.clone()

    def encoder_parameters(self):
        return list(self.model.hidden_layers.parameters())

    def send_anchors(self, clients):
        for client in clients:
            client.set_anchors(self.anchors)

    def send_encoder(self, clients):
        params = self.encoder_parameters()
        for client in clients:
            client.set_encoder(params)

    def aggregate_encoders(self, clients):
        """theta^(t+1) = sum_k omega_k theta_k with omega_k proportional to |D_k|."""
        params = self.encoder_parameters()
        new_params = torch.zeros_like(parameters_to_vector(params))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            new_params += client_size * parameters_to_vector(client.encoder_parameters())
        new_params /= total_size
        vector_to_parameters(new_params, params)

    def update_anchors(self, clients, presence, round_index):
        """Step 4 (Eqs. 13-16), applied every Delta T_anchor rounds after T_warm warm-up rounds."""
        cfg = self.cfg
        if (not cfg.anchor_update or round_index < cfg.anchor_warmup_rounds
                or round_index % max(1, int(cfg.anchor_update_every)) != 0):
            return False
        prototypes, supports = mean_prototypes(clients, presence, self.anchors)
        self.anchors, updated = update_anchors(
            self.initial_anchors, self.anchors, prototypes, supports,
            gram_weight=cfg.anchor_gram_weight, ema=cfg.anchor_ema,
            iterations=cfg.anchor_iterations, min_clients=cfg.anchor_min_clients,
            max_residual=cfg.anchor_max_residual, max_condition=cfg.anchor_max_condition,
        )
        return updated

    def group_clients(self, balanced_classifiers, presence):
        """Step 5 on the balanced classifiers W^bal_k, using the updated anchors A^(t+1)."""
        weights = {cid: params[0] for cid, params in balanced_classifiers.items()}
        return group_clients(weights, presence, self.anchors, self.cfg.dbscan_eps,
                             on_residuals=self.cfg.group_on_residuals)

    def average_groups(self, groups, selected, presence):
        """Step 6 (Eq. 19): give every group member the group classifier row and group prototype."""
        num_classes = self.cfg.num_classes
        for label, id_groups in groups.items():
            for ids in id_groups:
                group = [client for client in selected if client.id in ids]
                if self.cfg.class_aware_averaging:
                    row = class_aware_average(self.model, self.anchors, label, group, presence)
                else:
                    row = plain_average(self.model, label, group)
                proto = average_prototypes(label, group, presence)
                for client in group:
                    set_classifier_row(client.model, label, row)
                    if proto is not None and len(client.group_protos) == num_classes:
                        client.group_protos[label] = proto.clone()
