"""FedGELA (Fan et al., NeurIPS 2023): fixed simplex ETF classifier with local class-balance weights."""

from collections import Counter

import torch
from torch.nn import NLLLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, Server, clip_gradients
from utils.anchors import simplex_etf


class FedGELAClient(Client):
    """Trains only the encoder; the classifier is the frozen ETF with zero bias."""

    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.etf = None
        self.phi = None
        self.criterion = NLLLoss().to(cfg.device)

    def set_etf(self, etf):
        self.etf = etf.clone().to(self.device)
        with torch.no_grad():
            self.model.fc.weight.data.copy_(self.etf)
            self.model.fc.bias.data.zero_()
        for p in self.model.fc.parameters():
            p.requires_grad = False

    def set_params(self, encoder_params):
        for new, local in zip(encoder_params, self.model.hidden_layers.parameters()):
            local.data = new.data.clone()

    def compute_phi(self):
        """phi_c = C * n_c / n (0 for missing classes)."""
        num_classes = self.cfg.num_classes
        targets = [int(t) for t in self.train_set.targets]
        counts = Counter(targets)
        phi = torch.zeros(num_classes, device=self.device)
        if targets:
            for c, n_c in counts.items():
                if 0 <= c < num_classes:
                    phi[c] = (n_c / len(targets)) * num_classes
        self.phi = phi

    def train(self, round_index):
        self.compute_phi()
        for p in self.model.fc.parameters():
            p.requires_grad = False
        for p in self.model.hidden_layers.parameters():
            p.requires_grad = True
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = SGD(self.model.hidden_layers.parameters(), lr=self.cfg.lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, features = self.model(inputs, return_features=True)
                logits = (features @ self.etf.T) * self.phi.unsqueeze(0)
                log_p = torch.nn.functional.log_softmax(logits / self.cfg.temperature, dim=1)
                loss = self.criterion(log_p, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(self.model.hidden_layers.parameters(), self.cfg.max_grad_norm)
                optimizer.step()


class FedGELAServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.etf = simplex_etf(cfg.num_classes, self.model.fc.in_features, cfg.device)

    def send_params(self, clients):
        encoder = list(self.model.hidden_layers.parameters())
        for client in clients:
            client.set_etf(self.etf)
            client.set_params(encoder)

    def aggregate_by_params(self, clients):
        """Sample-size weighted average of the encoders; the ETF classifier is not aggregated."""
        total_size = sum(self.client_data_size[c.id] for c in clients)
        for p in self.model.hidden_layers.parameters():
            p.data.zero_()
        for client in clients:
            w = self.client_data_size[client.id] / total_size
            for sp, cp in zip(self.model.hidden_layers.parameters(), client.model.hidden_layers.parameters()):
                sp.data += w * cp.data
