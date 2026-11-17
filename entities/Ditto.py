"""Ditto (Li et al., ICML 2021): FedAvg plus a personalized model kept close to the global model."""

from copy import deepcopy

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from entities.base import Client, Server
from utils.metrics import accuracy


class DittoOptimizer(Optimizer):
    """SGD on the gradient plus lambda * (w - w_global)."""

    def __init__(self, params, lr, lam=0.0, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, lam=lam, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group["params"], global_params):
                if p.grad is None:
                    continue
                if group["weight_decay"] != 0:
                    p.grad.data.add_(p.data, alpha=group["weight_decay"])
                grad = p.grad.data + group["lam"] * (p.data - g.data)
                if group["momentum"] != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(grad)
                    grad = buf
                p.data.add_(grad, alpha=-group["lr"])


class DittoClient(Client):
    """Evaluated with its personalized model."""

    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.personal_model = deepcopy(self.model)

    def train_personal(self, global_model):
        global_params = [p.data.clone() for p in global_model.parameters()]
        self.personal_model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = DittoOptimizer(self.personal_model.parameters(), lr=self.cfg.lr, lam=self.cfg.personal_lambda,
                                   weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.personal_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.personal_model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(global_params)

    def local_accuracy(self):
        return accuracy(self.personal_model, self.test_set, self.device)

    def test_accuracy(self, global_tests):
        return accuracy(self.personal_model, global_tests[self.global_test_id], self.device)


class DittoServer(Server):
    """Sample-size weighted average of the client updates of the global model."""
