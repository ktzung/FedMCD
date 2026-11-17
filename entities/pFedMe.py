"""pFedMe (T Dinh et al., NeurIPS 2020): personalized models via Moreau envelopes."""

from copy import deepcopy

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from entities.base import Client, Server


class pFedMeOptimizer(Optimizer):
    """Inner step on the gradient plus lambda * (theta - w_local); returns the parameters."""

    def __init__(self, params, lr=0.01, lam=0.1, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, lam=lam, momentum=momentum, weight_decay=weight_decay))

    def step(self, local_params):
        group = None
        for group in self.param_groups:
            for p, local in zip(group["params"], local_params):
                if p.grad is None:
                    continue
                if group["weight_decay"] != 0:
                    p.grad.data.add_(p.data, alpha=group["weight_decay"])
                grad = p.grad.data + group["lam"] * (p.data - local.data)
                if group["momentum"] != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(grad)
                    grad = buf
                p.data.add_(grad, alpha=-group["lr"])
        return group["params"]


class pFedMeClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.local_params = deepcopy(list(self.model.parameters()))

    def train_personalized(self):
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.cfg.personal_lr, lam=self.cfg.pfedme_lambda)
        for _ in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                for _ in range(self.cfg.inner_steps):
                    loss = self.criterion(self.model(inputs), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    personalized = optimizer.step(self.local_params)
                for new, local in zip(personalized, self.local_params):
                    local.data -= self.cfg.lr * self.cfg.pfedme_lambda * (local.data - new.data)
        self.set_params(self.local_params)


class pFedMeServer(Server):
    def aggregate_by_params(self, clients):
        """w = (1 - beta) w_previous + beta * (sample-size weighted average of the clients)."""
        previous = deepcopy(parameters_to_vector(self.model.parameters()).detach())
        new_params = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            new_params += client_size * parameters_to_vector(client.model.parameters())
        new_params /= total_size
        vector_to_parameters((1 - self.cfg.beta) * previous + self.cfg.beta * new_params, self.model.parameters())
