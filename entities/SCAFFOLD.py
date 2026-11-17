"""SCAFFOLD (Karimireddy et al., ICML 2020)."""

from copy import deepcopy

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from entities.base import Client, Server


class SCAFFOLDOptimizer(Optimizer):
    """SGD on the gradient corrected by the control variates, g - c_i + c."""

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, global_c, client_c):
        for group in self.param_groups:
            for p, c, ci in zip(group["params"], global_c, client_c):
                if p.grad is None:
                    continue
                if group["weight_decay"] != 0:
                    p.grad.data.add_(p.data, alpha=group["weight_decay"])
                grad = p.grad.data - ci + c
                if group["momentum"] != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(grad, alpha=1)
                    grad = buf
                p.data.add_(grad, alpha=-group["lr"])


class SCAFFOLDClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.local_c = [torch.zeros_like(p).to(cfg.device) for p in self.model.parameters()]
        self.local_c_update = []

    def train_with_control(self, global_c):
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        init_params = parameters_to_vector(self.model.parameters())
        start_params = deepcopy(list(self.model.parameters()))
        optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.cfg.lr,
                                      weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(global_c, self.local_c)
        with torch.no_grad():
            steps = len(loader) * self.cfg.local_epochs
            self.local_c_update = []
            for c, ci, x, yi in zip(global_c, self.local_c, start_params, self.model.parameters()):
                ci.data = ci - c + 1 / (steps * self.cfg.lr) * (x - yi)
                self.local_c_update.append(-c + 1 / (steps * self.cfg.lr) * (x - yi))
            self.local_update = parameters_to_vector(self.model.parameters()) - init_params


class SCAFFOLDServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_c = [torch.zeros_like(p).to(cfg.device) for p in self.model.parameters()]

    def aggregate_by_updates(self, clients):
        """Uniform average of the client updates, scaled by the global learning rate."""
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            update_sum += client.local_update
        new_params = parameters_to_vector(self.model.parameters()) + self.cfg.global_lr * (update_sum / len(clients))
        vector_to_parameters(new_params, self.model.parameters())

    def update_global_c(self, clients):
        for client in clients:
            for c, update in zip(self.global_c, client.local_c_update):
                c.data += update.data.clone() / self.cfg.num_clients
