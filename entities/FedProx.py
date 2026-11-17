"""FedProx (Li et al., MLSys 2020)."""

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from entities.base import Client, Server, clip_gradients


class PerturbedGradientDescent(Optimizer):
    """SGD with momentum whose update also contains mu * (w - w_global)."""

    def __init__(self, params, lr, mu=0.0, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, mu=mu, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group["params"], global_params):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["weight_decay"] != 0:
                    d_p.add_(p.data, alpha=group["weight_decay"])
                if group["momentum"] != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).add_(d_p)
                    d_p = buf
                g = g.to(device)
                d_p.add_(group["mu"] * (p.data - g.data))
                p.data.add_(d_p, alpha=-group["lr"])


class FedProxClient(Client):
    def train(self, round_index):
        global_params = [param.data.clone() for param in self.model.parameters()]
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        optimizer = PerturbedGradientDescent(self.model.parameters(), lr=self.cfg.lr, mu=self.cfg.mu,
                                             weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        for _ in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                clip_gradients([p for p in self.model.parameters() if p.requires_grad], self.cfg.max_grad_norm)
                optimizer.step(global_params, self.device)


class FedProxServer(Server):
    """Sample-size weighted averaging (FedAvg)."""
