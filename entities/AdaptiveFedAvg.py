"""Adaptive-FedAvg (Canonaco et al., IJCNN 2021).

As in the original experiment code, the clients run the FedAvg update with the
configured learning rate: the adaptive rate is computed by the server and stored
on the clients, but the FedAvg client does not read it.
"""

import numpy as np
from torch.nn.utils import parameters_to_vector

from entities.base import Client, Server


class AdaptiveFedAvgServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.prev_mean = 0.0
        self.prev_mean_norm = 0.0
        self.prev_variance = 0.0
        self.prev_variance_norm = 0.0
        self.prev_ratio = 0.0

    def adaptive_lr(self, round_index):
        t = round_index + 1
        beta1, beta2, beta3 = self.cfg.beta1, self.cfg.beta2, self.cfg.beta3
        params = parameters_to_vector(self.model.parameters()).detach().cpu().numpy()
        mean = beta1 * self.prev_mean + (1 - beta1) * params
        mean_norm = mean / (1 - pow(beta1, t))
        variance = beta2 * self.prev_variance + (1 - beta2) * np.mean(
            (params - self.prev_mean_norm) * (params - self.prev_mean_norm))
        variance_norm = variance / (1 - pow(beta2, t))
        if t == 1:
            ratio = beta3 * self.prev_ratio + (1 - beta3)
        else:
            ratio = beta3 * self.prev_ratio + (1 - beta3) * (variance_norm / self.prev_variance_norm)
        ratio_norm = ratio / (1 - pow(beta3, t))
        self.prev_mean, self.prev_mean_norm = mean, mean_norm
        self.prev_variance, self.prev_variance_norm = variance, variance_norm
        self.prev_ratio = ratio
        return min(self.cfg.lr, self.cfg.lr * ratio_norm / t)


class AdaptiveFedAvgClient(Client):
    """FedAvg client update."""
