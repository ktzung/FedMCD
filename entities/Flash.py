"""Flash (Panchal et al., ICML 2023): early-stopped local training and drift-aware adaptive server update."""

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, Server
from utils.metrics import average_loss


class FlashClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.prev_val_loss = -1

    def train_early_stopping(self):
        """Stop when the validation loss decreases by less than loss_decrement / epoch."""
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        init_params = parameters_to_vector(self.model.parameters())
        optimizer = SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        momentum=self.cfg.momentum)
        for epoch in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = average_loss(self.model, self.test_set, self.criterion, self.device)
            if self.prev_val_loss != -1:
                delta = self.prev_val_loss - val_loss
                if 0 < delta < self.cfg.loss_decrement / (epoch + 1):
                    break
            self.prev_val_loss = val_loss
        self.local_update = parameters_to_vector(self.model.parameters()) - init_params


class FlashServer(Server):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.beta3 = 0
        self.first_momentum = 0
        self.second_momentum = cfg.tau ** 2
        self.prev_second_momentum = 0
        self.delta_momentum = 0

    def aggregate_by_updates(self, clients):
        total_size = 0
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            update_sum += client_size * client.local_update
        update = (update_sum / total_size).detach().cpu().numpy()
        beta1, beta2, tau = self.cfg.beta1, self.cfg.beta2, self.cfg.tau
        self.first_momentum = beta1 * self.first_momentum + (1 - beta1) * update
        self.prev_second_momentum = self.second_momentum
        self.second_momentum = beta2 * self.second_momentum + (1 - beta2) * np.square(update)
        self.beta3 = np.abs(self.prev_second_momentum) / (
            np.abs(np.square(update) - self.second_momentum) + np.abs(self.prev_second_momentum))
        self.delta_momentum = self.beta3 * self.delta_momentum + (1 - self.beta3) * (
            np.square(update) - self.second_momentum)
        step = self.cfg.server_lr * self.first_momentum / (np.sqrt(self.second_momentum) - self.delta_momentum + tau)
        new_params = parameters_to_vector(self.model.parameters()) + torch.tensor(step).to(self.device)
        vector_to_parameters(new_params, self.model.parameters())
