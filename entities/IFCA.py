"""IFCA (Ghosh et al., NeurIPS 2020): each client joins the global model with the lowest training loss."""

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from entities.base import Client, Server
from utils.metrics import average_loss
from utils.models import build_model


class IFCAClient(Client):
    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.cluster_identity = 0

    def choose_cluster(self, global_models):
        losses = [average_loss(model, self.train_set, self.criterion, self.device) for model in global_models]
        self.cluster_identity = np.array(losses).argmin(0)
        self.set_params(global_models[self.cluster_identity].parameters())


class IFCAServer(Server):
    """Keeps one global model per cluster."""

    def __init__(self, cfg, num_models=None):
        super().__init__(cfg)
        num_models = cfg.num_clusters if num_models is None else num_models
        self.global_models = [build_model(cfg.dataset, cfg.num_classes).to(cfg.device) for _ in range(num_models)]

    def send_params(self, clients):
        for client in clients:
            client.set_params(self.global_models[client.cluster_identity].parameters())

    def aggregate_with_clustering(self, clients):
        """Sample-size weighted average of the clients of each cluster."""
        groups = {identity: [] for identity in range(len(self.global_models))}
        for client in clients:
            groups[client.cluster_identity].append(client)
        for identity, global_model in enumerate(self.global_models):
            if len(groups[identity]) == 0:
                continue
            total_size = 0
            new_params = torch.zeros_like(parameters_to_vector(global_model.parameters()))
            for client in groups[identity]:
                client_size = self.client_data_size[client.id]
                total_size += client_size
                new_params += client_size * parameters_to_vector(client.model.parameters())
            new_params /= total_size
            vector_to_parameters(new_params, global_model.parameters())
