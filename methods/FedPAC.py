"""FedPAC (Xu et al., ICLR 2023): feature alignment to global prototypes and classifier collaboration.

    python methods/FedPAC.py --config configs/FedPAC.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from copy import deepcopy

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from entities.base import classifier_parameters
from entities.FedPAC import FedPACClient, FedPACServer
from utils.drift import apply_drift
from utils.experiment import evaluate_client_models, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, FedPACClient, FedPACServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        variances, means = [], []
        selected = server.select_clients(clients)
        server.send_encoder(selected)
        for client in selected:
            client.update_label_distribution()
            v, h = client.feature_statistics()
            variances.append(deepcopy(v))
            means.append(deepcopy(h))
            client.train_with_protos(server.global_protos)
        server.aggregate_encoders(selected)
        server.aggregate_protos(selected)
        server.send_encoder(selected)
        weights = server.combination_weights(variances, means, len(selected))
        all_params = [parameters_to_vector(classifier_parameters(c.model)) for c in selected]
        for index, client in enumerate(selected):
            vector_to_parameters(server.combine_classifiers(all_params, weights[index]),
                                 classifier_parameters(client.model))
        log(round_index, *evaluate_client_models(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("fedpac", train)
