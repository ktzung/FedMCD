"""FedDrift (Jothimurugesan et al., AISTATS 2023): drift detection, new clusters and cluster merging.

    python methods/FedDrift.py --config configs/FedDrift.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from copy import deepcopy

from entities.FedDrift import FedDriftClient, FedDriftServer
from utils.drift import apply_drift
from utils.experiment import evaluate_client_models, run_experiment, setup
from utils.models import build_model


def train(cfg, log):
    clients, server, global_tests = setup(cfg, FedDriftClient, FedDriftServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        for client in selected:
            client.detect_and_choose(server.global_models)
        if len(server.global_models) > 1:
            server.merge_clusters(selected)
        for client in selected:
            if client.cluster_identity is None:
                server.global_models.append(build_model(cfg.dataset, cfg.num_classes).to(cfg.device))
                client.cluster_identity = len(server.global_models) - 1
        server.send_params(selected)
        for client in selected:
            client.train(0)  # the original calls train() without a round: the round-0 loader seed every round
        server.aggregate_with_clustering(selected)
        server.send_params(selected)
        for client in clients:
            client.prev_train_set = deepcopy(client.train_set)
        server.send_params(clients)
        log(round_index, *evaluate_client_models(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("feddrift", train)
