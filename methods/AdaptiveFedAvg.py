"""Adaptive-FedAvg (Canonaco et al., IJCNN 2021).

    python methods/AdaptiveFedAvg.py --config configs/AdaptiveFedAvg.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.AdaptiveFedAvg import AdaptiveFedAvgClient, AdaptiveFedAvgServer
from utils.drift import apply_drift
from utils.experiment import evaluate_global_model, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, AdaptiveFedAvgClient, AdaptiveFedAvgServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.train(0)  # the original calls train() without a round: the round-0 loader seed every round
        server.aggregate_by_params(selected)
        server.send_params(selected)
        client_lr = server.adaptive_lr(round_index)
        for client in selected:
            client.lr = client_lr
        log(round_index, *evaluate_global_model(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("adaptive_fedavg", train)
