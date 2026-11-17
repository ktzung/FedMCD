"""FedFM (Ye et al., IEEE TSP 2023): feature matching to global class anchors.

    python methods/FedFM.py --config configs/FedFM.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.FedFM import FedFMClient, FedFMServer
from utils.drift import apply_drift
from utils.experiment import evaluate_global_model, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, FedFMClient, FedFMServer)
    for round_index in range(cfg.rounds):
        anchor_weight = cfg.anchor_weight if round_index >= cfg.anchor_start_round else 0
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.update_label_distribution()
            client.train_with_anchors(server.global_anchors, anchor_weight)
        server.aggregate_by_params(selected)
        server.aggregate_anchors(selected)
        server.send_params(selected)
        log(round_index, *evaluate_global_model(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("fedfm", train)
