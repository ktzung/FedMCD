"""Ditto (Li et al., ICML 2021): FedAvg plus a personalized model kept close to the global model.

    python methods/Ditto.py --config configs/Ditto.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.Ditto import DittoClient, DittoServer
from utils.drift import apply_drift
from utils.experiment import evaluate_client_models, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, DittoClient, DittoServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.train(0)  # the original calls train() without a round: the round-0 loader seed every round
            client.train_personal(server.model)
        server.aggregate_by_updates(selected)
        log(round_index, *evaluate_client_models(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("ditto", train)
