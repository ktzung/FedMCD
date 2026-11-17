"""IFCA (Ghosh et al., NeurIPS 2020): each client joins the global model with the lowest training loss.

    python methods/IFCA.py --config configs/IFCA.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.IFCA import IFCAClient, IFCAServer
from utils.drift import apply_drift
from utils.experiment import evaluate_client_models, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, IFCAClient, IFCAServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        for client in selected:
            client.choose_cluster(server.global_models)
            client.train(0)  # the original calls train() without a round: the round-0 loader seed every round
        server.aggregate_with_clustering(selected)
        server.send_params(selected)
        server.send_params(clients)
        log(round_index, *evaluate_client_models(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("ifca", train)
