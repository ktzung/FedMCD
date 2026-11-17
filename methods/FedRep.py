"""FedRep (Collins et al., ICML 2021): shared encoder, local classifiers.

    python methods/FedRep.py --config configs/FedRep.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.FedRep import FedRepClient, FedRepServer
from utils.drift import apply_drift
from utils.experiment import evaluate_client_models, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, FedRepClient, FedRepServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_encoder(selected)
        for client in selected:
            client.train_decoupled()
        server.aggregate_encoders(selected)
        server.send_encoder(selected)
        log(round_index, *evaluate_client_models(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("fedrep", train)
