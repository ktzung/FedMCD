"""SCAFFOLD (Karimireddy et al., ICML 2020).

    python methods/SCAFFOLD.py --config configs/SCAFFOLD.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.SCAFFOLD import SCAFFOLDClient, SCAFFOLDServer
from utils.drift import apply_drift
from utils.experiment import evaluate_global_model, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, SCAFFOLDClient, SCAFFOLDServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.train_with_control(server.global_c)
        server.aggregate_by_updates(selected)
        server.update_global_c(selected)
        server.send_params(selected)
        log(round_index, *evaluate_global_model(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("scaffold", train)
