"""FedGELA (Fan et al., NeurIPS 2023): fixed simplex ETF classifier with local class-balance weights.

    python methods/FedGELA.py --config configs/FedGELA.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.FedGELA import FedGELAClient, FedGELAServer
from utils.drift import apply_drift
from utils.experiment import prepare_data, run_experiment
from utils.gen_dataset import draw_missing_classes


def train(cfg, log):
    client_train, client_test, global_tests = prepare_data(cfg)
    clients = [FedGELAClient(k, cfg, client_train[k], client_test[k]) for k in range(cfg.num_clients)]
    draw_missing_classes(clients, cfg.missing_prob, cfg.seed, cfg.num_classes)
    server = FedGELAServer(cfg)
    server.record_client_sizes(clients)
    server.send_params(clients)

    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.train(round_index)
        server.aggregate_by_params(selected)
        server.send_params(clients)
        local = server.local_evaluate(clients)
        log(round_index, local, server.evaluate_shared_model(clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("fedgela", train)
