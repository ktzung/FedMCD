"""FedBABU (Oh et al., ICLR 2022): the encoder is trained with a fixed random classifier,

    python methods/FedBABU.py --config configs/FedBABU.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.FedBABU import FedBABUClient, FedBABUServer
from utils.drift import apply_drift
from utils.experiment import run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, FedBABUClient, FedBABUServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.train_encoder()
        server.aggregate_encoders(selected)
        server.send_params(selected)
        for client in clients:
            client.fine_tune()
        log(round_index, server.local_evaluate(clients), server.evaluate_each_client(clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("fedbabu", train)
