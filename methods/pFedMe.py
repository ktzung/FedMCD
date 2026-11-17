"""pFedMe (T Dinh et al., NeurIPS 2020): personalized models via Moreau envelopes.

    python methods/pFedMe.py --config configs/pFedMe.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from copy import deepcopy

from entities.pFedMe import pFedMeClient, pFedMeServer
from utils.drift import apply_drift
from utils.experiment import evaluate_client_models, run_experiment, setup


def train(cfg, log):
    clients, server, global_tests = setup(cfg, pFedMeClient, pFedMeServer)
    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            client.local_params = deepcopy(list(server.model.parameters()))
            client.train_personalized()
        server.aggregate_by_params(selected)
        log(round_index, *evaluate_client_models(server, clients, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("pfedme", train)
