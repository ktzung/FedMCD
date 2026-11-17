"""FedMCD training loop (Algorithm 1 of the paper).

One communication round:
  server broadcast: shared encoder, anchor matrix A^(t), group classifier and prototypes;
  each client k:    Pre-step (cone radius, Section 4.2) and Steps 1-3 (Section 4.3);
  server:           encoder averaging, Step 4 (anchor update, Section 4.4),
                    Step 5 (grouping) and Step 6 (class-aware averaging, Section 4.5).
``round_index`` is 0-based; round t of the paper is ``round_index + 1``.
Equation numbers refer to the current version of the manuscript.

    python methods/FedMCD.py --config configs/FedMCD.yaml [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.FedMCD import FedMCDClient, FedMCDServer
from utils.diagnostics import Diagnostics
from utils.drift import apply_drift
from utils.experiment import evaluate_personalized, prepare_data, run_experiment
from utils.gen_dataset import draw_missing_classes


def train(cfg, log):
    client_train, client_test, global_tests = prepare_data(cfg)
    clients = [FedMCDClient(k, cfg, client_train[k], client_test[k]) for k in range(cfg.num_clients)]
    server = FedMCDServer(cfg)
    draw_missing_classes(clients, cfg.missing_prob, cfg.seed, cfg.num_classes)
    for client in clients:
        client.update_label_distribution()
    server.record_client_sizes(clients)
    server.send_anchors(clients)
    diagnostics = Diagnostics(cfg.num_classes) if cfg.diagnostics else None

    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        # Broadcast: encoder theta^(t) and server classifier, anchors A^(t).
        server.send_params(selected)
        server.send_anchors(selected)
        for client in selected:
            if not client.local_classifier:
                client.local_classifier = client.copy_classifier()

        # Client side: Pre-step and Steps 1-3; uploads W^bal_k, W^pred_k, theta_k, Z_k, S_k.
        balanced, presence = {}, {}
        for client in selected:
            client.update_label_distribution()                 # n_{k,c} and S_k
            presence[client.id] = set(client.present_classes)
            client.train_balanced_classifier(round_index)      # Step 1 -> W^bal_k
            balanced[client.id] = client.copy_classifier()
            client.train(round_index)                          # Steps 2-3 -> W^pred_k, theta_k, Z_k
            if diagnostics:
                diagnostics.record_client(client, client.cone_radii(round_index))

        # Server side.
        server.aggregate_encoders(selected)                    # theta^(t+1)
        server.send_encoder(selected)
        server.update_anchors(selected, presence, round_index)  # Step 4 -> A^(t+1)
        groups, assignments = server.group_clients(balanced, presence)  # Step 5
        if diagnostics:
            diagnostics.end_round(round_index, assignments, presence, balanced, server.anchors)
        server.average_groups(groups, selected, presence)      # Step 6 -> group classifier and prototypes
        for client in selected:
            # Group classifier: used for prediction and as the start of Step 2 in the next round.
            client.local_classifier = client.copy_classifier()

        log(round_index, *evaluate_personalized(server, clients, selected, global_tests))

    extras = {"diagnostics": diagnostics.rows} if diagnostics else {}
    return clients, server, extras


if __name__ == "__main__":
    run_experiment("fedmcd", train)
