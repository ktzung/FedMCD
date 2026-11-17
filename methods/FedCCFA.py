"""FedCCFA (Chen et al., NeurIPS 2024): classifier clustering and feature alignment.

    python methods/FedCCFA.py --config configs/FedCCFA.yaml [--class-support-mask] [options]
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from entities.FedCCFA import FedCCFAClient, FedCCFAServer
from utils.averaging import set_classifier_row
from utils.drift import apply_drift
from utils.experiment import evaluate_personalized, prepare_data, run_experiment
from utils.gen_dataset import draw_missing_classes


def train(cfg, log):
    client_train, client_test, global_tests = prepare_data(cfg)
    clients = [FedCCFAClient(k, cfg, client_train[k], client_test[k]) for k in range(cfg.num_clients)]
    draw_missing_classes(clients, cfg.missing_prob, cfg.seed, cfg.num_classes)
    server = FedCCFAServer(cfg)
    server.record_client_sizes(clients)

    for round_index in range(cfg.rounds):
        apply_drift(clients, global_tests, round_index, cfg.drift, cfg.drift_fraction)
        selected = server.select_clients(clients)
        server.send_params(selected)
        for client in selected:
            if not client.local_classifier:
                client.local_classifier = client.copy_classifier()

        balanced = {}
        for client in selected:
            client.update_label_distribution()
            client.train_balanced_classifier()
            balanced[client.id] = client.copy_classifier()
            client.train(round_index)

        server.aggregate_encoders(selected)
        server.aggregate_prototypes(selected)
        server.send_encoder(selected)

        support = {client.id: {int(c) for c in range(cfg.num_classes) if client.label_distribution[c] > 0}
                   for client in selected}
        groups = server.group_clients(balanced, support)
        for label, id_groups in groups.items():
            for ids in id_groups:
                group = [client for client in selected if client.id in ids]
                if cfg.class_support_mask:
                    group = [client for client in group if client.label_distribution[label] > 0]
                    if len(group) < 2:
                        continue
                row = server.average_rows(label, group)
                proto = server.average_group_prototypes(label, group)
                for client in group:
                    set_classifier_row(client.model, label, row)
                    client.global_protos[label] = proto.clone()
        for client in selected:
            client.local_classifier = client.copy_classifier()

        log(round_index, *evaluate_personalized(server, clients, selected, global_tests))
    return clients, server, {}


if __name__ == "__main__":
    run_experiment("fedccfa", train)
