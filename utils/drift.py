"""Concept drift by label swaps (FedCCFA protocol).

Clients are split by ``client_id % 10`` into three groups that swap classes
(1, 2), (3, 4) and (5, 6). Only the first ``round(fraction * K)`` clients drift.
Copies 1-3 of the global test set receive the corresponding swaps so that each
client is evaluated under its current concept.
"""

import numpy as np

DRIFT_PATTERNS = ("none", "sudden", "incremental", "recurrent")
SUDDEN_ROUNDS = (100,)
RECURRENT_ROUNDS = (100, 150)
INCREMENTAL_SWAPS = {100: (1, 2), 110: (3, 4), 120: (5, 6)}
INCREMENTAL_TEST_COPY = {100: 1, 110: 2, 120: 3}


def swap_labels(dataset, class_a, class_b):
    """Swap two labels in place."""
    class_a_indices = np.where(np.array(dataset.targets) == class_a)[0]
    class_b_indices = np.where(np.array(dataset.targets) == class_b)[0]
    for i in class_a_indices:
        dataset.targets[i] = -1
    for i in class_b_indices:
        dataset.targets[i] = class_a
    class_a_indices = np.where(np.array(dataset.targets) == -1)[0]
    for i in class_a_indices:
        dataset.targets[i] = class_b


def drifting_client_ids(clients, fraction):
    per_mille = max(0, min(1000, int(round(float(fraction) * 1000))))
    n_drift = int(round(len(clients) * per_mille / 1000.0))
    ordered = sorted(clients, key=lambda client: client.id)
    return {client.id for client in ordered[:n_drift]}


def sudden_drift(clients, global_tests, fraction):
    drifting = drifting_client_ids(clients, fraction)
    if not drifting:
        return False
    swap_labels(global_tests[1], 1, 2)
    swap_labels(global_tests[2], 3, 4)
    swap_labels(global_tests[3], 5, 6)
    for client in clients:
        if client.id not in drifting:
            continue
        if client.id % 10 < 3:
            pair, test_copy = (1, 2), 1
        elif client.id % 10 < 6:
            pair, test_copy = (3, 4), 2
        else:
            pair, test_copy = (5, 6), 3
        swap_labels(client.train_set, *pair)
        swap_labels(client.test_set, *pair)
        client.global_test_id = test_copy
    return True


def incremental_drift(clients, global_tests, round_index, fraction):
    drifting = drifting_client_ids(clients, fraction)
    if not drifting:
        return False
    test_copy = INCREMENTAL_TEST_COPY[round_index]
    pair = INCREMENTAL_SWAPS[round_index]
    swap_labels(global_tests[test_copy], *pair)
    for client in clients:
        if client.id not in drifting:
            continue
        bucket = client.id % 10
        in_stage = ((round_index == 100 and bucket < 3)
                    or (round_index == 110 and 3 <= bucket < 6)
                    or (round_index == 120 and bucket >= 6))
        if not in_stage:
            continue
        swap_labels(client.train_set, *pair)
        swap_labels(client.test_set, *pair)
        client.global_test_id = test_copy
    return True


def apply_drift(clients, global_tests, round_index, pattern, fraction):
    """Apply the drift event scheduled at ``round_index`` (0-based), if any.

    Under recurrent drift the second sudden swap restores the original labels.
    """
    if pattern == "sudden" and round_index in SUDDEN_ROUNDS:
        return sudden_drift(clients, global_tests, fraction)
    if pattern == "recurrent" and round_index in RECURRENT_ROUNDS:
        return sudden_drift(clients, global_tests, fraction)
    if pattern == "incremental" and round_index in INCREMENTAL_SWAPS:
        return incremental_drift(clients, global_tests, round_index, fraction)
    return False
