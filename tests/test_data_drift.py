from types import SimpleNamespace

import numpy as np

from utils.drift import apply_drift
from utils.gen_dataset import draw_missing_classes


class TinyDataset:
    def __init__(self, labels):
        self.data = np.zeros((len(labels), 2, 2), dtype=np.uint8)
        self.targets = np.array(labels)
        self.transform = None

    def __len__(self):
        return len(self.targets)


def make_clients(num_clients=20):
    labels = list(range(10)) * 5
    return [SimpleNamespace(id=k, train_set=TinyDataset(labels), test_set=TinyDataset(labels), global_test_id=0)
            for k in range(num_clients)]


def make_tests():
    return [TinyDataset(list(range(10))) for _ in range(4)]


def test_missing_classes_never_remove_all():
    clients = make_clients()
    draw_missing_classes(clients, 1.0, seed=0)
    for client in clients:
        assert set(client.train_set.targets.tolist()) == set(range(10))
    clients = make_clients()
    draw_missing_classes(clients, 0.6, seed=0)
    present = [set(client.train_set.targets.tolist()) for client in clients]
    assert all(0 < len(p) < 10 for p in present)


def test_sudden_drift_fraction_and_recurrent_reversal():
    clients, tests = make_clients(), make_tests()
    apply_drift(clients, tests, 100, "recurrent", 0.5)
    drifted = [c.id for c in clients if c.global_test_id != 0]
    assert drifted == list(range(10))
    assert clients[0].train_set.targets[1] == 2 and clients[0].train_set.targets[2] == 1
    assert clients[4].train_set.targets[3] == 4
    assert clients[7].train_set.targets[5] == 6
    assert clients[15].train_set.targets[1] == 1
    apply_drift(clients, tests, 150, "recurrent", 0.5)
    assert all((c.train_set.targets == np.array(list(range(10)) * 5)).all() for c in clients)


def test_incremental_drift_stages():
    clients, tests = make_clients(), make_tests()
    for round_index in (100, 110, 120):
        apply_drift(clients, tests, round_index, "incremental", 1.0)
    assert [c.global_test_id for c in clients[:10]] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert clients[0].train_set.targets[3] == 3      # stage-1 clients do not swap (3, 4)
    assert clients[3].train_set.targets[1] == 1      # stage-2 clients do not swap (1, 2)
