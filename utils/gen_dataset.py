"""Datasets, Dirichlet partition across clients, and removal of missing classes."""

import random

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms

DATASETS = ("cifar10", "fashion_mnist")
NUM_CLASSES = 10
# Every client first receives this many samples of every class (train / test split).
GUARANTEED_TRAIN_SAMPLES = 5
GUARANTEED_TEST_SAMPLES = 2
# Number of copies of the full test split; copies 1-3 receive the drifted labels.
NUM_GLOBAL_TEST_COPIES = 4


class ClientDataset(Dataset):
    """A subset of an image dataset that owns a copy of its labels."""

    def __init__(self, dataset, indices, transform=None):
        self.data = dataset.data[indices]
        self.targets = np.array(dataset.targets)[indices]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def partition_dataset(dataset, num_clients, alpha, guaranteed_per_class, seed):
    """Split sample indices across clients.

    Each client first receives ``guaranteed_per_class`` samples of every class;
    the remaining samples of class k are split by Dir(alpha) with seed ``seed + k``.
    """
    labels = np.array(dataset.targets)
    classes = set(labels)
    client_indices = [[] for _ in range(num_clients)]
    for k in range(len(classes)):
        class_indices = np.where(labels == k)[0]
        guaranteed = np.array_split(class_indices[:num_clients * guaranteed_per_class], num_clients)
        client_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_indices, guaranteed)]
        class_indices = class_indices[num_clients * guaranteed_per_class:]

        random.seed(seed + k)
        np.random.seed(seed + k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        client_indices = [idx_j + idx.tolist() for idx_j, idx in
                          zip(client_indices, np.split(class_indices, proportions))]

    for i in range(num_clients):
        random.shuffle(client_indices[i])
    return client_indices


def load_datasets(name, data_dir):
    """Return (train_set, train_transform, test_set, test_transform)."""
    if name == "fashion_mnist":
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True)
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif name == "cifar10":
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    return train_set, train_transform, test_set, test_transform


def distribute_dataset(name, data_dir, num_clients, alpha, seed):
    """Return per-client train sets, per-client test sets and copies of the full test set."""
    train_set, train_transform, test_set, test_transform = load_datasets(name, data_dir)
    train_indices = partition_dataset(train_set, num_clients, alpha, GUARANTEED_TRAIN_SAMPLES, seed)
    test_indices = partition_dataset(test_set, num_clients, alpha, GUARANTEED_TEST_SAMPLES, seed)
    client_train = [ClientDataset(train_set, train_indices[k], train_transform) for k in range(num_clients)]
    client_test = [ClientDataset(test_set, test_indices[k], test_transform) for k in range(num_clients)]
    global_tests = [ClientDataset(test_set, list(range(len(test_set))), test_transform)
                    for _ in range(NUM_GLOBAL_TEST_COPIES)]
    return client_train, client_test, global_tests


def remove_classes(dataset, missing_classes):
    """Return a view of ``dataset`` without the samples of ``missing_classes``."""
    if not missing_classes:
        return dataset
    targets = np.array(dataset.targets)
    keep = np.ones(len(targets), dtype=bool)
    for c in missing_classes:
        keep &= targets != c
    kept = np.where(keep)[0].tolist()
    if not kept:
        return dataset
    return ClientDataset(dataset, kept, dataset.transform)


def draw_missing_classes(clients, missing_prob, seed, num_classes=NUM_CLASSES):
    """Remove each class of each client independently with probability ``missing_prob``.

    If every class of a client is drawn, the client keeps all of them.
    """
    if not missing_prob or missing_prob <= 0.0:
        return
    rng = np.random.RandomState(seed)
    for client in clients:
        is_missing = rng.rand(num_classes) < missing_prob
        missing = {c for c in range(num_classes) if is_missing[c]}
        if missing and len(missing) < num_classes:
            client.train_set = remove_classes(client.train_set, missing)
