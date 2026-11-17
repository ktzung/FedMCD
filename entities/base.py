"""Client and server base classes; on their own they implement FedAvg."""

import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils.metrics import EVAL_BATCH_SIZE, accuracy
from utils.models import build_model

# Offsets that give each data loader of a client its own random stream.
LOADER_PURPOSES = {"train": 11, "prototype": 23, "balanced": 37}


def clip_gradients(parameters, max_norm):
    """Clip the gradient norm of ``parameters`` if ``max_norm`` is set."""
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))


class Client:
    def __init__(self, client_id, cfg, train_set, test_set):
        self.id = client_id
        self.cfg = cfg
        self.train_set = train_set
        self.test_set = test_set
        self.global_test_id = 0
        self.device = cfg.device
        self.model = build_model(cfg.dataset, cfg.num_classes).to(cfg.device)
        self.criterion = CrossEntropyLoss().to(cfg.device)
        self.pin_memory = torch.cuda.is_available()

    def loader_seed(self, round_index, purpose):
        return int(self.cfg.seed * 1_000_003 + self.id * 10_007 + int(round_index) * 101
                   + LOADER_PURPOSES[purpose])

    def seeded_loader(self, purpose, dataset, batch_size, round_index):
        """Shuffled loader driven by a client-local generator seeded per round and purpose."""
        generator = torch.Generator()
        generator.manual_seed(self.loader_seed(round_index, purpose))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                          pin_memory=self.pin_memory, generator=generator)

    def to_device(self, inputs, labels):
        return (inputs.to(self.device, non_blocking=self.pin_memory),
                labels.to(self.device, non_blocking=self.pin_memory))

    def set_params(self, new_params):
        for new_param, local_param in zip(new_params, self.model.parameters()):
            local_param.data = new_param.data.clone()

    def train(self, round_index):
        """Local SGD on all parameters (FedAvg)."""
        self.model.train()
        loader = self.seeded_loader("train", self.train_set, self.cfg.batch_size, round_index)
        init_params = parameters_to_vector(self.model.parameters())
        optimizer = SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        momentum=self.cfg.momentum)
        for _ in range(self.cfg.local_epochs):
            for inputs, labels in loader:
                inputs, labels = self.to_device(inputs, labels)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                clip_gradients([p for p in self.model.parameters() if p.requires_grad],
                               self.cfg.max_grad_norm)
                optimizer.step()
        self.local_update = parameters_to_vector(self.model.parameters()) - init_params

    def local_accuracy(self):
        return accuracy(self.model, self.test_set, self.device)

    def test_accuracy(self, global_tests):
        return accuracy(self.model, global_tests[self.global_test_id], self.device)


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.model = build_model(cfg.dataset, cfg.num_classes).to(cfg.device)
        self.client_data_size = []

    def record_client_sizes(self, clients):
        """Aggregation weights are the train-set sizes after class removal."""
        self.client_data_size = [len(client.train_set) for client in clients]

    def select_clients(self, clients):
        return random.sample(clients, int(self.cfg.num_clients * self.cfg.participation))

    def send_params(self, clients):
        for client in clients:
            client.set_params(self.model.parameters())

    def aggregate_by_params(self, clients):
        """Sample-size weighted average of all client parameters."""
        new_params = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            new_params += client_size * parameters_to_vector(client.model.parameters())
        new_params /= total_size
        vector_to_parameters(new_params, self.model.parameters())

    def aggregate_by_updates(self, clients):
        """Add the sample-size weighted average of the client updates to the global model."""
        total_size = 0
        update_sum = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            update_sum += client_size * client.local_update
        new_params = parameters_to_vector(self.model.parameters()) + update_sum / total_size
        vector_to_parameters(new_params, self.model.parameters())

    @staticmethod
    def local_evaluate(clients):
        """Mean accuracy of each client on its own local test partition."""
        return np.array([client.local_accuracy() for client in clients]).mean()

    @staticmethod
    def evaluate_each_client(clients, global_tests):
        """Mean accuracy of each client's model on the full test split under its concept."""
        return np.array([client.test_accuracy(global_tests) for client in clients]).mean()

    def evaluate_shared_model(self, clients, global_tests):
        """Same metric as ``evaluate_each_client`` when every client holds the same model."""
        counts = {}
        for client in clients:
            counts[client.global_test_id] = counts.get(client.global_test_id, 0) + 1
        weighted = 0.0
        model = clients[0].model
        for test_id, count in counts.items():
            weighted += count * accuracy(model, global_tests[test_id], self.device)
        return weighted / len(clients)

    def evaluate_shared_encoder(self, clients, global_tests):
        """Same metric as ``evaluate_each_client`` when all clients share one encoder.

        The encoder runs once per test copy; each client applies its own classifier.
        """
        grouped = {}
        for client in clients:
            grouped.setdefault(client.global_test_id, []).append(client)
            client.model.eval()
        accuracies = []
        with torch.no_grad():
            for test_id, group in grouped.items():
                correct = {client.id: 0 for client in group}
                total = 0
                loader = DataLoader(global_tests[test_id], batch_size=EVAL_BATCH_SIZE)
                encoder = group[0].model.hidden_layers
                for inputs, labels in loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    features = encoder(inputs)
                    total += labels.numel()
                    for client in group:
                        predictions = client.model.fc(features).argmax(dim=1)
                        correct[client.id] += (predictions == labels).sum().item()
                accuracies.extend(correct[client.id] / total for client in group)
        return float(np.mean(accuracies))


def encoder_parameters(model):
    return list(model.hidden_layers.parameters())


def classifier_parameters(model):
    return [model.fc.weight, model.fc.bias]


def freeze(model, train_classifier):
    for p in encoder_parameters(model):
        p.requires_grad = not train_classifier
    for p in classifier_parameters(model):
        p.requires_grad = train_classifier


class EncoderServer(Server):
    """Averages and sends only the encoder."""

    def send_encoder(self, clients):
        params = encoder_parameters(self.model)
        for client in clients:
            for new, local in zip(params, encoder_parameters(client.model)):
                local.data = new.data.clone()

    def aggregate_encoders(self, clients):
        params = encoder_parameters(self.model)
        new_params = torch.zeros_like(parameters_to_vector(params))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            new_params += client_size * parameters_to_vector(encoder_parameters(client.model))
        new_params /= total_size
        vector_to_parameters(new_params, params)


def class_mean_features(model, dataset, batch_size, device):
    """Mean feature of every class present in ``dataset`` (shuffled loader on the global generator)."""
    features_by_class = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            _, features = model(inputs, True)
            features = features.clone().detach()
            for i in range(len(labels)):
                features_by_class.setdefault(labels[i].item(), []).append(features[i, :])
    return {label: torch.mean(torch.stack(values), dim=0) for label, values in features_by_class.items()}
