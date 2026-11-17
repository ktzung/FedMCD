"""FedBABU (Oh et al., ICLR 2022): the encoder is trained with a fixed random classifier,
then each client fine-tunes the whole model before testing."""

from copy import deepcopy

from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, EncoderServer, encoder_parameters, freeze
from utils.metrics import accuracy


def fit(model, criterion, loader, optimizer, epochs, device):
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            loss = criterion(model(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class FedBABUClient(Client):
    """Evaluated with a fine-tuned copy of its model; the copy does not affect training."""

    def __init__(self, client_id, cfg, train_set, test_set):
        super().__init__(client_id, cfg, train_set, test_set)
        self.evaluation_model = self.model

    def train_encoder(self):
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        freeze(self.model, train_classifier=False)
        optimizer = SGD(encoder_parameters(self.model), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        momentum=self.cfg.momentum)
        fit(self.model, self.criterion, loader, optimizer, self.cfg.local_epochs, self.device)

    def fine_tune(self):
        """Fine-tune a copy of the whole model for evaluation."""
        model = deepcopy(self.model)
        model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        for p in model.parameters():
            p.requires_grad = True
        optimizer = SGD(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        momentum=self.cfg.momentum)
        fit(model, self.criterion, loader, optimizer, self.cfg.local_epochs, self.device)
        self.evaluation_model = model

    def local_accuracy(self):
        return accuracy(self.evaluation_model, self.test_set, self.device)

    def test_accuracy(self, global_tests):
        return accuracy(self.evaluation_model, global_tests[self.global_test_id], self.device)


class FedBABUServer(EncoderServer):
    """Averages only the encoder; sends the whole model."""
