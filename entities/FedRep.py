"""FedRep (Collins et al., ICML 2021): shared encoder, local classifiers."""

from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.base import Client, EncoderServer, classifier_parameters, encoder_parameters, freeze


class FedRepClient(Client):
    def train_decoupled(self):
        """Classifier epochs with the encoder frozen, then encoder epochs with the classifier frozen."""
        self.model.train()
        loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True)
        freeze(self.model, train_classifier=True)
        optimizer = SGD(classifier_parameters(self.model), lr=self.cfg.classifier_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self._fit(loader, optimizer, self.cfg.classifier_epochs)
        freeze(self.model, train_classifier=False)
        optimizer = SGD(encoder_parameters(self.model), lr=self.cfg.encoder_lr,
                        weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self._fit(loader, optimizer, self.cfg.encoder_epochs)

    def _fit(self, loader, optimizer, epochs):
        for _ in range(epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device=self.device), labels.to(device=self.device)
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class FedRepServer(EncoderServer):
    """Averages and sends only the encoder."""
