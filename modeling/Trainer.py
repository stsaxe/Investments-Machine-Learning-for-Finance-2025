import copy
from typing import Tuple, Any

import torch.nn
import warnings

from typing_extensions import Self

import configuration
from modeling.AbstractModelWrapper import AbstractModelWrapper


class Trainer(AbstractModelWrapper):
    def __init__(self,
                 model: torch.nn.Module = None,
                 device: torch.device = configuration.device,
                 epochs: int = 100,
                 learning_rate: float = 1e-4,
                 momentum: float = 0.9,
                 early_stopping: bool = True,
                 loss_function=torch.nn.MSELoss()):

        super().__init__(model, device)

        self.model = model
        self.device = device
        self.loss_values = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.loss_function = loss_function

    def __validate(self, network, validation_loader: torch.utils.data.DataLoader) -> float:
        validation_loss = 0
        total_samples_validation = 0

        network.eval()

        for x, t in validation_loader:
            x = x.to(self.device)
            t = t.to(self.device)

            with torch.no_grad():
                out = network(x)
                loss = self.loss_function(out, t)

                validation_loss += loss.item() * len(t)
                total_samples_validation += len(t)

        validation_loss /= total_samples_validation

        return validation_loss

    def __train_epoch(self, network, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim) -> tuple[torch.optim, int]:

        train_loss = 0
        total_samples_training = 0

        network.train()

        for x, t in train_loader:
            x = x.to(self.device)
            t = t.to(self.device)

            optimizer.zero_grad()
            out = network(x)

            loss = self.loss_function(out, t)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(t)
            total_samples_training += len(t)

        train_loss /= total_samples_training

        return optimizer, train_loss

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              validation_loader: torch.utils.data.DataLoader = None,
              print_out: bool = True
              ) -> Self:

        self.check_model()
        self.check_device()
        assert not (validation_loader is None and self.early_stopping), 'early stopping requires a validation set'

        optimizer = torch.optim.SGD(params=self.model.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum)

        network = self.model.to(self.device)
        best_network = copy.deepcopy(network)

        best_loss = float('inf')
        all_loss = []

        for epoch in range(1, 1 + self.epochs):

            optimizer, train_loss = self.__train_epoch(network, train_loader, optimizer)

            if print_out:
                print(f"Epoch {epoch}; train loss: {round(train_loss, 5)}")

            if validation_loader is not None:
                validation_loss = self.__validate(network, validation_loader)

                if validation_loss < best_loss and self.early_stopping:
                    best_network = copy.deepcopy(network)
                    best_loss = validation_loss

                if print_out:
                    print(f"Epoch {epoch}; validation loss: {round(validation_loss, 5)}")
                    print("\n")

                all_loss.append((train_loss, validation_loss))

            else:
                all_loss.append(train_loss)

        if self.early_stopping:
            self.model = copy.deepcopy(best_network)
        else:
            self.model = copy.deepcopy(network)

        torch.cuda.empty_cache()

        return self.model
