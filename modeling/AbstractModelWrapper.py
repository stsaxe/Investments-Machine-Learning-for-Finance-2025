import warnings
from abc import ABC

import torch.nn
from typing_extensions import Self

import configuration


class AbstractModelWrapper(ABC):
    def __init__(self,
                 model: torch.nn.Module = None,
                 device: torch.device = configuration.device):
        self.model = model
        self.device = device

    def save(self, name: str, path: str) -> Self:
        self.check_model()
        torch.save(self.model, path + name)
        return self

    def load(self, path: str) -> Self:
        self.model = torch.load(path, weights_only=False),
        return self

    def check_model(self):
        assert self.model is not None, 'model is None and must be set at initialization or loaded from disk with load()'

    def check_device(self):
        if self.device == torch.device('cpu'):
            warnings.warn("Warning: Training on a CPU can be very slow!")
