import torch

from models.BaseMLP import BaseMLP


class MLP(torch.nn.Module):
    def __init__(self,
                 window_size: int,
                 num_features: int,
                 output_size: int = 1,
                 num_hidden_layers: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.1):

        super().__init__()

        self.mlp = BaseMLP(window_size * num_features,
                           output_size,
                           num_hidden_layers,
                           hidden_size,
                           dropout)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        return self.mlp(x)
