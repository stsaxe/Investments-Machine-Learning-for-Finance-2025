import torch

from models.BaseMLP import BaseMLP


class LSTM(torch.nn.Module):
    def __init__(self,
                 num_features: int,
                 output_size: int = 1,
                 num_LSTM_layers: int = 3,
                 num_hidden_layers: int = 2,
                 hidden_size: int = 100,
                 dropout: int = 0.1):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=num_features,
                                  hidden_size=hidden_size,
                                  batch_first=True,
                                  num_layers=num_LSTM_layers)

        self.mlp = BaseMLP(hidden_size,
                           output_size,
                           num_hidden_layers,
                           hidden_size,
                           dropout)

    def forward(self, x):
        # removes the channel dimension as it is not needed
        x = x.squeeze()

        out, (final_hidden_state, final_cell_state) = self.lstm(x)

        # axis are B x S x K, we need S[-1]
        out = self.mlp(out[:, -1, :])

        return out
