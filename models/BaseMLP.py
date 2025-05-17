import torch


class BaseMLP(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 num_hidden_layers: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.1):

        super().__init__()
        assert num_hidden_layers >= 2, 'number of hidden layers must be at least 2'

        layers = []
        n = num_hidden_layers

        for i in range(n):
            if i < n - 1:
                layer = torch.nn.Sequential(torch.nn.Linear(in_features=input_size * (i == 0) + (i != 0) * hidden_size,
                                                            out_features=hidden_size,
                                                            bias=False),
                                            torch.nn.BatchNorm1d(hidden_size),
                                            torch.nn.PReLU(),
                                            torch.nn.Dropout(dropout))
            else:
                layer = torch.nn.Linear(in_features=hidden_size,
                                        out_features=output_size,
                                        bias=True)

            layers.append(layer)

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
