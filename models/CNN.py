import torch

from models.BaseMLP import BaseMLP


class CNN(torch.nn.Module):
    def __init__(self,
                 window_size: int,
                 num_features: int,
                 output_size: int = 1,
                 num_conv: int = 3,
                 kernel_size: int = 3,
                 channels: int = 2,
                 padding: bool = False,
                 num_hidden_layers: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.1):
        super().__init__()

        assert kernel_size % 2 == 1, 'kernel size must be odd'
        k = kernel_size
        Q = channels
        f = 2

        conv_blocks = []
        out_size = window_size

        for i in range(num_conv):
            if padding:
                out_size = out_size // f

            else:
                out_size = (out_size - k + 1) // f

            block = torch.nn.Sequential(torch.nn.Conv2d(in_channels=max(Q * (i != 0), 1),
                                                        out_channels=Q * f,
                                                        kernel_size=(k, 1),
                                                        padding=((k // 2) * padding, 0),
                                                        stride=1,
                                                        bias=False),

                                        torch.nn.BatchNorm2d(Q*f),
                                        torch.nn.MaxPool2d(kernel_size=(f, 1)),
                                        torch.nn.PReLU()
                                        )

            conv_blocks.append(block)
            Q = Q * f

        self.convolutions = torch.nn.Sequential(*conv_blocks)

        flat_size = num_features * Q * out_size

        self.mlp = BaseMLP(flat_size,
                           output_size,
                           num_hidden_layers,
                           hidden_size,
                           dropout)

    def forward(self, x):
        x = self.convolutions(x)

        x = torch.flatten(x, start_dim=1)

        return self.mlp(x)
