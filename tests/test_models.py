import unittest

import torch

from models.CNN import CNN
from models.LSTM import LSTM
from models.MLP import MLP


class TestModels(unittest.TestCase):
    def test_mlp(self):
        network = MLP(num_features=10,
                      window_size=20,
                      num_hidden_layers=3,
                      hidden_size=50,
                      output_size=3
                      )

        input = torch.zeros(250, 1, 20, 10)
        output = network(input)
        self.assertEqual(output.size(), (250, 3))

    def test_LSTM_model(self):
        network = LSTM(num_features=10,
                       output_size=3,
                       num_LSTM_layers=5,
                       num_hidden_layers=4,
                       hidden_size=50
                       )

        input = torch.zeros(250, 1, 20, 10)
        output = network(input)
        self.assertEqual(output.size(), (250, 3))

    def test_CNN_model(self):
        window_size = 104
        features = 30

        network = CNN(window_size=window_size,
                      num_features=features,
                      output_size=3,
                      num_conv=4,
                      kernel_size=5,
                      channels=4,
                      padding=True,
                      num_hidden_layers=4,
                      hidden_size=50)

        input = torch.zeros(250, 1, window_size, features)
        output = network(input)

        self.assertEqual(output.size(), (250, 3))

        network = CNN(window_size=window_size,
                      num_features=features,
                      output_size=1,
                      num_conv=2,
                      kernel_size=7,
                      channels=4,
                      padding=False,
                      num_hidden_layers=3,
                      hidden_size=50)

        input = torch.zeros(250, 1, window_size, features)
        output = network(input)
        self.assertEqual(output.size(), (250, 1))
