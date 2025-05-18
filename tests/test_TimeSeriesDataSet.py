import unittest

import pandas as pd
import torch

from data_loading import TimeSeriesDataSet


class Test_TimeSeriesDataSet(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='1/1/2020', end='3/15/20', freq='W')

        targets = range(0, 11)
        features = range(20, -1, -2)

        self.dataframe = pd.DataFrame({'Datum': dates, 'Target': targets, 'Feature': features})

        self.target = 'Target'
        self.date = 'Datum'

        self.min = self.dataframe.iloc[:, 1:].min() * 0
        self.max = self.dataframe.iloc[:, 1:].max() * 0 + 1

        self.dataframe = self.dataframe.sample(frac=1)

    def test_base(self):
        timeseries = TimeSeriesDataSet(dataset=self.dataframe,
                                       window_size=3,
                                       prediction_length=2,
                                       look_ahead=2,
                                       target_column=self.target,
                                       date_column=self.date,
                                       max_val=None,
                                       min_val=None)

        self.assertEqual(len(timeseries), 6)

        x0, t0 = timeseries[0]

        x0_solution = torch.tensor([[0, 1], [0.1, 0.9], [0.2, 0.8]])
        t0_solution = torch.tensor([[4.0, 5.0]])

        self.assertTrue(torch.allclose(x0, x0_solution, atol=1e-4))
        self.assertTrue(torch.allclose(t0, t0_solution, atol=1e-4))

        x5, t5 = timeseries[5]

        x5_solution = torch.tensor([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]])
        t5_solution = torch.tensor([[9.0, 10.0]])

        self.assertTrue(torch.allclose(x5, x5_solution, atol=1e-4))
        self.assertTrue(torch.allclose(t5, t5_solution, atol=1e-4))

        self.assertEqual(x0.size(), (1, 3, 2))
        self.assertEqual(t0.size(), torch.Size([2]))

    def test_scaling(self):
        timeseries = TimeSeriesDataSet(dataset=self.dataframe,
                                       window_size=3,
                                       prediction_length=2,
                                       look_ahead=2,
                                       target_column=self.target,
                                       date_column=self.date,
                                       max_val=self.max,
                                       min_val=self.min)

        x0, t0 = timeseries[0]

        x0_solution = torch.tensor([[0, 20], [1, 18], [2, 16]]).float()
        t0_solution = torch.tensor([[4.0, 5.0]])

        self.assertTrue(torch.allclose(x0, x0_solution, atol=1e-4))
        self.assertTrue(torch.allclose(t0, t0_solution, atol=1e-4))

    def test_non_indexed_returns(self):
        timeseries = TimeSeriesDataSet(dataset=self.dataframe,
                                       window_size=3,
                                       prediction_length=1,
                                       look_ahead=2,
                                       target_column=self.target,
                                       date_column=self.date,
                                       max_val=self.max,
                                       min_val=self.min,
                                       is_indexed=False)

        x0, t0 = timeseries[0]
        self.assertEqual(len(timeseries), 7)

        x0_solution = torch.tensor([[0, 20], [1, 18], [2, 16]]).float()
        t0_solution = torch.tensor([[(3 + 1) * (4 + 1) - 1]]).float()

        self.assertEqual(t0.size(), torch.Size([1]))

        self.assertTrue(torch.allclose(x0, x0_solution, atol=1e-4))
        self.assertTrue(torch.allclose(t0, t0_solution, atol=1e-4))

    def test_padding(self):
        timeseries = TimeSeriesDataSet(dataset=self.dataframe,
                                       window_size=3,
                                       prediction_length=2,
                                       look_ahead=2,
                                       target_column=self.target,
                                       date_column=self.date,
                                       max_val=None,
                                       min_val=None,
                                       fixed_feature_size=5)

        print(timeseries.padding)

        self.assertEqual(len(timeseries), 6)

        x0, t0 = timeseries[0]

        self.assertEqual(x0.size(), (1, 3, 5))

        x0_solution = torch.tensor([[[0, 1, 0, 0, 0], [0.1, 0.9, 0, 0, 0], [0.2, 0.8, 0, 0, 0]]])
        t0_solution = torch.tensor([[4.0, 5.0]])

        self.assertTrue(torch.allclose(x0, x0_solution, atol=1e-4))
        self.assertTrue(torch.allclose(t0, t0_solution, atol=1e-4))

    def test_shape(self):
        timeseries = TimeSeriesDataSet(dataset=self.dataframe,
                                       window_size=2,
                                       prediction_length=4,
                                       look_ahead=1,
                                       target_column=self.target,
                                       date_column=self.date,
                                       max_val=None,
                                       min_val=None,
                                       fixed_feature_size=5,
                                       )

        loader = torch.utils.data.DataLoader(timeseries, batch_size=3)

        for i, (x, t) in enumerate(loader):
            if i == 0:
                self.assertEqual(x.size(), (3, 1, 2, 5))
                self.assertEqual(t.size(), (3, 4))

        timeseries = TimeSeriesDataSet(dataset=self.dataframe,
                                       window_size=2,
                                       prediction_length=1,
                                       look_ahead=1,
                                       target_column=self.target,
                                       date_column=self.date,
                                       max_val=None,
                                       min_val=None,
                                       fixed_feature_size=5,
                                       is_indexed=False
                                       )

        loader = torch.utils.data.DataLoader(timeseries, batch_size=3)

        for i, (x, t) in enumerate(loader):
            if i == 0:
                self.assertEqual(x.size(), (3, 1, 2, 5))
                self.assertEqual(t.size(), (3, 1))

