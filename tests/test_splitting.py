import unittest

import pandas as pd
import torch

from data_loading import split_dataset, TimeSeriesDataSet
from data_loading import TimeSeriesDataSet


class TestSplitting(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='1/1/2020', end='12/2/21', freq='W')

        targets = range(1, 101)
        features = range(200, 0, -2)

        self.dataframe = pd.DataFrame({'Datum': dates, 'Target': targets, 'Feature': features})

        self.target = 'Target'
        self.date = 'Datum'

        self.min = self.dataframe.iloc[:, 1:].min() * 0
        self.max = self.dataframe.iloc[:, 1:].max() * 0 + 1

        self.dataframe = self.dataframe.sample(frac=1)

    def test_splitting(self):
        train, val, test = split_dataset(self.dataframe,
                                         window_size=5,
                                         prediction_length=4,
                                         look_ahead=3,
                                         test_split=0.2,
                                         validation_split=0.2,
                                         target_column='Target',
                                         date_column='Datum',
                                         scale_target=True
                                         )

        self.assertIsInstance(train, TimeSeriesDataSet)
        self.assertIsInstance(val, TimeSeriesDataSet)
        self.assertIsInstance(test, TimeSeriesDataSet)

        self.assertEqual(len(train), 50)
        self.assertEqual(len(test), 20)
        self.assertEqual(len(val), 20)

