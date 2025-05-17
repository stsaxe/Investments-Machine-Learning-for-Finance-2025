import unittest

import pandas as pd

from data_loading import split_dataset, TimeSeriesDataSet


class TestSplitting(unittest.TestCase):
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
        train, val, test = split_dataset(self.dataframe,
                                         window_size=3,
                                         prediction_length=2,
                                         look_ahead=2,
                                         test_split=0.2,
                                         validation_split=0.2,
                                         target_column='Target',
                                         date_column='Datum',
                                         scale_target=False
                                         )
        self.assertIsInstance(train, TimeSeriesDataSet)
        self.assertIsInstance(val, TimeSeriesDataSet)
        self.assertIsInstance(test, TimeSeriesDataSet)

        print(len(train), len(val), len(test))
        x_train_0, t_train_0 = train[0]
        x_val_0, t_val_0 = val[0]
        x_test_0, t_test_0 = test[0]
        print("hi")
        print(x_train_0, t_train_0)
        print("hi")
        print(x_val_0, t_val_0)
        print("hi")
        print(x_test_0, t_test_0)





"""
def split_dataset(dataset: pd.DataFrame,
              window_size: int,
              prediction_length: int,
              look_ahead: int,
              validation_split: pd.Timestamp | float = configuration.validation_split,
              test_split: pd.Timestamp | float = configuration.test_split,
              target_column: str = configuration.target_column,
              date_column: str = configuration.date_column,
              is_indexed: bool = True,
              scale_target: bool = True,
              time_step_size: int = configuration.time_step_size,
              time_step_unit: str = configuration.time_step_unit
              ) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
"""
