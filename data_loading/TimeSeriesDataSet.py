import pandas as pd
import torch

import configuration


class TimeSeriesDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 dataset: pd.DataFrame,
                 window_size: int,
                 prediction_length: int = 1,
                 look_ahead: int = 1,
                 target_column: str = configuration.target_column,
                 date_column: str = configuration.date_column,
                 max_val: pd.Series = None,
                 min_val: pd.Series = None,
                 fixed_feature_size: int = None,
                 is_indexed: bool = True,
                 scale_target: float = None
                 ):

        assert len(dataset) >= window_size + look_ahead + prediction_length - 1, ('dataset is too short for given '
                                                                                  'parameters')

        assert type(max_val) == type(min_val), 'max and min must both be of same type'

        assert not ((not is_indexed) and prediction_length > 1), ('for non-indexed targets the prediction length must '
                                                                  'be 1')

        self.window_size = window_size
        self.look_ahead = look_ahead
        self.prediction_length = prediction_length
        self.is_indexed = is_indexed
        self.scale_target = scale_target

        dataset = dataset.copy(deep=True)

        dataset.sort_values(date_column, ascending=True, inplace=True)
        dataset.drop(date_column, axis=1, inplace=True)

        self.targets = torch.Tensor(dataset[target_column].values)

        if max_val is None and min_val is None:
            self.min = dataset.min()
            self.max = dataset.max()

        else:
            self.max = max_val
            self.min = min_val

        dataset = (dataset - self.min) / (self.max - self.min)

        self.data = torch.Tensor(dataset.values)

        assert len(self.data) == len(self.targets)

        if fixed_feature_size is None:
            self.padding = 0

        else:
            assert fixed_feature_size >= self.data.shape[1]
            self.padding = fixed_feature_size - self.data.shape[1]

    def __len__(self) -> int:
        return len(self.targets) - self.window_size - (self.look_ahead - 1) - (self.prediction_length - 1)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.window_size
        l = self.look_ahead
        p = self.prediction_length

        window = self.data[idx:idx + w, :]

        if self.is_indexed:
            target = self.targets[idx + w + l - 1:idx + w + l - 1 + p]

        else:
            target = torch.prod(self.targets[idx + w:idx + w + l] + 1) - 1

        if self.scale_target is not None:
            target = target / self.scale_target

        window = torch.cat([window, torch.zeros((self.window_size, self.padding))], dim=1)

        window = window.unsqueeze(0)

        return window, target
