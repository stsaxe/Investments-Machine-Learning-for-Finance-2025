import pandas as pd
import torch

import configuration
from data_loading.TimeSeriesDataSet import TimeSeriesDataSet


def load_data(path: str = configuration.data_path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding='utf-8', sep=';')

    columns = list(data.columns)

    assert configuration.target_column in columns, (f'The target column {configuration.target_column} was not '
                                                    f'found in the dataset')

    date_column = configuration.date_column

    assert date_column in columns, f'The date column {date_column} was not found in the dataset'

    columns = list(data.columns)

    assert columns[0] == configuration.date_column, 'Date column must be first column'

    data[date_column] = pd.to_datetime(data[date_column], format='%d.%m.%Y')

    data.dropna(axis=0, inplace=True)
    data.sort_values(by=[date_column], inplace=True, ascending=True)

    # this verifies that all data points are seperated by exactly the same amount of time as configured
    temp = (data[configuration.date_column].iloc[1:] - data[configuration.date_column].shift(1).dropna())
    delta = pd.Timedelta(**{configuration.time_step_unit: configuration.time_step_size})

    assert temp.max() == delta and temp.min() == delta, 'actual time step size and unit do not match configuration'

    return data


def split_dataset(dataset: pd.DataFrame,
                  window_size: int,
                  prediction_length: int,
                  look_ahead: int,
                  validation_split: pd.Timestamp | float = configuration.validation_split,
                  test_split: pd.Timestamp | float = configuration.test_split,
                  target_column: str = configuration.target_column,
                  date_column: str = configuration.date_column,
                  is_indexed: bool = True,
                  scale_target: bool = False,
                  fixed_feature_size: int = None,
                  time_step_size: int = configuration.time_step_size,
                  time_step_unit: str = configuration.time_step_unit
                  ) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    assert window_size >= 1
    assert prediction_length >= 1
    assert look_ahead >= 1
    assert target_column in dataset.columns
    assert date_column in dataset.columns

    assert type(test_split) is type(validation_split)
    assert not (scale_target and not is_indexed), 'target scaling requires indexed targets'

    dataset = dataset.copy(deep=True)
    dataset.sort_values(by=date_column, inplace=True, ascending=True)

    scaling = None

    if scale_target and is_indexed:
        scaling = float(dataset[target_column].iloc[0])

    elif not is_indexed:
        dataset[target_column] = dataset[target_column] / dataset[target_column].shift(1) - 1
        dataset.dropna(axis=0, inplace=True)

    if isinstance(test_split, float) and isinstance(validation_split, float):
        assert test_split + validation_split < 1

        train_split = float(1 - test_split - validation_split)

        length = dataset.shape[0]

        validation_split = dataset.loc[int(length * train_split), date_column]
        test_split = dataset.loc[int(length * (1 - test_split)), date_column]

    else:
        assert validation_split < test_split

    shift = (window_size + (look_ahead - 1) + (prediction_length - 1)) * time_step_size

    validation_split_shifted = validation_split - pd.Timedelta(shift, unit=time_step_unit)
    test_split_shifted = test_split - pd.Timedelta(shift, unit=time_step_unit)

    data_train = dataset[dataset[date_column] < validation_split]
    data_val = dataset[(dataset[date_column] >= validation_split_shifted) & (dataset[date_column] < test_split)]
    data_test = dataset[dataset[date_column] >= test_split_shifted]

    train_set = TimeSeriesDataSet(dataset=data_train,
                                  window_size=window_size,
                                  prediction_length=prediction_length,
                                  look_ahead=look_ahead,
                                  target_column=target_column,
                                  date_column=date_column,
                                  is_indexed=is_indexed,
                                  scale_target=scaling,
                                  fixed_feature_size = fixed_feature_size)

    max_val, min_val = train_set.max, train_set.min

    val_set = TimeSeriesDataSet(dataset=data_val,
                                window_size=window_size,
                                prediction_length=prediction_length,
                                look_ahead=look_ahead,
                                target_column=target_column,
                                date_column=date_column,
                                is_indexed=is_indexed,
                                max_val=max_val,
                                min_val=min_val,
                                scale_target=scaling,
                                fixed_feature_size = fixed_feature_size)

    test_set = TimeSeriesDataSet(dataset=data_test,
                                 window_size=window_size,
                                 prediction_length=prediction_length,
                                 look_ahead=look_ahead,
                                 target_column=target_column,
                                 date_column=date_column,
                                 is_indexed=is_indexed,
                                 max_val=max_val,
                                 min_val=min_val,
                                 scale_target=scaling,
                                 fixed_feature_size = fixed_feature_size)

    assert len(train_set) + len(val_set) + len(test_set) + window_size + look_ahead - 1 + prediction_length - 1 == len(
        dataset)

    return train_set, val_set, test_set


def create_dataloaders(train_set: TimeSeriesDataSet,
                       val_set: TimeSeriesDataSet,
                       test_set: TimeSeriesDataSet,
                       batch_size: int = configuration.batch_size,
                       shuffle_train_loader: bool = True
                       ) -> tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=shuffle_train_loader)

    validation_loader = torch.utils.data.DataLoader(val_set,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, validation_loader, test_loader
