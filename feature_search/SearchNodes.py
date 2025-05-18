from abc import abstractmethod, ABC
from typing import Self
import pandas as pd
import torch.utils.data
from typing_extensions import override


from data_loading import split_dataset, create_dataloaders
from feature_search.ModelConfig import ModelConfigCNN
from feature_search.SplitConfig import SplitConfig
from modeling.Trainer import Trainer
from models.CNN import CNN


class AbstractSearchNode(ABC):
    def __init__(self, data: pd.DataFrame,
                 features: list[str],
                 selection: list[str],
                 model_trainer: Trainer,
                 target_column: str,
                 date_column: str,
                 model_config: ModelConfigCNN,
                 split_config: SplitConfig,
                 num_iterations: int = 1,
                 parent: Self = None):

        self.data = data
        self.features = features
        self.selection = selection
        self.target = target_column
        self.date_column = date_column
        self.split_config = split_config
        self.h = None
        self.model_trainer = model_trainer
        self.parent = parent
        self.children = None
        self.num_iterations = num_iterations
        self.model_config = model_config

        assert split_config.window_size == model_config.window_size
        assert split_config.prediction_length == model_config.output_size

        assert target_column not in selection
        assert date_column not in selection
        assert date_column not in features
        assert target_column not in features

    def get_depth(self) -> int:
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_depth()

    @abstractmethod
    def generate_children(self) -> list[Self]:
        pass

    def get_children(self) -> list[Self]:
        if self.children is None:
            self.children = self.generate_children()

        return self.children

    def __model_factory(self) -> torch.nn.Module:
        configured_model = CNN(window_size=self.split_config.window_size,
                               num_features=len(self.selection) + 1,
                               output_size=self.split_config.prediction_length,
                               num_conv=self.model_config.num_conv,
                               kernel_size=self.model_config.kernel_size,
                               channels=self.model_config.channels,
                               padding=self.model_config.padding,
                               num_hidden_layers=self.model_config.num_hidden_layers,
                               hidden_size=self.model_config.hidden_size,
                               dropout=self.model_config.dropout
                               )

        return configured_model

    def __generate_data_loaders(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        features = list(set(self.selection + [self.target, self.date_column]))

        train, val, test = split_dataset(self.data[features],
                                         window_size=self.split_config.window_size,
                                         prediction_length=self.split_config.prediction_length,
                                         look_ahead=self.split_config.look_ahead,
                                         validation_split=self.split_config.val_split,
                                         test_split=self.split_config.test_split,
                                         is_indexed=self.split_config.is_indexed,
                                         scale_target=self.split_config.scale_target,
                                         fixed_feature_size=self.split_config.fixed_feature_size,
                                         target_column=self.target,
                                         date_column=self.date_column
                                         )

        train, val, _ = create_dataloaders(train, val, test, batch_size=self.split_config.batch_size)

        return train, val

    def __compute_heuristic(self) -> float:
        train, val = self.__generate_data_loaders()

        all_loss_values = []

        for i in range(self.num_iterations):
            self.model_trainer.model = self.__model_factory()
            _, loss_values = self.model_trainer.train(train, val, return_loss=True, print_out=False)

            min_value = min([v[1] for v in loss_values])

            print(min_value)

            all_loss_values.append(min_value)

        return float(sum(all_loss_values) / self.num_iterations)

    def get_heuristic(self) -> float:
        if self.h is None:
            self.h = self.__compute_heuristic()

        return self.h

    def __lt__(self, other):
        return self.get_heuristic() < other.get_heuristic()

    def __hash__(self):
        return hash(self.selection)

    def __eq__(self, other):
        return set(self.selection) == set(other.selection)


class SearchNodeAdding(AbstractSearchNode):

    def __ordered_expansion(self, remaining_features: list):
        assert self.date_column not in remaining_features
        assert self.target not in remaining_features

        l = len(self.data) * (1 - self.split_config.test_split - self.split_config.val_split)

        temp = self.data[remaining_features + [self.target]]

        corr_matrix = temp.iloc[:int(l), :].corr()

        corr_target_column = corr_matrix[[self.target]].sort_values(by=self.target,
                                                                    ascending=False,
                                                                    key=lambda x: abs(x))

        corr_target_column.drop(self.target, axis=0, inplace=True)

        return corr_target_column.index.tolist()

    @override
    def generate_children(self) -> list[Self]:
        children = []

        _ = self.get_heuristic()

        remaining_features = list(set(self.features) - set(self.selection))

        ordering = self.__ordered_expansion(remaining_features)

        for feature in ordering:
            new_node = SearchNodeAdding(data=self.data,
                                        features=self.features,
                                        selection=self.selection + [feature],
                                        model_trainer=self.model_trainer,
                                        target_column=self.target,
                                        date_column=self.date_column,
                                        split_config=self.split_config,
                                        num_iterations=self.num_iterations,
                                        parent=self,
                                        model_config=self.model_config)

            children.append(new_node)

        return children
