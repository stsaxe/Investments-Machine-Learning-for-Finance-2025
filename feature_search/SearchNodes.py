import copy
from abc import abstractmethod, ABC
from queue import Queue
from typing import Self

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.linear_model import LinearRegression
from typing_extensions import override

from data_loading import split_dataset, create_dataloaders
from feature_search.ModelConfig import ModelConfigCNN
from feature_search.SplitConfig import SplitConfig
from modeling.Trainer import Trainer
from models.CNN import CNN


class AbstractSearchNode(ABC):
    def __init__(self, data: pd.DataFrame,
                 selection: list[str],
                 model_trainer: Trainer,
                 target_column: str,
                 date_column: str,
                 model_config: ModelConfigCNN,
                 split_config: SplitConfig,
                 num_iterations: int = 1,
                 max_children: int = None,
                 parent: Self = None):

        self.data = data
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
        self.children_queue = None
        self.max_children = max_children

        assert split_config.window_size == model_config.window_size
        assert split_config.prediction_length == model_config.output_size

        assert target_column not in selection
        assert date_column not in selection

    def get_depth(self) -> int:
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_depth()

    @abstractmethod
    def generate_children(self) -> list[Self]:
        pass

    def has_next_child(self) -> bool:
        return self.get_children_queue().qsize() > 0

    def get_children_queue(self) -> Queue:
        if self.children_queue is None:
            children = self.get_children()
            self.children_queue = Queue()

            for child in children:
                self.children_queue.put(child)

        return self.children_queue

    def next_child(self) -> Self | None:
        if self.has_next_child():
            return self.children_queue.get()
        else:
            return None

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

    def find_elbow_point(self, data):
        # Create an array of (x, y) points where x is the index
        n_points = len(data)
        all_points = np.column_stack((np.arange(n_points), data))

        # Line from first to last point
        first_point = all_points[0]
        last_point = all_points[-1]

        # Compute the normalized vector of the line
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        # Compute distances from each point to the line
        distances = []
        for point in all_points:
            # Vector from the first point to the current point
            vec_from_first = point - first_point
            # Projection scalar of vec_from_first onto the normalized line vector
            proj_length = np.dot(vec_from_first, line_vec_norm)
            proj_point = first_point + proj_length * line_vec_norm
            # Calculate the perpendicular distance
            distance = np.linalg.norm(point - proj_point)
            distances.append(distance)

        # The elbow is the point with maximum distance
        elbow_index = np.argmax(distances)
        return elbow_index

    def __compute_heuristic(self) -> tuple[float, float, float]:
        train, val = self.__generate_data_loaders()

        all_loss_values = []

        for i in range(self.num_iterations):
            self.model_trainer.model = self.__model_factory()
            _, loss_values = self.model_trainer.train(train, val, return_loss=True, print_out=False)

            values_train = np.array([v[0] for v in loss_values])
            values_validation = np.array([v[1] for v in loss_values])

            ellbow_point = self.find_elbow_point(values_train)

            epochs = self.model_trainer.epochs
            selected_point = int(min(ellbow_point + 0.2 * epochs, epochs))

            value = values_validation[selected_point]

            all_loss_values.append(value)

        all_loss_values = np.array(all_loss_values)

        return float(all_loss_values.mean()), float(all_loss_values.std()), float(np.median(all_loss_values))

    def get_heuristic(self) -> tuple[float, float, float]:
        if self.h is None:
            self.h = self.__compute_heuristic()

        return self.h

    def __lt__(self, other):
        return self.get_heuristic()[2] < other.get_heuristic()[2]

    def __eq__(self, other):
        return set(self.selection) == set(other.selection)


class SearchNodeAdding(AbstractSearchNode):

    def ordered_expansion(self):
        idx = int(len(self.data) * (1 - self.split_config.test_split - self.split_config.val_split))

        temp = self.data.iloc[:idx, :].copy(deep=True)

        temp['Target'] = temp[self.target].shift(self.split_config.look_ahead)
        temp.dropna(axis=0, inplace=True)

        Y = temp['Target']
        X = temp[[self.target] + self.selection]

        linear_model = LinearRegression(fit_intercept=True)
        linear_model.fit(X, Y)
        residuals = list(Y - linear_model.predict(X))

        temp.drop(self.selection + ['Target', self.target, self.date_column], axis=1, inplace=True)
        temp['Residuals'] = residuals

        corr_matrix = temp.corr()
        corr_resid_column = corr_matrix[['Residuals']].sort_values(by='Residuals',
                                                                   ascending=False,
                                                                   key=lambda i: abs(i))
        corr_resid_column.drop("Residuals", axis=0, inplace=True)

        return corr_resid_column.index.tolist()

    @override
    def generate_children(self) -> list[Self]:
        children = []

        _, _, _ = self.get_heuristic()

        feature_ordering = self.ordered_expansion()

        if self.max_children is not None:
            feature_ordering = feature_ordering[:self.max_children]

        for feature in feature_ordering:
            new_node = SearchNodeAdding(data=self.data,
                                        selection=self.selection + [feature],
                                        model_trainer=copy.deepcopy(self.model_trainer),
                                        target_column=self.target,
                                        date_column=self.date_column,
                                        split_config=self.split_config,
                                        num_iterations=self.num_iterations,
                                        parent=self,
                                        max_children=self.max_children,
                                        model_config=self.model_config)

            children.append(new_node)

        return children
