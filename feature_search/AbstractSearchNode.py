from abc import abstractmethod, ABC
from typing import Self
from itertools import combinations
from typing_extensions import override


class AsbstractSearchNode(ABC):
  def __init__(self, data, features, selection,  model, parent=None):
    self.data = data
    self.features = features
    self.selection
    self.h = None
    self.model
    self.parent
    self.children = None

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

  def compute_heuristic(self) -> float:
    train_loader, validation_loader = self.data

    _, loss_values = train_network(self.model,
                                  train_loader,
                                  validation_loader,
                                  device = current_device,
                                  epochs= 30,
                                  lr =  1e-4,
                                  mu = 0.9,
                                  print_out = False
                                  )

    return float(min([i[1] for i in loss_values]))


  def get_heuristic(self) -> float:
    if self.h is None:
      self.h = self.compute_heuristic()

    return self.h

  def __lt__(self, other):
    return self.get_heuristic() < other.get_heuristic()

  def __hash__(self):
    return hash(self.selection)

  def __eq__(self, other):
    return set(self.selection) == set(other.selection)



class SearchNodeAdding(AsbstractSearchNode):
  @override
  def generate_children(self) -> list[Self]:
    children = []

    remaining_features = set(self.features) - set(self.selection)

    for feature in sorted(remaining_features):
      new_node = SearchNodeAdding(self.data, self.features, self.selection + [feature], self.model, self)
      children.append(new_node)

    return children
