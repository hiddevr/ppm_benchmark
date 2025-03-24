from abc import ABC, abstractmethod


class BaseMetric(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, predictions, targets):
        pass
