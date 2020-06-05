from abc import ABC, abstractmethod
from pandas import DataFrame

class Model(ABC):

    @abstractmethod
    def train(self, X: DataFrame, Y: DataFrame):
        pass

    @property
    @abstractmethod
    def metrics(self):
        pass
