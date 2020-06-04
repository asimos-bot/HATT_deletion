from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def forget(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

    @property
    @abstractmethod
    def metrics(self):
        pass
