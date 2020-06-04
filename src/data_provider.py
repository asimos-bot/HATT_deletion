from abc import ABC, abstractmethod

class DataProvider(ABC):

    @abstractmethod
    def update(self):
        pass

    @property
    @abstractmethod
    def has_more_samples(self):
        pass

    @property
    @abstractmethod
    def time_line(self):
        pass

    @property
    @abstractmethod
    def metrics(self):
        pass
