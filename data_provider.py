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
    def control_metrics(self):
        pass

    @property
    @abstractmethod
    def deletion_metrics(self):
        pass

