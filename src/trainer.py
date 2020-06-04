from data_provider import DataProvider
from model import Model

from typing import List
import pandas as pd
from skmultiflow.data import DataStream

class Trainer(DataProvider):

    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, models: List[Model], spacing: int = 100):

        ##### MODEL

        # create DataStream
        self.stream = DataStream(data, y=labels)
        self.stream.prepare_for_use()

        # create models, each one with information about the forgetting process
        self.models = models

        #####

        ##### DATA PROCESSING FOR PLOT

        # get number of samples
        self.num_samples = data.shape[0]

        # spacing between each data point (we don't want to draw 200000 points!)
        # also allow us to take a bunch of samples at once to train
        self.spacing = spacing

        #####

    def update(self):

        # if there aren't 'self.spacing' samples left, take all the samples left
        n_samples_to_consume = self.spacing if self.spacing <= self.stream.n_remaining_samples() else self.stream.n_remaining_samples()

        x, y = self.stream.next_sample(n_samples_to_consume)

        for model in self.models:
            model.train(x, y)
            model.forget()

    @property
    def has_more_samples(self):

        return self.stream.has_more_samples()

    @property
    def time_line(self):

        return range( 0, self.num_samples - self.stream.n_remaining_samples(), self.spacing )

    @property
    def metrics(self):

        return [ model.metrics for model in self.models ]
