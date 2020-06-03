from data_provider import DataProvider
import pandas as pd
from skmultiflow.data import DataStream
from skmultiflow.trees import HATT

class Trainer(DataProvider):

    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame, spacing: int = 100):

        ##### MODEL

        # create DataStream
        self.stream = DataStream(data, y=labels)
        self.stream.prepare_for_use()

        # get HATT models
        self.hatt_control = HATT()
        self.hatt_deletion = HATT()
        
        #####

        ##### DATA PROCESSING FOR PLOT

        # get number of samples
        self.num_samples = data.shape[0]

        # spacing between each data point (we don't want to draw 200000 points!)
        self.spacing = spacing

        # metrics
        self._control_metrics = []
        self._deletion_metrics = []

        #####

    def update(self):

        x, y = self.stream.next_sample()

        self.hatt_control.partial_fit(x, y)
        self.hatt_deletion.partial_fit(x, y)

        self._control_metrics.append(self.hatt_control.score(x, y))
        self._deletion_metrics.append(self.hatt_deletion.score(x, y))

    @property
    def has_more_samples(self):

        return self.stream.has_more_samples()

    @property
    def time_line(self):

        return range( 0, self.num_samples - self.stream.n_remaining_samples() )

    @property
    def control_metrics(self):

        return self._control_metrics

    @property
    def deletion_metrics(self):

        return self._deletion_metrics
