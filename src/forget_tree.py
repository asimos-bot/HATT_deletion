import skmultiflow
if( skmultiflow.__version__ == '0.4.1' ):
    from skmultiflow.trees import HATT as TreeClass
else:
    from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier as TreeClass

import numpy as np
import math
import forgetter
import csv

np.random.seed(0)

class ForgetHATT(TreeClass):


    def __init__(self, data: str, label: str, forget_percentage: float, delimiter=","):

        super().__init__()

        self.node_count_arr = []
        self.counter=0
        self.forgetter = forgetter.Forgetter(data, label, forget_percentage, delimiter=delimiter)

        if( forget_percentage != 0 ):
            self.interval = 1/forget_percentage

    def get_mean_nodes(self, mean_size = 1000):
        
        media = []#Yaxis
        x_axis = []

        for i in range(0, len(self.node_count_arr), mean_size):

            x_axis.append(i + mean_size)

            media.append( sum([ self.node_count_arr[j] for j in filter( lambda x: x < len(self.node_count_arr), range(i, i + mean_size)) ]) / mean_size )

        media = [0] + media
        x_axis = [0] + x_axis

        return media, x_axis

    def mean_nodes_to_csv(self, filepath: str, mean_size = 1000):

        with open(filepath, "w") as f:

            writer = csv.DictWriter(f, fieldnames=['x', 'mean'])

            writer.writeheader()

            media, x_axis = self.get_mean_nodes(mean_size)

            for mean, x in zip(media, x_axis):

                writer.writerow({ 'x':x, 'mean': mean })

    def forget_policy(self, X: np.ndarray, Y: np.ndarray):

        self.counter+=1

        if( self.counter % self.interval == 0):
            self.counter=0

            x, y = self.forgetter.next_to_forget()

            print(x)
            print(y)
            super().partial_fit(x, y, -1)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes=None, sample_weight=None):

        # train the model and get metrics based on its predicitons
        super().partial_fit(X, y, classes, sample_weight)

        self.node_count_arr.append(self._tree_root.count_nodes()[1])

        if( self.forgetter.forget_percentage != 0 ): self.forget_policy(X, y)

    def predict(self, X: np.ndarray):

        return super().predict(X)
