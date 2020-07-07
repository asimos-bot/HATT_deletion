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

    node_count_arr = []

    def __init__(self, data: str, label: str, forget_percentage: float, delimiter=","):

        super().__init__()

        self.counter=0
        self.forgetter = forgetter.Forgetter(data, label, forget_percentage, delimiter=delimiter)

        if( forget_percentage != 0 ):
            self.interval = 1/forget_percentage

    def get_mean_nodes(self, mean_size = 1000):
        
        media = []#Yaxis
        x_axis = []

        for i in range(0, len(node_count_arr), mean_size):
            media.append(0)
            x_axis.append(i + mean_size)
        
        for j in range(i, i + mean_size):
            media[i // mean_size] = media[i // mean_size] + self.node_count_arr[j]
            media[i // mean_size] = media[i // mean_size] / mean_size
 
        media = [0] + media
        x_axis = [0] + x_axis
        return media, x_axis

    def mean_nodes_to_csv(self, filepath: str, mean_size = 1000):

        with open(filepath, "w") as f:

            writer = csv.DictWriter(f, fieldnames=['x', 'mean number of nodes'])

            writer.writeheader()
            for mean, x in zip(self.get_mean_nodes(mean_size)):

                writer.write({ 'x':x, 'mean': mean })

    def forget_policy(self, X: np.ndarray, Y: np.ndarray):

        self.counter+=1

        if( self.counter % self.interval == 0):
            self.counter=0

            x, y = self.forgetter.next_to_forget()
            super().partial_fit(x, y, -1)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):

        self.node_count_arr.append(self._tree_root.count_nodes()[1])

        # train the model and get metrics based on its predicitons
        super().partial_fit(X, y, classes, sample_weight)

        if( self.forgetter.forget_percentage != 0 ): self.forget_policy(X, y)

    def predict(self, X: np.ndarray):

        return super().predict(X)
