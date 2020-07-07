import skmultiflow
if( skmultiflow.__version__ == '0.4.1' ):
    from skmultiflow.trees import HATT as TreeClass
else:
    from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier as TreeClass

import numpy as np
import math
import forgetter

np.random.seed(0)

class ForgetHATT(TreeClass):

    def __init__(self, data: str, label: str, forget_percentage: float, delimiter=","):

        super().__init__()

        self.counter=0
        self.forgetter = forgetter.Forgetter(data, label, forget_percentage, delimiter=delimiter)

    def forget_policy(self, X: np.ndarray, Y: np.ndarray):

        self.counter+=1

        if( self.counter % 100 ):
            self.counter=0

            x, y = self.forgetter.next_to_forget()
            super().partial_fit(x, y, -1)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes=None, sample_weight=None):

        # train the model and get metrics based on its predicitons
        super().partial_fit(X, y, classes, sample_weight)

        self.forget_policy(X, y)

    def predict(self, X: np.ndarray):

        return super().predict(X)
