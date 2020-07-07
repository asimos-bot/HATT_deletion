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

    def __init__(self, forget_percentage=0):

        super().__init__()

        self.forgetter = forgetter.Forgetter(forget_percentage)

        self.counter=0

    def forget(self, X: np.ndarray, Y: np.ndarray):

        pass

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes=None, sample_weight=None):

        # train the model and get metrics based on its predicitons
        super().partial_fit(X, y.T, classes, sample_weight)

        if( self.forget_percentage != 0.0 ): self.forget(X, y)

    def predict(self, X: np.ndarray):

        return super().predict(X)
