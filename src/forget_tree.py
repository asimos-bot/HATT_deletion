import skmultiflow
if( skmultiflow.__version__ == '0.4.1' ):
    from skmultiflow.trees import HATT as TreeClass
else:
    from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier as TreeClass


import numpy as np
import math

np.random.seed(0)

class ForgetHATT(TreeClass):

    def __init__(self, forget_cache_size=500, forget_percentage=0):

        super().__init__()

        # forget cache
        self.forget_cache = None

        # can't go above 1, can't go below 0
        self.forget_percentage = min(1, max(forget_percentage, 0))

        self.forget_cache_size = max([forget_cache_size, 0])

    def cache_is_full(self):

        return isinstance(self.forget_cache, np.ndarray) and self.forget_cache.shape[0] >= self.forget_cache_size

    def _add_to_cache(self, X, Y):
    
        if( not self.cache_is_full() ):

            X = np.concatenate((X, np.expand_dims(Y, axis=0).T), axis=1)

            if( not isinstance(self.forget_cache, np.ndarray) ):

                self.forget_cache = X

            else:

                self.forget_cache = np.concatenate((self.forget_cache, X))

    def _forget(self):

            # n of samples to forget
            n_samples = math.floor( self.forget_cache.shape[0] * self.forget_percentage )

            # get idx of rows to forget
            to_forget = np.random.choice(self.forget_cache.shape[0], n_samples, replace=False)

            # divide rows in X and Y
            X = self.forget_cache[to_forget, :-1]
            Y = self.forget_cache[to_forget, -1]

            super(ForgetHATT, self).partial_fit(X, Y.T, -1)

            # forget the selected rows
            np.delete(self.forget_cache, to_forget, axis=0)

    def forget(self, X: np.ndarray, Y: np.ndarray):

        self._add_to_cache(X, Y)

        if( self.cache_is_full() ): self._forget()

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes=None, sample_weight=None):

        # train the model and get metrics based on its predicitons
        super().partial_fit(X, y.T, classes, sample_weight)

        if( self.forget_percentage != 0.0 ): self.forget(X, y)

    def predict(self, X: np.ndarray):

        return super().predict(X)
