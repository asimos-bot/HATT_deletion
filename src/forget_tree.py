from model import Model

from skmultiflow.trees import HATT
import numpy as np
import random
import math

class HATT_Forget(Model):

    def __init__(self, forget_cache_size=1000, forget_percentage=0, bulk=False):
        
        self.hatt = HATT()

        self.mean_accuracies = []

        # forget cache
        self.forget_cache = None

        # can't go above 1, can't go below 0
        self.forget_percentage = min(1, max(forget_percentage, 0))

        self.forget_cache_size = max([forget_cache_size, 0])

        # True for only deleting when deletion cache is full
        self.bulk = bulk

    def _add_to_cache(self, X: np.ndarray, Y: np.ndarray):

        n_samples = math.floor( X.shape[0] * self.forget_percentage )

        if( isinstance(self.forget_cache, np.ndarray) ):

            n_samples = min( n_samples, self.forget_cache_size - self.forget_cache.shape[0] - X.shape[0] )

        X = np.concatenate((X, Y), axis=1)

        self.forget_cache = X if not isinstance(self.forget_cache, np.ndarray) else np.concatenate((self.forget_cache, X))

    def _forget(self):

        # if bulk, only delete when cache is full
        if( self.bulk ):

            if( self.forget_cache_size == self.forget_cache.shape[0] ):

                X = self.forget_cache[:, :-1]
                Y = self.forget_cache[:, -1]

                self.hatt.partial_fit(X, Y.T, -1)

                self.forget_cache = None
        else:

            # n of samples to forget
            n_samples = math.floor( self.forget_cache.shape[0] * self.forget_percentage )

            # get idx of rows to forget
            to_forget = np.random.choice(self.forget_cache.shape[0], n_samples, replace=False)

            # divide rows in X and Y
            X = self.forget_cache[to_forget, :-1]
            Y = self.forget_cache[to_forget, -1]

            self.hatt.partial_fit(X, Y.T, -1)

            # forget the selected rows
            np.delete(self.forget_cache, to_forget, axis=0)

    def forget(self, X: np.ndarray, Y: np.ndarray):

        # if cache size is not full, add some random samples
        self._add_to_cache(X, Y)

        if( isinstance(self.forget_cache, np.ndarray) ): self._forget()

    def train(self, X: np.ndarray, Y: np.ndarray):

        # train the model and get metrics based on its predicitons
        self.hatt.partial_fit(X, Y.T[0], classes=["0", "1", "2", "3"])

        if( self.forget_percentage != 0 ): self.forget(X, Y)

        self.mean_accuracies.append( self.hatt.score(X, Y) )

    @property
    def metrics(self):
        return self.mean_accuracies
