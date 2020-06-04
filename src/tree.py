from skmultiflow.trees import HATT
from model import Model
from pandas import DataFrame

class HATT_Forget(Model):

    def __init__(self):
        
        self.hatt = HATT()

        self.mean_accuracies = []

    def forget(self):
        pass

    def train(self, X: DataFrame, Y: DataFrame):

        # train the model and get metrics based on its predicitons
        self.hatt.partial_fit(X, Y)

        self.mean_accuracies.append( self.hatt.score(X, Y) )

    @property
    def metrics(self):
        return self.mean_accuracies
