from plotter import Plotter
from trainer import Trainer
import pandas as pd

class Visualizer():

    def __init__(self, data_filename: str, labels_filename: str):

        data = pd.read_csv(data_filename)
        labels = pd.read_csv(labels_filename)
        labels['y'] = labels['y'].astype('category')

        self.plotter = Plotter(Trainer(data, labels))

    def plot(self):

        self.plotter.plot()

if( __name__ == "__main__" ):

    visual = Visualizer("../movingSquares.data", "../movingSquares.labels")
    visual.plot()