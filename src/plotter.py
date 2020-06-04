import matplotlib.pyplot as plt
from data_provider import DataProvider

class Plotter():

    colors = ["orange", "blue", "black", "yellow", "green", "red", "purple"]

    def __init__(self, provider: DataProvider):

        self.provider = provider # give us the values to plot at each iteration

    def plot(self):

        while( self.provider.has_more_samples ):

            self.provider.update() # train the models, get new points

            t = self.provider.time_line # get x axis points (timeline)

            for idx, metric in enumerate(self.provider.metrics):

                plt.plot(t, metric, color=Plotter.colors[idx])

            # label axis
            plt.xlabel("Time")
            plt.ylabel("Mean Accuracy")

            plt.pause(0.05)

        plt.show()
