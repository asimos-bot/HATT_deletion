import matplotlib.pyplot as plt
from data_provider import DataProvider

class Plotter():

    def __init__(self, provider: DataProvider):

        self.provider = provider # give us the values to plot at each iteration

    def plot(self):

        while( self.provider.has_more_samples ):

            self.provider.update() # train the models, get new points

            t = self.provider.time_line # get x axis points (timeline)

            # plot them in distinct colors
            plt.plot(t, self.provider.control_metrics, color="red") # plot control model
            plt.plot(t, self.provider.deletion_metrics, color="blue") # plot deletion model

            # label axis
            plt.xlabel("Time")
            plt.ylabel("Accuracy")

            plt.pause(0.05)

        plt.show()
