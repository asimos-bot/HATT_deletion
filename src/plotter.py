#!/usr/bin/env python3
import matplotlib.pyplot as plt
import csv

class Plotter():

    def __init__(self, filename: str):

        self.filename = filename

        self.file = open(filename)
        self.reader = csv.DictReader(filter(lambda row: row[0]!='#', self.file))

    def __del__(self):

        if( hasattr(self,'file') and self.file ):

            self.file.close()

    def _field_has_keyword(self, field, keywords):

        for key in keywords:

            if( key in field ):

                return True

        return False

    def _filter_headers(self, list_of_keywords):

        return { field for field in self.reader.fieldnames if self._field_has_keyword(field, list_of_keywords) }

    def _get_field_lists(self, x_axis_field, list_of_keywords):

        fields = self._filter_headers(list_of_keywords)

        fields.add(x_axis_field)

        l = dict()

        # create list for each field
        for field in fields: l[field] = []

        for row in self.reader:

            for field in fields:

                l[field].append(float(row[field]))

        return l

    def plot_keyfields(self, x_axis_field, list_of_keywords, ylabel='mean accuracy', xlabel='number of objects given'):

        l = self._get_field_lists(x_axis_field, list_of_keywords)

        fields = self._filter_headers(list_of_keywords)

        axes = plt.axes()

        axes.yaxis.set_major_locator(plt.MaxNLocator(10))
        axes.xaxis.set_major_locator(plt.MaxNLocator(5))

        for field in [ key for key in l.keys() if key != x_axis_field ]:

            plt.plot(l[x_axis_field], l[field], label = field)

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(loc=2)
        plt.show()

if( __name__ == "__main__" ):

    import sys
    if( len(sys.argv) > 1 ):
        Plotter(sys.argv[1]).plot_keyfields('id', ['mean_acc_[HATT'])
    else:
        print("usage: plotter.py <path/to/log>")
