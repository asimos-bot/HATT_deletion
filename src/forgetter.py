import numpy as np
import pandas as pd

class Forgetter():

    def __init__(self, data_file: str, labels_file: str, forget_percentage : float, delimiter: str = ","):

        if( forget_percentage < 0 or forget_percentage > 1 ):
            raise ResourceWarning("Forgetter only accepts a forget percentage between 0 and 1")

        data = pd.read_csv(data_file, delimiter=delimiter)
        labels = pd.read_csv(labels_file, delimiter=delimiter)

        self.data, self.labels = self._objs_to_forget(data, labels, forget_percentage)

        self.pointer=0

    def _objs_to_forget(self, data : pd.DataFrame, labels: pd.DataFrame, forget_percentage: float):
 
        self._set_label_as_category(labels)

        concat_dataframe = pd.concat([data, labels], axis=1, sort=False).sample(frac = forget_percentage, replace=False, random_state=0)

        return concat_dataframe[ data.columns.values ], concat_dataframe[ labels.columns.values ]

    def _set_label_as_category(self, labels : pd.DataFrame):

        for label in labels.columns.values:
            labels[label] = labels[label].astype('category')

    def to_csv(self, filepath: str):

        pd.concat([self.data, self.labels], axis=1, sort=True).to_csv(filepath, index=False, float_format='%.7f')

    def next_to_forget(self):

        if self.pointer < self.data.shape[0]:

            data = self.data.iloc[self.pointer].to_numpy()
            label = self.labels.iloc[self.pointer]

            if isinstance(label, np.int64):
                label = np.expand_dims(label, axis=0)

            self.pointer+=1

            return data, label

        else:
            return None, None

    def __iter__(self):

        self.pointer=0
        return self

    def __next__(self):

        data, label = self.next_to_forget()

        if( isinstance(data, np.ndarray) and isinstance(label, np.ndarray) ):

            return data, label

        else:

            raise StopIteration

if __name__ == "__main__":

    Forgetter(data_file="../datasets/movingSquares.data", labels_file="../datasets/movingSquares.labels")
