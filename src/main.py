from skmultiflow.evaluation import EvaluatePrequential
from forget_tree import ForgetHATT
from skmultiflow.data import DataStream
import pandas as pd
from skmultiflow.trees import HATT

data_filepath="../datasets/transient_chess.data"
labels_filepath="../datasets/transient_chess.labels"


data = pd.read_csv(data_filepath, delimiter=" ")
labels = pd.read_csv(labels_filepath, delimiter=" ")
labels['y'] = labels['y'].astype('category')

stream = DataStream(data=data, y=labels)
stream.prepare_for_use()

evaluator = EvaluatePrequential(
        output_file="log.log",
        show_plot=True,
        metrics=[
                'accuracy'
            ])

models = [
        ForgetHATT(data_filepath, labels_filepath, forget_percentage=0, delimiter=" "),
        ForgetHATT(data_filepath, labels_filepath, forget_percentage=0.1, delimiter=" "),
        ForgetHATT(data_filepath, labels_filepath, forget_percentage=0.25, delimiter=" "),
        ForgetHATT(data_filepath, labels_filepath, forget_percentage=0.5, delimiter=" "),
        ForgetHATT(data_filepath, labels_filepath, forget_percentage=0.75, delimiter=" "),
        ]

model_names = [
            'HATT 0%',
            'HATT 10%',
            'HATT 25%',
            'HATT 50%',
            'HATT 75%'
        ]

evaluator.evaluate(stream=stream, model=models, model_names = model_names)

for i, model in enumerate(models):

    model.mean_nodes_to_csv('mean_nodes' + str(i) + '.csv')

