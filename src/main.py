from skmultiflow.evaluation import EvaluatePrequential
from forget_tree import ForgetHATT
from skmultiflow.data import DataStream
import pandas as pd
from skmultiflow.trees import HATT

data = pd.read_csv("../datasets/movingSquares.data")
labels = pd.read_csv("../datasets/movingSquares.labels")
labels['y'] = labels['y'].astype('category')

stream = DataStream(data=data, y=labels)
stream.prepare_for_use()

evaluator = EvaluatePrequential(
        show_plot=True,
        metrics=[
                'precision',
                'accuracy'
            ])

models = [
        ForgetHATT(forget_percentage=0),
        ForgetHATT(forget_percentage=0.1),
        ]

model_names = [
            'HATT',
            'HATT 0.1',
        ]

evaluator.evaluate(stream=stream, model=models, model_names = model_names)
