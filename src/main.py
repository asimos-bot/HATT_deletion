from skmultiflow.evaluation import EvaluatePrequential
from forget_tree import ForgetHATT
from skmultiflow.data import DataStream
import pandas as pd
from skmultiflow.trees import HATT

data = pd.read_csv("../datasets/transient_chess.data", delimiter=" ")
labels = pd.read_csv("../datasets/transient_chess.labels", delimiter=" ")
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
        ForgetHATT(forget_percentage=0),
        ForgetHATT(forget_percentage=0.1),
        ForgetHATT(forget_percentage=0.25),
        ForgetHATT(forget_percentage=0.5),
        ForgetHATT(forget_percentage=0.75),
        ]

model_names = [
            'HATT 0%',
            'HATT 10%',
            'HATT 25%',
            'HATT 50%',
            'HATT 75%'
        ]

evaluator.evaluate(stream=stream, model=models, model_names = model_names)
