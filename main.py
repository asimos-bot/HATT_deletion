from classes.forgetclass import HATTForget
from skmultiflow.trees import HATT

preTrainSize = 1
streamSize = 20
forget = HATTForget(name = "forget",
                    dataPath = "movingSquares.data",
                    labelsPath = "movingSquares.labels",
                    forgetdataPath = "movingSquares.data",
                    forgetlabelsPath = "movingSquares.labels",
                    method = HATT(),
                    preTrainSize = 1,
                    forgetSize = 3,
                    forgetMethod = 1,
                    startParam = (2/(streamSize-preTrainSize)), #esquecer apos o segundo treino(desconsiderando pre treino)
                    trainLimit = None,
                    probabilityOfForget = 1,
                    trainPostForgot = True)
forget.PrepareStream()
forget.PreTrain()
forget.Train()