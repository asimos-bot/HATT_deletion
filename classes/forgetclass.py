import random

import pandas as pd

from classes.label import label
from skmultiflow.data import DataStream


class HATTForget:
    def __init__(self, name = "forget", dataPath = None, labelsPath = None, forgetdataPath = None, forgetlabelsPath = None, method = None, preTrainSize = 1, forgetSize = 0, forgetMethod = -1, startParam = 0.0, trainLimit = None, probabilityOfForget = 1, trainPostForgot = True):
        self.name = name
        self.dataPath = dataPath
        self.labelsPath = labelsPath
        self.forgetDataPath = forgetdataPath
        self.forgetLabelsPath = forgetlabelsPath
        self.method = method
        self.preTrainSize = preTrainSize
        self.forgetSize = forgetSize
        self.forgetMethod = forgetMethod
        self.startParam = startParam
        self.trainLimit = trainLimit
        self.probabilityOfForget = probabilityOfForget
        self.trainPostForgot = trainPostForgot
    data = None
    labels = None
    forgetData = None
    forgetLabels = None
    labelsArray = []

    def __PickData(HATTForget):
        try:
            if(HATTForget.dataPath is None):
                print("Dataset para treino não informado")
                return False
            #caso o labels esteja descrito em um csv separado
            if(HATTForget.labelsPath is not None):
                HATTForget.labels = pd.read_csv(HATTForget.labelsPath)
                HATTForget.data = pd.read_csv(HATTForget.dataPath)
            else:
                #caso o labels esteja no data
                #para desenvolver !!!
                print("ainda nao desenvolvido(labelsPath nulo)")
                pass
            if(HATTForget.forgetDataPath is not None):
                if (HATTForget.forgetLabelsPath is not None):
                    HATTForget.forgetLabels = pd.read_csv(HATTForget.forgetLabelsPath)
                    HATTForget.forgetData = pd.read_csv(HATTForget.forgetDataPath)
                else:
                    # caso o labels do dataset a ser esquecido esteja no data
                    # para desenvolver !!!
                    print("ainda nao desenvolvido(forgetLabelsPath nulo)")
                    pass
            return True
        except Exception as e:
            print(e)
            print("Ocorreu um erro(PickData)")
            return False

    def PrepareStream(HATTForget):
        try:
            if(HATTForget.__PickData()):
                HATTForget.stream = DataStream(HATTForget.data, y=HATTForget.labels)
                HATTForget.stream.prepare_for_use()
                HATTForget.streamToBeForgotten = None
                if(HATTForget.forgetData is not None):
                    HATTForget.streamToBeForgotten = DataStream(HATTForget.forgetData, y=HATTForget.forgetLabels)
                    HATTForget.streamToBeForgotten.prepare_for_use()

            else:
                print("Problemas ao configurar os datasets")
        except Exception as e:
            print(e)
            print("Ocorreu um erro(PrepareSream)")

    def PreTrain(HATTForget):
        try:
            i = HATTForget.preTrainSize
            while(i>0 and HATTForget.stream.has_more_samples()):
                X, y = HATTForget.__NextSample(HATTForget.stream)
                HATTForget.method.partial_fit(X, y)
                i = i - 1
        except Exception as e:
            print(e)
            print("Ocorreu um erro (PreTrain)")

    def __addLabel(HATTForget, labelName):
        try:
            print("Cadastrando o label ", labelName)
            NewLabel = label(labelName)
            HATTForget.labelsArray.append(NewLabel)
        except Exception as e:
            print(e)
            print("Ocorreu um erro(__addLabel)")

    def __NextSample(HATTForget, stream):
        X, y = stream.next_sample()
        labelExist = False
        for i in HATTForget.labelsArray:
            if (y[0] == i.name):
                labelExist = True
        if (not labelExist):
            HATTForget.__addLabel(y[0])
        return X, y


    def Train(HATTForget):
        try:
            trainSize = 0
            forgetSize = 0
            corrects = 0
            while (HATTForget.stream.has_more_samples() and (HATTForget.trainLimit is None or trainSize > HATTForget.trainLimit)):
                Forgotten = False
                # forget conditions
                if(HATTForget.streamToBeForgotten is not None and(forgetSize < HATTForget.forgetSize) and
                        ((HATTForget.trainLimit is not None and trainSize/HATTForget.trainLimit >= HATTForget.startParam)
                or (HATTForget.trainLimit is None and trainSize/(HATTForget.stream.n_samples-HATTForget.preTrainSize) >= HATTForget.startParam))):
                    if(HATTForget.forgetMethod == 1 and random.randint(1, 10000) <= HATTForget.probabilityOfForget*10000):
                        X, y = HATTForget.__NextSample(HATTForget.streamToBeForgotten)
                        HATTForget.method.partial_fit(X, y, sample_weight=[-1])
                        forgetSize = forgetSize + 1
                        Forgotten = True


                # train
                if((Forgotten and HATTForget.trainPostForgot) or not Forgotten):
                    X, y = HATTForget.__NextSample(HATTForget.stream)
                    my_pred = HATTForget.method.predict(X)
                    if y[0] == my_pred[0]:
                        corrects = corrects + 1
                    for i in HATTForget.labelsArray:
                        if y[0] == my_pred[0]:
                            if y[0] == i.name:
                                i.tp += 1
                            else:
                                i.tn += 1
                        else:
                            if y[0] == i.name:
                                i.fn += 1
                            elif my_pred[0] == i.name:
                                i.fp += 1
                    HATTForget.method.partial_fit(X, y)
                    trainSize = trainSize + 1
            print("previsoes corretas: ", corrects)
            print(float(corrects / (trainSize)))
            for i in HATTForget.labelsArray:
                i.Show(trainSize)
            print(forgetSize)
        except Exception as e:
            print(e)
            print("Ocorreu um erro (Train)")
