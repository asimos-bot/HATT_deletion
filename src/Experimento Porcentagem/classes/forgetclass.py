import skmultiflow
import numpy as np

if( skmultiflow.__version__ == '0.4.1' ):
    from skmultiflow.trees import HATT as TreeClass
else:
    from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier as TreeClass

import pandas as pd
from classes.label import label
from skmultiflow.data import DataStream

class HATTForget:
    def __init__(self, name = "forget", dataPath = None, labelsPath = None, forgetPercentage = 0.0, method = TreeClass()):
        self.name = name
        self.dataPath = dataPath
        self.labelsPath = labelsPath
        self.forgetPercentage = forgetPercentage
        self.method = method
    #nao usada, serve pra anotar quantos labels diferentes entram em uma janela e suas metricas de avaliacao
    labelsArray = []

    def __PickData(HATTForget):
        try:
            if(HATTForget.dataPath is None):
                print("Dataset para treino não informado")
                return False
            #caso o labels esteja descrito em um csv separado
            if(HATTForget.labelsPath is not None):
                HATTForget.labels = pd.read_csv(HATTForget.labelsPath)
                for label in HATTForget.labels.columns.values:
                    HATTForget.labels[label] = HATTForget.labels[label].astype('category')
                HATTForget.data = pd.read_csv(HATTForget.dataPath)
            else:
                #caso o labels esteja no data
                #para desenvolver !!!
                print("ainda nao desenvolvido(labelsPath nulo)")
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
            else:
                print("Problemas ao configurar os datasets")
        except Exception as e:
            print(e)
            print("Ocorreu um erro(PrepareSream)")

    #nao usado
    def __addLabel(HATTForget, labelName):
        try:
            NewLabel = label(labelName)
            HATTForget.labelsArray.append(NewLabel)
        except Exception as e:
            print(e)
            print("Ocorreu um erro(__addLabel)")

    #nao usado
    def __NextSample(HATTForget, stream):
        X, y = stream.next_sample()
        labelExist = False
        for i in HATTForget.labelsArray:
            if (y[0] == i.name):
                labelExist = True
        if (not labelExist):
            HATTForget.__addLabel(y[0])
        return X, y


    def Train(HATTForget, size):
        size = int(size)
        print(HATTForget.name," treinando com ", size)
        for i in range(size):
            X, y = HATTForget.stream.next_sample()
            HATTForget.method.partial_fit(X, y)


    def Test(HATTForget, size, begin):
        size = int(size)
        begin = int(begin)
        corrects = 0
        print(HATTForget.name, " testando com ", size)
        TestStream = DataStream(HATTForget.data, y=HATTForget.labels)
        TestStream.prepare_for_use()
        if (begin != 0): TestStream.next_sample(begin)
        X, y = TestStream.next_sample((size))
        y = np.expand_dims(y, axis=0).T

        return(HATTForget.method.score(X, y))

    #o esquecimento deve ser no dataset especifico daquela arvore
    def Forget(HATTForget, size, begin):
        size = int(size)
        begin = int(begin)
        ForgetStream = DataStream(pd.read_csv("forgetDatasets/" + str(HATTForget.name)[2:] + ".data"), y=pd.read_csv("forgetDatasets/" + str(HATTForget.name)[2:] + ".labels"))
        ForgetStream.prepare_for_use()
        print(HATTForget.name, " esta esquecendo: ", size*HATTForget.forgetPercentage)
        if (begin != 0): ForgetStream.next_sample(int(begin * HATTForget.forgetPercentage)) 
        for i in range(int(size*HATTForget.forgetPercentage)):
            if (begin != 0): ForgetStream.next_sample(begin)
            X, y = ForgetStream.next_sample()

            if( X.shape[0] == 0 or y.shape[0] == 0 ): continue

            HATTForget.method.partial_fit(X, y, sample_weight=np.full(y.shape, -1))
