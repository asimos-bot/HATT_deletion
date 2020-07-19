import copy

from classes.forgetclass import HATTForget

size = 1000
percentage = 0.1
forgetParams = [0.1, 0.25, 0.5, 0.75]

exitLabels = ["main"] + forgetParams
exitLine = ""
Labels = ""
for i in exitLabels:
    Labels = Labels + str(i) + ","
f = open("exit.csv", "w")
Labels = Labels[:len(Labels)-1]
f.write(Labels)
f.close()

forgetHATT = []
MainHATT =  HATTForget(name = "Main", dataPath = "../../datasets/movingSquares.data", labelsPath = "../../datasets/movingSquares.labels")
MainHATT.PrepareStream()
for i in range(0, size, int(size*percentage)):
    #treina principal
    MainHATT.Train(size*percentage)
    #cria copias
    for j in forgetParams:
        AuxHATT = copy.deepcopy(MainHATT)
        AuxHATT.name = str(j)
        AuxHATT.forgetPercentage = j
        forgetHATT.append(AuxHATT)
    #esquece nas copias
    for j in forgetHATT:
        j.Forget(size*percentage, i)
    exitLine = exitLine + str(MainHATT.Test(size*percentage, i)) + ","
    for j in forgetHATT:
        exitLine = exitLine + str(j.Test(size*percentage, i)) + ","
    exitLine = exitLine[:len(exitLine)-1]
    f = open("exit.csv", "a")
    f.write("\n")
    f.write(exitLine)
    f.close()
    exitLine = ""
    forgetHATT.clear()
    MainHATT.labelsArray = []