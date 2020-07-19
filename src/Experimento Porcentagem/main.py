import copy

from classes.forgetclass import HATTForget
#tamanho total do set a ser lido
size = 1000
#porcentagem do dataset que equivale a uma window, exemplo size 1000 e percentage = 0.1 entao cada window tem 100 elementos
percentage = 0.1
#as porcentagens de esquecimento do experimento
forgetParams = [0.1, 0.25, 0.5, 0.75]

#gerando o inicio do csv
exitLabels = ["main"] + forgetParams
exitLine = ""
Labels = ""
for i in exitLabels:
    Labels = Labels + str(i) + ","
f = open("exit.csv", "w")
Labels = Labels[:len(Labels)-1]
f.write(Labels)
f.close()

#array auxiliar para abrigar as arvores clonadas que serão usadas para esquecimento
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
    #testa a window nas arvores e pega as acuracias
    exitLine = exitLine + str(MainHATT.Test(size*percentage, i)) + ","
    for j in forgetHATT:
        exitLine = exitLine + str(j.Test(size*percentage, i)) + ","

    #atualiza csv
    exitLine = exitLine[:len(exitLine)-1]
    f = open("exit.csv", "a")
    f.write("\n")
    f.write(exitLine)
    f.close()
    exitLine = ""

    #zera array de arvores
    forgetHATT.clear()