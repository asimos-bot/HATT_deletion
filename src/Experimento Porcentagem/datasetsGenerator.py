from random import randint


def setsGenerator(forgetArrays, datasetSize, datasetPercentage):
    ForgetArrays = sorted(forgetArrays)
    data = open("../../datasets/movingSquares.data", "r")
    labels = open("../../datasets/movingSquares.labels", "r")
    data.readlines(1);
    labels.readlines(1);
    dataArr = data.readlines();
    dataArr = dataArr[0:datasetSize]
    labelsArr = labels.readlines();
    labelsArr = labelsArr[0:datasetSize]
    data.close()
    labels.close()
    for i in forgetArrays:
        data = open("forgetDatasets/" + str(i) + ".data", "w")
        labels = open("forgetDatasets/" + str(i) + ".labels", "w")
        data.write("a,b\n")
        labels.write("y\n")
        data.close()
        labels.close()

    for i in range(0,datasetSize,int(datasetSize*datasetPercentage)):
        subDataArr = dataArr[i:i+int(datasetSize*datasetPercentage)]
        subLabelsArr = labelsArr[i:i+int(datasetSize*datasetPercentage)]
        for j in reversed(forgetArrays):
            while(len(subDataArr)> j*int(datasetSize*datasetPercentage)):
                pop = randint(0, len(subDataArr)-1)
                subDataArr.pop(pop)
                subLabelsArr.pop(pop)

            data = open("forgetDatasets/" + str(j) + ".data", "a")
            labels = open("forgetDatasets/" + str(j) + ".labels", "a")
            for i in range(len(subDataArr)):
                data.write(subDataArr[i])
                labels.write(subLabelsArr[i])
            data.close()
            labels.close()
