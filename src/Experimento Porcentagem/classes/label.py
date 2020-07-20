class label:
    def __init__(self, name):
        self.name = name
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    def Accuracy(label, total):
        try:
            return (label.tn+label.tp)/total
        except ZeroDivisionError:
            print("Erro, total = 0, impossivel calcular accuracy")
            return -1

    def Recall(label):
        try:
            return (label.tp) / (label.tp + label.fn)
        except ZeroDivisionError:
            print("Erro, tp + fn  = 0, impossivel calcular Recall")
            return -1

    def Precision(label):
        try:
            return (label.tp) / (label.tp + label.fp)
        except ZeroDivisionError:
            print("Erro, tp + fp  = 0, impossivel calcular Precision")
            return -1
    def ShowPartials(label):
        print("tp: ", label.tp,"tn: ", label.tn,"fp: ", label.fp,"fn: ", label.fn)
    def Show(label, total):
        print("\nlabel: ", label.name,"\nAccuracy: ", label.Accuracy(total),"\nRecall: ", label.Recall(),"\nPrecision: ", label.Precision(), "\n\n")
