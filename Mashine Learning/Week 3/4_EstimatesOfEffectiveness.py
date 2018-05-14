import pandas as pd
import numpy as np
import sklearn.metrics as metr

def Metrics():

    data = pd.read_csv("d:\classification.csv", header=None)
    data = data.drop([0])
    data = data.reset_index()

    y_true = data[0]
    y_pred = data[1]

    #переменные для водной таблицы ошибок и вреных срабатываний алгоритма
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in np.arange(0, len(y_true)):

        y_t = int(y_true[i])
        y_p = int(y_pred[i])

        if y_t == 1 and y_p == 1:
            tp += 1
        if y_t == 0 and y_p == 1:
            fp += 1
        if y_t == 1 and y_p == 0:
            fn += 1
        if y_t == 0 and y_p == 0:
            tn += 1

    Accur = (tp + tn) / (tp + fp + fn + tn)
    #общее число правильных ответов по отношению ко всей выборке
    acc = metr.accuracy_score(y_true=y_true, y_pred=y_pred)
    #полнота
    prec = metr.precision_score(y_true=y_true, y_pred=y_pred, average=None)
    #точность
    recall = metr.recall_score(y_true=y_true, y_pred=y_pred, average=None)
    #F-мера
    f1 = metr.f1_score(y_true=y_true, y_pred=y_pred, average=None)

    print(Accur, prec, recall, f1)

#функуия для поиска максимального значения точности работы методы при полноте >= 70%
def maxprec(y_true, y_prec):

    #precision_recall_curve возвращает 3 массива: точность, полнота, шаг
    prec, recall, trees = metr.precision_recall_curve(y_true=y_true, probas_pred=y_prec)

    max = 0

    for i in np.arange(0, len(prec)):
        if recall[i] >= 0.7 and max < prec[i]:
            max = prec[i]

    return max


def Metrics_1():
    scores = pd.read_csv("D:\scores.csv")

    #считаем площадь под roc кривой
    roc_auc_logreg = metr.roc_auc_score(scores['true'], scores['score_logreg'])
    roc_auc_svm = metr.roc_auc_score(scores['true'], scores['score_svm'])
    roc_auc_knn = metr.roc_auc_score(scores['true'], scores['score_knn'])
    roc_auc_tree = metr.roc_auc_score(scores['true'], scores['score_tree'])

    #печатаем и выбираем наилучший результат // лучший - score_logreg
    print(roc_auc_logreg, roc_auc_svm, roc_auc_knn, roc_auc_tree)

    #создаем словарь с названием метлда и его результатми точности при полноте >= 70%
    grid = {'score_logreg' : maxprec(scores['true'], scores['score_logreg']),
            'score_svm': maxprec(scores['true'], scores['score_svm']),
            'score_knn': maxprec(scores['true'], scores['score_knn']),
            'score_tree': maxprec(scores['true'], scores['score_tree'])
            }
    # лучший - score_tree
    print(grid)


#Metrics()
Metrics_1()
