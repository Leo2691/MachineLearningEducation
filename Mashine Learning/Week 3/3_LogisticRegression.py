import pandas as pd
import math
import numpy as np
import sklearn.metrics as metr


def Gradient_descent():

    #загрузили данные из файла
    X = pd.read_csv("D:\data-logistic.csv", usecols=[1, 2], header=None)
    y = pd.read_csv("D:\data-logistic.csv", usecols=[0], header=None)

    #веса
    w1 = 0
    w2 = 0

    #коэффициенты уравнения
    k = 0.1 #размер шага
    l = len(y) # нормирующий коэффициент
    C = 10 #сила регуляризации

    #находим сумму

    #цикл по итерациям для расчета w1 и w2
    for j in np.arange(0, 10000):
        sumLog = 0
        for i, row in X.iterrows():
            x0 = float(row.values[0])
            x1 = float(row.values[1])

            #sumLog = sumLog + y.values[i] * x0
            sumLog = sumLog + y.values[i] * x0 * (1.0 - 1.0 / (1.0 + math.exp((-y.values[i] * (w1 * x0 + w2 * x1))))) - k * C * w1
            #i = i + 1

        w1_old = w1
        #обновляем w1 (вес 1)
        w1 = w1 + k * (1.0 / l) * sumLog - k * C * w1

        sumLog = 0

        for i, row in X.iterrows():
            x0 = float(row.values[0])
            x1 = float(row.values[1])

            sumLog = sumLog + y.values[i] * x1 * (1.0 - 1.0 / (1.0 + math.exp((-y.values[i] * (w1 * x0 + w2 * x1))))) - k * C * w2

        w2_old = w2
        # обновляем w2 (вес 2)
        w2 = w2 + k * (1.0 / l) * sumLog - k * C * w2

        #евклидово расстояние между старым и новым векторами w
        diff = np.sqrt(pow((w1_old - w1), 2) + pow((w2_old - w2), 2))


        if diff < 0.00001:
            print("break ", w1, w2, j)
            break

        #print(w1, w2)

    prob = []
    for i, row in X.iterrows():
        x0 = float(row.values[0])
        x1 = float(row.values[1])

        prob.append( 1.0 / (1 + math.exp(-w1 * x0 - w2 * x1)))

    result = metr.roc_auc_score(y, prob)

    print(result)


Gradient_descent()