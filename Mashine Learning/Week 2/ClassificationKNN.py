import pandas as nd
import numpy as np
import sklearn.model_selection as sk
import os

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import scale

from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier


"""Пример работы классификатора"""
def classification():

    PATH = os.path.dirname(os.path.realpath(__file__))

    print (os.listdir(PATH))

    #app_train = pd.read_csv(PATH + "train_1.8.csv", encoding = "cp1251", decimal=',')
    #app_test = pd.read_csv(PATH + "test_1.9.csv", encoding = "cp1251")
    
    X = nd.read_csv(PATH + "/wine.csv", usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    #print(pX.shape)

    Y = nd.read_csv(PATH + "/wine.csv", usecols=[0])

    #X = np.array(pX)
    #Y = np.array(pY)

    #классификатор ближайших соседей
    clf = KNeighborsClassifier(n_neighbors=5)
    #обучение классификатора
    clf.fit(X=X, y=Y)

    #создаем генератор разбиейний выборки на обучающую и тестовую
    kf = sk.KFold(n_splits=5,shuffle=True,random_state=42)

    #проверяем, как работает генератор
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = Y.values[train_index], Y.values[test_index]

    #result = cross_validation.cross_val_score(estimator = clf, X=X, y=Y,cv = kf, scoring="accuracy")

    #вычисляем ошибку обучения на разбитой выботке (процент распознвание для каждого куска выборки)
    result = cross_val_score(estimator = clf, X=X, y=Y,cv = kf, scoring="accuracy")

    pred = clf.predict(X[:].values[0].reshape((1, -1)))

    #среднее значение распознавания
    mean = np.array(result).mean()

    def Simple():
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3, 4, 5])

        kf = sk.KFold(n_splits=2)
        # kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


"""Выполняем задание"""
def classification_cross_val():
    X = nd.read_csv("D:\wine.csv", usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    #print(pX.shape)

    #масштабирование данных функция sklearn.preprocessing.scale
    X = scale(X)

    Y = nd.read_csv("D:\wine.csv", usecols=[0])

    #X = np.array(pX)
    #Y = np.array(pY)

    #создаем генератор разбиейний выборки на обучающую и тестовую
    kf = sk.KFold(n_splits=5,shuffle=True,random_state=42)

    #создаем массив, где будем хранить усрденнный процент распозанвания по разбитой выборке
    meanK = np.array([])
    for k in range(1, 51):
        # классификатор ближайших соседей
        clf = KNeighborsClassifier(n_neighbors=k)
        # обучение классификатора
        clf.fit(X=X, y=Y)

        #вычисляем ошибку обучения на разбитой выботке (процент распознвание для каждого куска выборки)
        result = cross_val_score(estimator = clf, X=X, y=Y,cv = kf, scoring="accuracy")

        #среднее значение распознавания для k-соседей
        meanK = np.append(meanK, np.array(result).mean())

    max = np.where(meanK == meanK.max())

    print(max)





    """for train_index, test_index in kf(X, Y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]"""








#Simple()
classification()
#classification_cross_val()