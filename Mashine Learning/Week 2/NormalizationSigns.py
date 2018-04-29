import numpy as np
import sklearn.preprocessing as pr
import pandas as pn
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def PerseptronSimple():
    X = np.array([[1,2], [3,1], [5,6]])
    y = np.array([0, 1, 0])

    #обучение персептрона
    clf = Perceptron()
    clf.fit(X=X,y=y)

    #прогноз на основе оучения
    prediction = clf.predict([[3,1]])

    print(prediction)

def AccuracySimple():

    #объявляем объект класса стандартизации
    scaler = pr.StandardScaler()
    #обучающая и тестовая выборки
    X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
    X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])

    #обучаем стандартизатор на тестовой выборке и выполняем ее стандартизацию
    X_train_scaled = scaler.fit_transform(X=X_train)
    #с помощью найденных при обучении параметров выполняем стандартизацию тестовой выборки
    X_test_scaled = scaler.transform(X_test)

    print (X_train_scaled)
    print (X_test_scaled)

#выполняем задание
def PerseptronTask():

    #загружаем обущающую выборку
    Y_train = pn.read_csv("D:/perceptron-train.csv", usecols=[0])
    X_train = pn.read_csv("D:/perceptron-train.csv", usecols=[1, 2])

    # загружаем тестовую выборку
    Y_test = pn.read_csv("D:/perceptron-test.csv", usecols=[0])
    X_test = pn.read_csv("D:/perceptron-test.csv", usecols=[1, 2])

    #объявляем классификатор
    clf = Perceptron(random_state=241)
    #обучаем
    clf.fit(X=X_train, y=Y_train)

    #выполянем распознавание на тестовой выборке
    yPredNonStand = clf.predict(X_test)

    #сравниваем с выходными результатами тестовой выборки
    result = accuracy_score(y_true=Y_test, y_pred=yPredNonStand)

    print(result)

    scale = pr.StandardScaler()

    X_train_scaled = scale.fit_transform(X_train, y=Y_train)
    X_test_scaled = scale.transform(X_test, y=X_test)

    clf = Perceptron(random_state=241)
    clf.fit(X=X_train_scaled,y=Y_train)
    yPredScaled = clf.predict(X_test_scaled)

    result1 = accuracy_score(y_true=Y_test,y_pred=yPredScaled)

    print(result1)

    print("differense = ", result1 - result)





PerseptronTask()
