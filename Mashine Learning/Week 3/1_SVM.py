import pandas as pd
import numpy as np
import matplotlib
import mlxtend
from mlxtend.plotting import plot_decision_regions

from sklearn.svm import SVC
import matplotlib.pyplot as plt



"""Метод опорных веторов"""


def SVMMethod():
    # загружаем из файла без заголовка header = None ДЛЯ MAC OS
    #X = pd.read_csv("/Users/lev/Образование/Машинное обучение/Week 3/3_work1_support vector machines/svm-data.csv", usecols=[1], header=None)
    #Y = pd.read_csv("/Users/lev/Образование/Машинное обучение/Week 3/3_work1_support vector machines/svm-data.csv", usecols=[0], header=None)

    # загружаем из файла без заголовка header = None ДЛЯ WINDOWS
    X = pd.read_csv("D:\svm-data.csv", usecols=[1, 2], header=None)
    Y = pd.read_csv("D:\svm-data.csv", usecols=[0], header=None)
    # print(Y, X)

    # условия линейно разделимой выборки
    svm = SVC(kernel='linear', C=100000, random_state=241)

    svm.fit(X, Y)

    # print(svm.support_vectors_)


    # print(1)

    X = pd.DataFrame.as_matrix(X)
    Y = pd.DataFrame.as_matrix(Y)

    """plot_decision_regions требует, чтобы y был типа int и передавался в формате [0,1, ..., 1]"""
    Y = Y.astype(int)
    """flatten возвращает одномерный (сглаженый) массив из формата n x m"""
    Y = Y.flatten()

    # X = np.ndarray(X)
    # Y = np.ndarray(Y)
    # X_combined_std = np.vstack((X))
    # y_combined_std = np.vstack((Y))
    # test_idx = range(X_combined_std.shape[0], )

    # Рисуем график
    plot_decision_regions(X, Y, clf=svm, legend=2)
    plt.xlabel('1')
    plt.ylabel('1')
    plt.title('SVM')
    plt.show()

SVMMethod()
