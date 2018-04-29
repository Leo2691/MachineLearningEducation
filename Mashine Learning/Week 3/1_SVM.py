import pandas as pd
import numpy as np
import matplotlib
import mlxtend
from mlxtend.plotting import plot_decision_regions

from sklearn.svm import SVC
import matplotlib.pyplot as plt



"""Метод опорных веторов"""


def SVMMethod():
    # загружаем из файла без заголовка header = None
    X = pd.read_csv("/Users/lev/Образование/Машинное обучение/Week 3/3_work1_support vector machines/svm-data.csv", usecols=[1], header=None)
    Y = pd.read_csv("/Users/lev/Образование/Машинное обучение/Week 3/3_work1_support vector machines/svm-data.csv", usecols=[0], header=None)
    print(X, Y)

    # условия линейно разделимой выборки
    svm = SVC(kernel='linear', C=100000, random_state=241)

    svm.fit(X, Y)

    #X_combined_std = np.vstack((X))
    #y_combined_std = np.vstack((Y))
    # test_idx = range(X_combined_std.shape[0], )

    # Plotting decision regions
    plot_decision_regions(X, Y, clf=svm, legend=1)

    # Adding axes annotations
    plt.xlabel('1')
    plt.ylabel('1')
    plt.title('SVM')
    plt.show()

    print(1)


SVMMethod()
