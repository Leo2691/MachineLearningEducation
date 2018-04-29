import pandas as pd
import numpy as np

import sklearn.neighbors as ng
import sklearn.preprocessing as sk
import sklearn.datasets as ds
import sklearn.model_selection as ms

from sklearn.model_selection import cross_val_score

def regressorKNN():
    #загружаем данные
    data = ds.load_boston()

    #масштабирование данных
    X = sk.scale(data.data)
    Y = sk.scale(data.target)


    #генератор разбиений
    kf = ms.KFold(n_splits=5, shuffle=True, random_state=42)

    #все возможные варианты параметра p
    dergree = np.linspace(1, 10, 200)
    #массив результатов
    result = np.array([[]])

    r = np.array([])

    for p in dergree:
        reg = ng.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
        reg.fit(X=X, y=Y)

        res = cross_val_score(reg, X=X, y=Y, scoring='neg_mean_squared_error',cv=kf)

        result = np.append(result, [np.array(res).mean(), p])

        r = np.append(r, np.array(res).mean())

    max_mean = np.max(r)
    print(result)
    print(max_mean)

regressorKNN()