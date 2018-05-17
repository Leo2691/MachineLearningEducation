import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def StockIndex():

    data = pd.DataFrame([])
    data = pd.read_csv("D:/close_prices.csv")

    #есть исходная матрица с 375 объектами и 30 признаками
    data = data.drop("date", axis=1)

    #Объявляем преобразование метода главных комопнент с количесвтом компонент = 10
    clf = PCA(n_components=10)

    #обучили трансформатор
    clf.fit(data)

    count = 0
    sum = 0

    #ищем какое каличество информативных признаков составляет 90% от всей суммарной дисперсии датафрейма
    #ищем в параметрах классфикатора
    while(sum <= 0.9):
        sum += clf.explained_variance_ratio_[count]
        count += 1

    print(clf.explained_variance_ratio_)
    print("количесвто трасформированных признаков, объясняющих 90% дисперсии = ", count)

    #преобразуем исходную матрицу: снижаем размерность с 30 до 10
    #получили матрицу 10 признаков, 375 объектов
    data_trans = pd.DataFrame(clf.transform(data))

    #загружаем файл с тестовыми данными. Индекс DowJouns
    data_dow_jones = pd.read_csv("D:/djia_index.csv")

    data_dow_jones = data_dow_jones.drop("date", axis=1)

    X = data_trans[0]
    #ищем корреляцию между первым компонентом (преобразованным признаком) и данными из теста DowJouns
    pirs = data_dow_jones["^DJI"].corr(X, method='pearson')

    print("корреляция между первым трансформированным признаком и общим ондексом биржи = ", pirs)

    #первый компонент преобразованной матрицы запихиваем в Series
    w_coef = pd.Series(clf.components_[0])

    #сортируем и находим индекс компании, которая вносит максимальный вклад к преобразованные компоненты
    comp0_w_top = w_coef.sort_values(ascending=False).head(1).index[0]

    comp = data.columns[comp0_w_top]

    print(comp)





StockIndex()