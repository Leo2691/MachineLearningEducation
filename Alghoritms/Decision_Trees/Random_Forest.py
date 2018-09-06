import numpy as np
import random as rn
import math as mt
import matplotlib.pyplot as plt
import pandas as pd

from Decision_Tree import buildtree
from Decision_Tree import predict
from Extracting_Data import extracting_data

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zeland', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zeland', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


#считеаем уникальные элементы в списке и их количество
def uniquecounts(array):
    results = {}

    for elem in array:

        if elem not in results:
            results[elem] = 0
        results[elem] += 1

    return results


def random_forest(rows, count_trees = 100):

    #создаем бутстрап выборку
    #генерируем массив индексов для будущей тестовой и обучающих выборок
    ind_tr = [rn.randint(0, len(rows) - 1) for i in np.arange(len(rows))]
    ind_test = [i for i in np.arange(len(rows)) if i not in ind_tr]

    #создаем подвыборку
    data_tr = [rows[i] for i in ind_tr]
    data_test = [rows[i] for i in ind_test]

    #случайно выбираем набор признаков для построения дерева
    ind_feat = [rn.randint(0, len(rows[0]) - 2) for i in np.arange(mt.sqrt(len(rows[0])))]

    #получили одно дерево
    rf = buildtree(data_tr, ind_feat)

    #проверяем его на тестовой выборке

    Y_pred = predict(rf, data_test[0]) #for row in data_test]

    return rf


#random_forest(my_data, 100)

path = "D:/1/11/application_train.csv"
data = extracting_data(path, 1000)

R = random_forest(data, 100)

print(1)

#np.

#X.insert(Y, )




"""for name in data.columns:
    s = ''
    s += name
    print(s)"""


"""
lim = 100
list = [rn.randint(0, lim) for i in np.arange(lim)]
elem = uniquecounts(list)
count = len(elem)
plt.hist(list)
plt.show()
print(elem)
print(count)
"""
