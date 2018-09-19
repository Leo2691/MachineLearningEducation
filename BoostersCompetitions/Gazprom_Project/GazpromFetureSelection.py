import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
from BoostersCompetitions.Gazprom_Project import NaNStatistic as NaNst

"""Осуществляем выгрузку основного файла"""
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

PATH = os.path.dirname(os.path.realpath(__file__))  + "/../input/"

print (os.listdir(PATH))

"""обучающая и тестовая"""
app_train = pd.read_csv(PATH + "train_1.8.csv", encoding = "cp1251")#, decimal=',')
app_test = pd.read_csv(PATH + "test_1.9.csv", encoding = "cp1251")

print("Размерность обучающей выборки ", app_train.shape)
print("Размерность тестовой выборки ", app_test.shape)

#заменяем точки на запятые сохраняем в файл и потом считываем снова | КУСОК ЗАКОММЕНТИТЬ!

"""list_cat = [col for col in app_train.columns if app_train[col].dtype == 'object']
list_cat_test = [col for col in app_test.columns if app_test[col].dtype == 'object']

list_cat.remove('Состояние')

df = app_train[list_cat]

for col in list_cat:
    app_train[col] = app_train[col].str.replace(',', '.')

for col in list_cat_test:
    app_test[col] = app_test[col].str.replace(',', '.')

pd.DataFrame(app_train).to_csv(PATH + "train_1.8.csv", encoding = "cp1251", index = False)
pd.DataFrame(app_test).to_csv(PATH + "test_1.9.csv", encoding = "cp1251", index = False)"""

#target
y = app_train["Нефть, т"]

"""Список столбцов, которых нет в обучающей выборке"""
dif = [col for col in pd.DataFrame(app_train).columns if col not in pd.DataFrame(app_test).columns]

#Выравниваем выборку
app_train = pd.DataFrame(app_train).drop(dif, axis=1)

"""столбец с номером месяца----------------------------------------------------------------------------"""
selectDF = pd.DataFrame(app_train[["Скважина", "Дата"]])

list = []
for name, group in selectDF.groupby('Скважина', sort=False):
    list += [i for i in np.arange(group.shape[0])]

app_train['Номер месяца'] = pd.Series(list)

selectDF_test = pd.DataFrame(app_test[["Скважина", "Дата"]])
list = []
for name, group in selectDF_test.groupby('Скважина', sort=False):
    list += [i for i in np.arange(group.shape[0])]

app_test['Номер месяца'] = pd.Series(list)

"""Статистика по пропускам"""
#print(missing_values_table(app_train))
"""---------------------------------------"""

"""Преобразование признаков-----------------------------------------------------------------------------"""
"""Исключаем из исследования признаки, где процент пропусков > 90"""
stat = pd.DataFrame(app_train).isnull().sum() * 100 / len(app_train)
list_drop = pd.DataFrame(app_train).isnull().columns[pd.DataFrame(app_train).isnull().sum() * 100 / len(app_train) > 90]

app_train = pd.DataFrame(app_train).drop(list_drop, axis=1)
app_test = pd.DataFrame(app_test).drop(list_drop, axis=1)

print("Размерность обучающей выборки ", app_train.shape)
print("Размерность тестовой выборки ", app_test.shape)

"""Работа с категориальными признаками-----------------------------------------"""
#список категориальных признаков
list_cat = [col for col in app_train.columns if app_train[col].dtype == 'object']
list_cat.remove('Скважина')

app_train[list_cat] = app_train[list_cat].fillna("Не известно")
app_test[list_cat] = app_test[list_cat].fillna("Не известно")

"""статистика по пропускам в категориальных признаках"""
#print(missing_values_table(app_train[list_cat]))

"""Остальные вещественные столбцы заполняем путые значения на -999999"""

list_float_nan = pd.DataFrame(app_train).isnull().columns[pd.DataFrame(app_train).isnull().sum()  * 100 / len(app_train) != 0]
print(list_float_nan)

#убедились, что все оставшиеся столбцы - float
list_types = pd.DataFrame(app_train[list_float_nan]).dtypes

app_train[list_float_nan] = app_train[list_float_nan].fillna(-999999)
app_test[list_float_nan] = app_test[list_float_nan].fillna(-999999)

"""статистика по пропускам в вещественных признаках"""
#print(NaNst.missing_values_table(app_train))
#print(NaNst.missing_values_table(app_test))

print("Размерность обучающей выборки ", app_train.shape)
print("Размерность тестовой выборки ", app_test.shape)

#список категориальных признаков
#list_cat = [col for col in app_train.columns if app_train[col].dtype == 'object']
#list_cat = list_cat.remove('Скважина')

"""one hot кодирование для категориальных признаков"""
#объединяем обучающую и тестовую выборки
one_hot = pd.concat([app_train, app_test])
print("Размерность выборки для кодирования", one_hot.shape)
one_hot = pd.get_dummies(one_hot, columns=list_cat)
print("Размерность выборки после кодирования", one_hot.shape)

app_train = one_hot[:].iloc[:app_train.shape[0],:]
app_test = one_hot.iloc[app_train.shape[0]:,]

print("Размерность обучающей выборки ", app_train.shape)
print("Размерность тестовой выборки ", app_test.shape)
#print(list_cat)

app_train["Нефть, т"] = y
print(NaNst.missing_values_table(app_train))

"""очищаем обучающую выборку от объектов, где значение "Нефть т" NaN"""
#можно попробовать заменить средним по каждому
app_train = pd.DataFrame(app_train).dropna()

print("Размерность обучающей выборки ", app_train.shape)
print("Размерность тестовой выборки ", app_test.shape)

"""делим выборку на 6 частей для обучения------------------------------------------"""

list_dict = []
for name, group in app_train.groupby('Скважина', sort=False):
    count = pd.DataFrame(group['Нефть, т']).shape[0]
    elem = dict(('Нефть_' + str(i + 1), pd.Series(group['Нефть, т']).values[i]) for i in np.arange(count))
    elem['Скважина'] = name
    list_dict.append(elem)

additional = pd.DataFrame(list_dict)
print("Размерность обучающей выборки ", app_train.shape)
print("Размерность дополнительной части выборки ", additional.shape)

full_train = pd.merge(app_train, additional, on='Скважина')
print("Размерность расширенной обучающей выборки ", full_train.shape)



print(1)