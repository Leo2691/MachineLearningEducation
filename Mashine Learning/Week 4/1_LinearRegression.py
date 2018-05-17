import pandas as pd
import numpy as np
import re
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


tfid = TfidfVectorizer(min_df=5)
enc = DictVectorizer()
enc_loc = DictVectorizer()
enc_con = DictVectorizer()



#функцция преобразования текста. Передаем массив данных, признак и указатель, является выборка тестовой или обучающей
def TransformText(data, column, Test = False):
    #приводим к нижнему регистру и заменяем все, что не является буквами и цифрами на пробелы
    data[column] = data[column].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)

    #вычисляем tf-idf и возвращаем для тестовой выборки transform для обучающей fit_transform
    return tfid.transform(data[column]) if Test else tfid.fit_transform(data[column])

#очищаем признаки от пропущенных значений и выполняем one-hot кодированием
def ReplaseNan(data, column, enc, Test = False):
    data[column].fillna('nan', inplace = True)

    #X = data[[column]].to_dict('records')
    #print(X)

    #to_dict функция преобразования столбца dataFrame в словарь

    return enc.transform(data[[column]].to_dict('records')) if Test else enc.fit_transform(data[[column]].to_dict('records'))





def LinearRegression():

    data = pd.read_csv("D:/salary-train.csv")

    #X_train_text = data["FullDescription"].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
    X_train_text = TransformText(data, "FullDescription", False)

    #data["FullDescription"] = re.sub('[^a-zA-Z0-9]', ' ', data["FullDescription"].str.lower())

    #X_train_loc = ReplaseNan(data, "LocationNormalized", enc_loc, False)
    #X_train_con = ReplaseNan(data, "ContractTime", enc_con, False)

    """data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    X_train_cat = enc_fit.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))"""

    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)

    # Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.

    enc = DictVectorizer()
    X_train_cat = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

    X_train = sp.hstack([X_train_text, X_train_cat])

    #print(X_train)

    y_train = data['SalaryNormalized']
    clf = Ridge(alpha=1)

    clf.fit(X_train, y_train)

    #print(X_train)

    data_test = pd.read_csv("D:/salary-test-mini.csv")

    data_test["FullDescription"] = data_test["FullDescription"].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)

    X_test_text = TransformText(data_test, "FullDescription", Test=True)

    X_test_loc = ReplaseNan(data_test, "LocationNormalized", enc, Test=True)
    X_test_cat = ReplaseNan(data_test, "ContractTime", enc, Test=True)

    X_test = sp.hstack([X_test_loc, X_test_cat, X_test_text])

    #print(X_test)

    result = clf.predict(X_test)

    print(result)

    #data["ContractTime"].to_csv("D:/salary-train1.csv")

    #print(data["LocationNormalized"])"""

LinearRegression()