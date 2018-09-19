import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns

"""----------------------------------------------------------------------------------------------------------"""
def missing_values_table(df):
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(df)

    #всего не достает
    mis_val = df.isnull().sum()

    #Процент недостащих данных
    mis_val_percent = 100 * mis_val / len(df)

    #таблица с результатами
    mis_val_table = pd.DataFrame(pd.concat([mis_val, mis_val_percent], axis=1))

    # Переименование столбцов
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    #Сортировка по проценту

    #dataframe со строками, где хтатистика показавает наличие пропусков. [обращение по условию] возвращает подвыборку данных
    # iloc используем потому что обращаемся по условию
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(by=['% of Total Values'], ascending=False).round(1)


    #print(mis_val_table_ren_columns)

    return(mis_val_table_ren_columns)

#пропуски в графическом виде

def plt_missing(app_train, app_test):

    plt.style.use('seaborn-talk')

    fig = plt.figure(figsize=(18, 6))
    miss_train = pd.DataFrame((app_train.isnull().sum())*100/app_train.shape[0]).reset_index()

    miss_test = pd.DataFrame(app_test.isnull().sum() * 100 / app_test.shape[0]).reset_index()

    miss_train["type"] = "Тренировочная"
    miss_test["type"] = "Тестовая"

    missing = pd.concat([miss_train, miss_test], axis=0)
    sns.pointplot("index",0,data=missing,hue="type")
    #plt.xticks(rotation = 90, fountsize = 7)
    plt.title("Доля отсутсвующих значений в данных")
    plt.ylabel("Доля в %")
    plt.xlabel("Столбцы")

    plt.show()