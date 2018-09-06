"""Добавляем данные из вспомогательных таблиц----------------------------------------------------------------------------------"""

import gc
import os

PATH = "../input/"

print(os.listdir(PATH))

gc.collect()

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold

from lightgbm import LGBMClassifier

data = pd.read_csv(PATH + 'application_train.csv')
test = pd.read_csv(PATH + 'application_test.csv')
#prev = pd.read_csv(PATH + 'previous_application.csv')
buro = pd.read_csv(PATH + 'bureau.csv')
buro_balance = pd.read_csv(PATH + 'bureau_balance.csv')
#credit_card  = pd.read_csv(PATH + 'credit_card_balance.csv')
#POS_CASH  = pd.read_csv(PATH + 'POS_CASH_balance.csv')
#payments = pd.read_csv(PATH + 'installments_payments.csv')



#Отделяем метки
y = data['TARGET']
del data['TARGET']

#кодирем категориальные признаки

categorial_features = [col for col in data.columns if data[col].dtype == 'object']

one_hot_df = pd.concat([data, test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorial_features)

data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]

print('Формат тренировочной выборки', data.shape)
print('Формат тестовой выборки', test.shape)

#работа с признаками
buro_group_size = buro_balance.groupby('SK_ID_BUREAU')

print (1)