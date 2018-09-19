import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Imputer

"""Осуществляем выгрузку основного файла"""
pd.set_option('display.max_columns', None)

PATH = os.path.dirname(os.path.realpath(__file__))  + "/../input/"

print (os.listdir(PATH))

app_train = pd.read_csv(PATH + "train_1.8.csv", encoding = "cp1251", decimal=',')
app_test = pd.read_csv(PATH + "test_1.9.csv", encoding = "cp1251")

dif = [col for col in pd.DataFrame(app_train).columns if col not in pd.DataFrame(app_test).columns]
dif.append('Скважина')


"""y = [app_train['Нефть, т'][np.isnan(app_train['Нефть, т']) == False ]]
#y = y.reshape(len(y) - 1, 1)

imputer = Imputer(strategy='median')

y_not_nan = imputer.fit_transform(y)
y_mean = np.mean(y_not_nan)

#Датафрейм для загрузки
submit = {"id":pd.Series([i for i in np.arange(len(app_test) * 6)],),
     "predict": pd.Series([y_mean for i in np.arange(len(app_test) * 6)])}

#Сохранение датафрейма
pd.DataFrame(submit).to_csv('mean.csv', index = False)"""


selectDF = pd.DataFrame(app_train[["Скважина", "Дата", "Нефть, т"]])

а = selectDF.groupby('Скважина')['Дата'].min()

list = []
list_id = []
list_data = []
for name, group in selectDF.groupby('Скважина'):
    list += [i for i in np.arange(group.shape[0])]

    for i in np.arange(group.shape[0]):
        list_id.append(name)
        a = group['Дата'].reset_index()
        list_data.append(a['Дата'].loc[i])

ll = {'Скважина': list_id, 'Номер месяца': list, 'Дата': list_data}

dataframe = pd.DataFrame(ll)

selectDF = selectDF.merge(dataframe, how='left', on=['Скважина', 'Дата'])

y = [selectDF[['Номер месяца', 'Нефть, т']][np.isnan(selectDF['Нефть, т']) == False ]]


mean_mounth = selectDF.groupby('Номер месяца').mean()['Нефть, т']

list = []
d = 0
for i in np.arange(len(app_test)):
    for j in np.arange(1, len(mean_mounth)):
           list.append(mean_mounth[j])
    list.append(mean_mounth[len(mean_mounth) - 1])


#Датафрейм для загрузки
submit = {"id":pd.Series([i for i in np.arange(len(app_test) * 6)],),
     "predict": pd.Series(list)}

#Сохранение датафрейма
pd.DataFrame(submit).to_csv('mean_mouth.csv', index = False)




#data = data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

"""Получаем номер месяца довольно криво"""
i = 0
corr_id = ""
new_feat = pd.Series()

for i, elem in enumerate(selectDF["Скважина"].values):
    if(corr_id != elem):
        j = 0
        corr_id = elem
    selectDF["Дата"].loc[i] = j
    j += 1



s = pd.Series([10, 20, 30, 40, 50], ['a', 'b', 'c', 'd', 'e'])
    #print(i, elem)


print(1)



#print("Размерность обучающей выборки ", app_train.shape)
#print("Размерность тестовой выборки ", app_test.shape)

#dif_features = app_train[dif]

#print(dif)