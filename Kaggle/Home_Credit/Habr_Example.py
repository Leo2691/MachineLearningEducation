"""На основе https://habr.com/post/414613/ """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

""" 1. Первочное знакомство с данными -------------------------------------------"""
import os
PATH = "../input/"

print(os.listdir(PATH))

app_train = pd.read_csv(PATH + 'application_train.csv')
app_test = pd.read_csv(PATH + 'application_test.csv',)

print(app_train.shape)
print(app_test.shape)

#подробно рассмотрим данные

#pd.set_option('display.max_columns', None)
#print(app_train.head())

"""2. Exploratory Data Analysis или первичное исследование данных -----------------"""

#проверяем распределение целевой переменной

print(app_train['TARGET'].value_counts()) #выборка несбалансирована
print(np.bincount(app_train['TARGET']))   #выборка несбалансирована

#Функция для подсчета недостающих столбцов------------------------------------------------------

def missing_values_table(df):
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


    print(mis_val_table_ren_columns)

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

#missing_values_table(app_train)
#plt_missing(app_train, app_test)

#Смотрим, какие столбцы у нас с категориальными признаками------------------------------------------------------

print(app_train.dtypes.value_counts())

print(app_train.select_dtypes(include=[object]).apply(pd.Series.nunique, axis=0))

#One-hot кодирование категориальных признаков

app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print("Training features shape:", app_train.shape)
print("Testing features shape:", app_test.shape)

"""Так как после кодирования количество признаков 
в тестовой и обучающей выборках различно, нужно убрать из
 обучающей те столбцы, которых нет в тестовой"""

#сохраним лейблы, так как при выравнивании они будут потеряны
train_labels = app_train['TARGET']

# Выравнивание. Сохраним только столбцы, имеющиеся в обоих датафреймах
app_train, app_test = app_train.align(app_test, join = 'inner', axis=1)

print('формат обучающей выборки', app_train.shape)
print('формат тестовой выборки', app_test.shape)

#добавим целевой столбец назад к обечающей выборке

app_train['TARGET'] = train_labels

#Ислледование корреляции данных----------------------------------------------------------------
#корреляция и сортировка # закомментил, так как считается очень долго
"""correlations = pd.Series(app_train.corr()['TARGET']).sort_values()

print("Наивысшая позитивная корреляция: \n", correlations.tail(15))
print("\nНивысшая негативная корреляцияЖ \n", correlations.tail(15))"""

#DAYS_BIRTH - самый сильный корреляционный признак - 0.078239

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

plt.hist(app_train['DAYS_BIRTH'] / 365, bins=25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
#plt.show()

#сгладим гистограмму гаусовским ядром
#KDE зайков выплаченных вовремя
m = app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365

#sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365)

#чистые данные без кодирований и изменений
application_train = pd.read_csv(PATH+"application_train.csv")
application_test = pd.read_csv(PATH+"application_test.csv")
#функция для графического отражения влияния признака на целевую переменную
def plot_stats(feature, label_rotation=False, horizontal_layout=True):
    #получили, сколько у нас по Этому признаку уникальных значений и инициализировали DataFrame
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Количество займов': temp.values})

    #расчет доли Traget=1 в категории
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    #рисуем графики
    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Количество займов", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Доля проблемных', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();


"""plot_stats('NAME_CONTRACT_TYPE')
plot_stats('CODE_GENDER')
plot_stats('FLAG_OWN_CAR')
plot_stats('FLAG_OWN_REALTY')
plot_stats('NAME_FAMILY_STATUS',True, True)
plot_stats('ORGANIZATION_TYPE',True, False)"""

#Распределение сумм кредитования

"""plt.figure(figsize=(12,5))
plt.title("Распределение AMT_CREDIT")
amt_c = pd.Series(app_train["AMT_CREDIT"] / 1000).astype(int)
v = amt_c.max()
ax = sns.distplot(amt_c)"""

"""Обрабатываем признаки------------------------------------------------------------------------------------------------------------"""
#Создаем полиномиальные признаки

#создаем новый DataFrame для полиномиальных признаков
poly_features = pd.DataFrame(app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']], columns=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET'])
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

#обрабатываем отсутсвующие значения

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop('TARGET', axis = 1)

poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

#подгружаем функцию для работы с полиномиальными признаками

from sklearn.preprocessing import PolynomialFeatures

#создадим полиномиальных объект степени 3
poly_transformer = PolynomialFeatures(degree=3)

#тренировка полиномиальных признаков
poly_transformer.fit(poly_features)

poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])

#трансформация признаков
poly_features = pd.DataFrame(poly_transformer.transform(poly_features))
poly_features_test = poly_transformer.transform(poly_features_test)
print("Формат полиномиальных признаков: ", poly_features.shape)

#списвоить признакам имена можно при помощи метода get_future_names
#Новый DataFrame для новых признаков

#---------------------
#так инициализация не работает, поэтому создаем так, как ниже
#poly_features = pd.DataFrame(poly_features, columns=poly_features.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
#---------------------

list_col = poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])
dict_name = {i: list_col[i] for i in np.arange(len(list_col))}
poly_features = pd.DataFrame(poly_features)
poly_features = poly_features.rename(index=str, columns=dict_name)

#добавим в DataFrame целевой столбец
poly_features['TARGET'] = poly_target.values

#расчитываем корреляцию
poly_corrs = poly_features.corr()['TARGET'].sort_values()

print(poly_corrs.head(10))
print(poly_corrs.tail(5))

"""Некоторые признаки имеют более высокую корреляцию, чем исходные
обучим модели классификации и на чистых данных и с дополнительными признаками"""

#создадим DataFrame для тестовых признаков
poly_features_test = pd.DataFrame(poly_features_test)
poly_features_test = poly_features_test.rename(index=str, columns=dict_name)

#объединим тренировочные датафреймы
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR'].values
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how='left')

#объединим тестовые датафреймы
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR'].values
app_test_poly = app_test.merge(poly_features_test, on=['SK_ID_CURR'], how='left')

#выровняем выборки
app_train_poly, app_test_poly = pd.DataFrame(app_train_poly).align(app_test_poly, join='inner', axis=1)

#смотрим формат
print('Тренировочная выборка с полиномиальными признаками: ', app_train_poly.shape)
print('Тестовая выборка с полиномиальными признаками: ', app_test_poly.shape)

"""Обучение моделей----------------------------------------------------------------------------------------------------------------------"""

"""Для Логистической регрессии необходимо:
- закодировать категориальные признаки
- заполнить недостающие данные
- выполнить приведение признаков"""

from sklearn.preprocessing import MinMaxScaler, Imputer

#уберем TARGET из тренировочных даыннх
train = app_train.drop(['TARGET'], axis = 1)
features = list(train.columns)

test = app_test.copy()

#заполняем недостающие значения
imputer = Imputer(strategy='median')

#Нормализация
scaler = MinMaxScaler(feature_range=(0,1))

#заполнение тренировочной выборки
imputer.fit(train)

#трансформирование тренировочной и тестовой выборок
train = imputer.transform(train)
test = imputer.transform(test)

#тоже самое с нормализацией
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Формат тренировочной выборки: ', train.shape)
print('Формат тестовой выборки: ', test.shape)

# Логистическая регрессия-----------------------------------------------
from sklearn.linear_model import LogisticRegression

#создаем модель
log_reg = LogisticRegression(C = 0.0001)

#тренируем модель
#log_reg.fit(train, train_labels)
"""LogisticRegression(C=0.001, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, max_iter=100,
                   multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)"""
"""Теперь модель можно использовать для предсказаний. Метод prdict_proba 
даст на выходе массив m x 2, где m - количество наблюдений, 
первый столбец - вероятность 0, второй - вероятность 1. 
Нам нужен второй (вероятность невозврата)."""

#log_reg_pred = log_reg.predict_proba(test)[:,1]

#submit = app_test[['SK_ID_CURR']]
#log_reg_pred = np.array(log_reg_pred).reshape(-1, 1)
#submit['TARGET'] = log_reg_pred

#submit.to_csv('log_reg_baseline.csv', index = False)

"""Случайный лес---------------------------------------------"""
from sklearn.ensemble import RandomForestClassifier

"""#Создадим классификатор
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50)

#тренировка на обучающей выборке
random_forest.fit(train, train_labels)

#предстказание на тестовых данных
predictions = random_forest.predict_proba(test)[:, 1]

#создаем DataFrame для загрузки

submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

#Сохранение
pd.DataFrame(submit).to_csv('random_forest_baseline.csv', index = False)"""

"""Теперь тестируем случайный лес на полиномиальных признаках--------------"""

""""#Создание и тернировка экземпляра для заполненния пропузщенных данных
imputer = Imputer(strategy='median')

poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

#Нормализация

scaler = MinMaxScaler(feature_range=(0,1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators=100, random_state=50)

#Тренировка на полиномиальных данных
random_forest_poly.fit(poly_features, train_labels)

#предсказания
predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]

#Датафрейм для загрузки
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

#сохранение датафрейма
pd.DataFrame(submit).to_csv('random_forest_baseline_engineered.csv', index = False)"""

"""Попробуем градиентный густинг-------------------------------------------------------"""
from lightgbm import LGBMClassifier

clf = LGBMClassifier()
clf.fit(train,train_labels)

predictions = clf.predict_proba(test)[:, 1]

#Датафрейм для загрузки
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

#Сохранение датафрейма
submit.to_csv('lightgbm_baseline.csv', index = False)

"""Интерпретация модели - важность признаков------------------------------------------------------"""

#функция для расчета важности признаков
def show_feature_importances(model, features):
    plt.figure(figsize = (12, 8))

    #Создание датафрейм фич и их важностей и отсортируем его
    results = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    results = results.sort_values('importance', ascending = False)

    #Отображение
    print(results.head(10))
    print("\n Признаков с важностью выше 0.01 = ", np.sum(results['importance'] > 0.01))

    #график
    results.head(20).plot(x = 'feature', y = 'importance', kind = 'barh', color = 'red', edgecolor = 'k', title = 'Feature Importances')


    return results

feature_importances = show_feature_importances(clf, features)


