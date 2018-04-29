import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def SipleWorkDecicsionTree():
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

    y = np.array([0, 1, 0])

    clf = DecisionTreeClassifier()
    #обучение классификатора
    A = clf.fit(X, y)

    importances = clf.feature_importances_

    f = 0

def loading_data():


    #Извлекаем данные из нужных столбцов
    #Y  Survived

    dataNonFilterX = pd.read_csv("D:/titanic.csv", usecols= ['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])#, index_col='PassengerId')
    count = dataNonFilterX.count()

    i = 0
    # заменяем строки на числа, так как обучаение деревьев происходит только на числах
    for item in dataNonFilterX['Sex']:
        if (item == 'male'):

            dataNonFilterX['Sex'][i] = 1
            i += 1
        else:
            dataNonFilterX['Sex'][i] = 0
            i += 1

    #Обработка пропусков
    dataX = pd.DataFrame(dataNonFilterX.dropna())
    #после удаления строк с пропусками делаем переиндексацию датафрейма
    dataX = dataX.reset_index()
    dataX = dataX.drop('index', 1)  # удаляем столбец index из выборки X
    count1 = dataX.count()

    #созжажим новый datafraym c выходами Y
    dataY = pd.DataFrame(dataX['Survived'])

    dataX = dataX.drop('Survived', 1)#удаляем столбец Survived из выборки X

    #обучаем дерево решений
    clf = DecisionTreeClassifier(random_state=241)
    A = clf.fit(dataX, dataY)
    print(A)

    #print(dataX)

    #A = clf.fit(dataX, dataY)

    #print(A)

    return dataX



#SipleWorkDecicsionTree()

data = loading_data()

SipleWorkDecicsionTree()