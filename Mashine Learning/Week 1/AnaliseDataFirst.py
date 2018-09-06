import pandas as pn
import numpy as np

data = pn.read_csv("D:/titanic.csv", index_col='PassengerId') #колонка PassengerId задает нумерацию строк датафрейма

def Experiments():
    date20 = data[10:20] #диапозон значений
    headAndSample = data[0:1] #заголовок и первая строка

    dt = data.head(5) #5 первых строк фрейма

    dt = data.head(0)

    dt1 = data.head(1)

    print(dt)
    # print(dt1)

def count_Mens_Wom():
    n = data['Sex'].values
    m = data['Sex'].value_counts()
    print(n, m)

def Survived():
    n = data['Survived'].value_counts()

    perc = (n[1] / (n[1] + n[0])) * 100

    print (n, perc)

def Pclass():
    pcl = data['Pclass'].value_counts()

    first = pcl[1] * 100 / (np.sum(pcl))

    print(pcl, first)

def Age():
    pcl = data['Age'].mean()

    pcl1 = data['Age'].median()
    #first = pcl[1] * 100 / (np.sum(pcl))
    print(pcl, pcl1)

def PirsonCor():
    pcl = data['SibSp'].corr(data['Parch'], method='pearson')

    #pcl1 = data['Parch'].corr(data['SibSp'], method='pearson')

    print (pcl)

#def mostPopularName():
    #pcl = data.Name[data['Sex'].]

    #print(pcl)

Experiments()
#count_Mens_Wom()
#Survived()

Pclass()

#Age()

#PirsonCor()

#mostPopularName()

aaaaa1 = data.groupby('Sex')[['Survived']].apply(np.mean)
aaaaa2 = data.groupby('Sex')['Survived'].apply(np.mean)

aaaaa3 = data.groupby('Sex')[['Survived']].aggregate(np.mean)
aaaaa4 = data.groupby('Sex')[['Survived']].mean()

print(aaaaa1)
print('\n', aaaaa2)
print('\n', aaaaa3)
print('\n', aaaaa4)

aaaaa5 = data.groupby(['Sex', 'Pclass'])['Survived'].aggregate(np.mean)
aaaaa6 = data.groupby(['Sex', 'Pclass'])['Survived'].aggregate(np.mean).unstack()
aaaaa7 = data.pivot_table('Survived', index='Sex', columns='Pclass')

print('\n', aaaaa5)
print('\n', aaaaa6)
print('\n', aaaaa7)




