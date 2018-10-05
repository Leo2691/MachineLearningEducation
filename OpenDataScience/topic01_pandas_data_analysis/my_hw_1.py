#%%

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import os
plt.rcParams['figure.figsize'] = (10, 8)

#PATH = os.path.dirname(os.path.realpath(__file__))  + "/../data/"

data = pd.read_csv('/Users/lev/PycharmProjects/MachineLearningEducation/OpenDataScience/topic01_pandas_data_analysis/../data/adult.data.csv')
#data = pd.read_csv(PATH + 'adult.data.csv')

#1. Сколько мужчин и женщин (признак sex) представлено в этом наборе данных?
data.head(100)
man = pd.Series(data['sex']).value_counts()

#2. Каков средний возраст (признак age) женщин?
mean_age = pd.DataFrame(data[['age', 'sex']]).groupby('sex')['age'].mean()

#3. Какова доля граждан Германии (признак native-country)?
perc_Germ =  data['native-country'][data['native-country'] == 'Germany'].count() * 100 / data.shape[0]

#4. Каковы средние и среднеквадратичные отклонения возраста тех, кто получает более 50K в год (признак salary)?
print(data[data['salary'] == '>50K'].describe())

#5. Каковы средние значения и среднеквадратичные отклонения возраста тех, кто получает менее 50K в год (признак salary)?
print(data[data['salary'] == '<=50K'].describe())

#6. Правда ли, что люди, которые получают больше 50k, имеют как минимум высшее образование? (признак education равен Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters или Doctorate)
print(data[data['salary'] == '>50K']['education'].unique())

#7. Выведите статистику возраста для каждой расы (признак race) и каждого пола. Используйте groupby и describe.
print(data['race'].unique()) #== 'Amer-Indian-Eskimo'].values)

for (race, sex), row in pd.groupby(data, ['race', 'sex']):
    print("Расса: {0},  пол: {1}".format(race, sex))
    print(row['age'].describe())

"""8. Среди кого больше доля зарабатывающих много (>50K): среди женатых или
 холостых мужчин (признак marital-status)?"""
marital = pd.DataFrame(data[(data['salary'] == '>50K') & (data['sex'] == 'Male')]).groupby('marital-status')['marital-status'].agg(pd.value_counts)


"""9. Какое максимальное число часов человек работает в неделю 
(признак hours-per-week)? Сколько людей работают такое количество 
часов и каков среди них процент зарабатывающих много?"""
max_hour = data['hours-per-week'].max()
count_people = data[data['hours-per-week'] == max_hour]['salary'].count()
perc = data[(data['hours-per-week'] == max_hour) & (data['salary'] == '>50K')]['salary'].count() * 100 / count_people

print(max_hour, count_people, perc)

"""10. Посчитайте среднее время работы (hours-per-week) зарабатывающих много и мало 
(salary) для каждой страны (native-country), в частности, для Японии."""

mean_salary_country = pd.DataFrame(data).groupby(['salary', 'native-country'])['hours-per-week'].mean()

print(mean_salary_country)