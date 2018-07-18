import pandas as pd
import numpy as np

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

class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.result = results
        self.tb = tb
        self.fb = fb

#Разбиение множества по указанному столбцу. Может обрабатывать как числовые
#так и дискретные значения
def divideset(rows, column, value):
    #Создать функцию, которая сообщит, относится ли строка к первой группе
    #(true) или ко второй (false)
    split_funtion = None
    if isinstance(value, int) or isinstance(value, float):
        split_funtion = lambda row: row[column] > value
    else:
        split_funtion = lambda row: rows[column] == value

    # Разбить множество строк на две части и вернуть их
    set1 = [row for row in rows if split_funtion(row)]
    set2 = [row for row in rows if not split_funtion(row)]

    return(set1, set2)

#Вычислиь счетчики вхождения каждого результата в множество строк
#(результат - это последний столбец в каждой строке)
def uniquecounts(rows):
    results = {}
    for row in rows:
        #результат находится в последнем столбце
        r = row[len(row)-1]
        #если r входит в словарь
        if r not in results: results[r]=0
        results[r]+=1
    return results

#Коэффициент Джини
#Вероятность того, что случайный образец пренадлежит не к той категории
def giniimpurity(rows):
    total=len(rows)
    counts=uniquecounts(rows)
    imp=0

    for k1 in counts:
        p1=float(counts[k1])/total
        for k2 in counts:
            if k1==k2:continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2

    return imp
#divideset(my_data,2,'yes')

#Энтропия вычисляется как сумма p(x)log(p(x)) по всем различным результатам
def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    #теперь вычислим энтропию
    ent = 0.0
    for r in results.keys():
        p = float(results[r] / len(rows))
        ent = ent - p * log2(p)
    return ent

def buildtree (rows, scoref=entropy):


    if(len(rows)==0): decisionnode()
    current_score=scoref(rows)

    #Инициализировать переменные для выбора наилучшего критерия

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    colunm_count = len(rows[0]) - 1

    for col in range(0, colunm_count):
        # создаем список различных значений в этом столбце
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1




buildtree(my_data, scoref=entropy)

print(giniimpurity(my_data))
print(entropy(my_data))
