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
           ['kiwitobes', 'UK', 'no', 19, ' '],
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
        split_funtion = lambda row: row[column] == value

    # Разбить множество строк на две части и вернуть их
    set1 = [row for row in rows if split_funtion(row)]
    set2 = [row for row in rows if not split_funtion(row)]

    return(set1, set2)

#divideset(my_data,0,'slashdot')

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

def buildtree (rows, list_index, scoref=entropy):


    if(len(rows)==0): decisionnode()
    current_score=scoref(rows)

    #Инициализировать переменные для выбора наилучшего критерия

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    #colunm_count = len(rows[0]) - 1

    for col in list_index:
        # создаем список различных значений в этом столбце
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1

    #Пробуем разбить множество строк по каждому значению
    # из этого столбца

        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            #информационный выигрыш
            p = float(len(set1))/len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)

            if(gain > best_gain and len(set1) > 0 and len(set2) > 0):
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # создаем подветви
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0], list_index)
        falseBranch = buildtree(best_sets[1], list_index)
        return decisionnode(col= best_criteria[0], value= best_criteria[1], tb= trueBranch, fb= falseBranch)

    else:
        results = uniquecounts(rows)
        return decisionnode(results= results)

#предсказываем значение для неизвестной строки
def predict(tree, row):

    #значение предиката сравнения в строке
    key_point = row[tree.col]

    l_function = None

    #если значение предиката целочисленное, то лямбда-функция будет работать в числами
    if isinstance(key_point, int) or isinstance(key_point, float):
        l_function = lambda row: row[tree.col] > tree.value
    #иначе лямбда функция работает со строками
    else:
        l_function = lambda row: row[tree.col] == tree.value

    #если в экземпляре класса дерева значение атрибута result пустое, значит это промежуточная вершина
    if (tree.result == None):

        #сравниваем значение предиката с порогом value через вызов лямбда-функции
        if(l_function(row)):
            return(predict(tree.tb, row))

        else:
            return(predict(tree.fb,row))
    # если в экземпляре класса дерева значение атрибута result не пустое, значит мы дошли до вершины
    else:
        res = tree.result
        return res


def prune(tree, mingain):
    #если ветки не листовые, то вызвать рекрсию
    if tree.tb.result == None:
        prune(tree.tb, mingain)

    if tree.fb.result == None:
        prune(tree.fb, mingain)

    #Если обе подветки не заканчиваются листьями, смотрим, нужно ли их объединить
    if tree.tb.result != None and tree.fb.result != None :
        #строим объединенный набор данных
        tb, fb = [],[]

        for v, c in tree.tb.relult.items():
            tb += [[v]] * c

        for v, c in tree.fb.result.items():
            fb += [[v]] * c

        #вычисляем, насколько уменьшилась энтропия

    delta = entropy(tb+fb)-(entropy(tb) + entropy(fb)/2)

    if delta<mingain:
        #объединяем ветви
        tree.tb, tree.fb = None, None
        tree.result=uniquecounts(tb+fb)


list_index = [i for i in np.arange(len(my_data[0]) - 1)]
tree = buildtree(my_data, list_index, scoref=entropy)
#prune(tree, 0.1)

row = ['google', 'USA', 'no', 24]
row1 = ['digg', 'USA', 'no', 27]

pr = predict(tree, row)
pr1 = predict(tree, row1)

for i in pr1.keys():
    print(i)

print(pr)

print(entropy(my_data))
