import numpy as np
import sklearn.model_selection as sk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import datasets



def SVM_Text():
    newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

    X = newsgroups.data
    y = newsgroups.target

    vectorizer = TfidfVectorizer()
    X_tr = vectorizer.fit_transform(X)

    # словарь массивов, где массив доступен по ключу 'C'
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    #print(grid['C'])

    #генератор разбиений выборки
    cv = sk.KFold(n_splits=5, shuffle=True, random_state=241)

    #классификатор на основе метода опорных векторов
    clf = SVC(kernel='linear', random_state=241)

    #поиск наилучших значанией коэффциентов для данного классфикатора
    gs = sk.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

    #обучили классфикатор
    gs.fit(X_tr, y)

    #узнали наилучшее значение параметра С
    best_C = gs.best_score_

    #задали классификатор заново с наилучшим С
    clf = SVC(C=best_C, kernel='linear', random_state=241)

    #обучили классфикатор
    clf.fit(X_tr, y)




    print(1)




SVM_Text()