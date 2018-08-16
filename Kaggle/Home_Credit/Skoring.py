import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xbg

from Home_Credit.Extracting import Extracting










X, Y = Extracting(path='D:/Machine_Learning_Competition/Home Credit/application_train.csv', test=True)

print(1)