import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb
import os
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold, train_test_split

PATH = os.path.dirname(os.path.realpath(__file__))

print(os.listdir(PATH))

data = pd.read_csv("prepearing_data_train.csv")
y = data['target']
del data['target']

test = pd.read_csv("prepearing_data_test.csv")

print("Размер обучающей выборки ", data.shape)
print("Размер тестовой выборки ", test.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.3, random_state=17)

def score(params):
    from sklearn.metrics import log_loss
    print("Training with params:")
    print(params)
    params['max_depth'] = int(params['max_depth'])
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    num = params['num_round']
    model = xgb.train(params, dtrain, num)
    predictions = model.predict(dvalid)
    #predictions = predictions.reshape((X_test.shape[0], 2))
    score = log_loss(y_test, predictions)
    print("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}

def optimize(trials):
    space = {
        'num_round': 100,
        'learning_rate': 0.1,
        'max_depth': hp.quniform('max_depth',3, 14, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.01),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 1, 0.05),
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'nthread': 4,
        'silent': 1
    }
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=10)
    return best

trails = Trials()
best_params = optimize(trails)

print(best_params)

"""Training with params:
{'colsample_bytree': 0.9, 'eval_metric': 'error', 'gamma': 0.59, 'learning_rate': 0.1, 'max_depth': 7.0, 'min_child_weight': 1.0, 'nthread': 4, 'num_round': 100, 'objective': 'binary:logistic', 'silent': 1, 'subsample': 0.59}
	Score 0.24333039776697662"""