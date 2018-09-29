"""Добавляем данные из вспомогательных таблиц----------------------------------------------------------------------------------"""

import gc
import os

PATH = "../input/"

print(os.listdir(PATH))

gc.collect()

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold

from lightgbm import LGBMClassifier

data = pd.read_csv(PATH + 'application_train.csv')
test = pd.read_csv(PATH + 'application_test.csv')
prev = pd.read_csv(PATH + 'previous_application.csv')
buro = pd.read_csv(PATH + 'bureau.csv')
buro_balance = pd.read_csv(PATH + 'bureau_balance.csv')
credit_card  = pd.read_csv(PATH + 'credit_card_balance.csv')
POS_CASH  = pd.read_csv(PATH + 'POS_CASH_balance.csv')
payments = pd.read_csv(PATH + 'installments_payments.csv')



#Отделяем метки
y = data['TARGET']
del data['TARGET']

#кодирем категориальные признаки

categorial_features = [col for col in data.columns if data[col].dtype == 'object']

one_hot_df = pd.concat([data, test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorial_features)

data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]

print('Формат тренировочной выборки', data.shape)
print('Формат тестовой выборки', test.shape)

#работа с признаками
buro_group_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_group_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_group_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

#сложный срез
buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize=False)
#раскладываем его в DataFrame c помощью unstack
buro_counts_unstacked = buro_counts.unstack('STATUS')

buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C', 'STATUS_X']
buro_counts_unstacked['MONTHS_COUNT'] = buro_group_size
buro_counts_unstacked['MONTHS_MIN'] = buro_group_min
buro_counts_unstacked['MONTHS_MAX'] = buro_group_max

buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')

print(buro.head(20))

"""Довольно много данных, которые, в общем-то, можно попробовать просто 
закодировать One-Hot-Encoding'ом, сгруппировать по SK_ID_CURR, усреднить 
и, таки образом, подготовить для объединения с основной таблице"""

buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
buro = pd.get_dummies(buro, columns=buro_cat_features)

avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

del avg_buro['SK_ID_BUREAU']

del buro
gc.collect()

"""Данные по предыдущим заявкам
Точно также закодируем категориальные признаки, усредним и объединим по текущему ID."""

prev_cat_future = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_future)
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

del prev
gc.collect()

"""Баланс по кредитной карте
Закодируем категориальные признаки и подготовим таблицу для объединения"""

le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].max()
#POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
#POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

"""Данные по картам. Аналогичная работа"""
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
#nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
#credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
#credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

"""Данные по платежам
Создадим три таблицы — со средними, минимальными и максимальными значениями из этой таблицы."""

avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()
del avg_payments['SK_ID_PREV']

del payments
gc.collect()

"""Объединение таблиц. Самый важный этап"""

data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

del avg_prev, avg_buro, POS_CASH, credit_card, avg_payments, avg_payments2, avg_payments3
gc.collect()

print ('Формат тренировочной выборки', data.shape)
print ('Формат тестовой выборки', test.shape)
print ('Формат целевого столбца', y.shape)

"""И, собственно, ударим по этой выросшей в два раза таблице градиентным бустингом!------------------------------"""

"""from lightgbm import LGBMClassifier

clf2 = LGBMClassifier()
clf2.fit(data, y)

predictions = clf2.predict_proba(test)[:, 1]

# Датафрейм для загрузки
submission = test[['SK_ID_CURR']]
submission['TARGET'] = predictions

# Сохранение датафрейма
submission.to_csv('lightgbm_full.csv', index = False)"""

"""ОК, напоследок попробуем более сложную методику с разделением на фолды,-----------------------------------------
кросс-валидацией и выбором лучшей итерации."""

"""folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()

feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]

clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=34,
        colsample_bytree=0.9,
        subsample=0.8,
        max_depth=8,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=375,
        silent=-1,
        verbose=-1,
        )

clf.fit(trn_x, trn_y,
            eval_set= [(trn_x, trn_y), (val_x, val_y)],
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )

oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = feats
fold_importance_df["importance"] = clf.feature_importances_
fold_importance_df["fold"] = n_fold + 1
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
del clf, trn_x, trn_y, val_x, val_y
gc.collect()

print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('submission_cross.csv', index=False)"""

"""Пробуем смесь lightgbm и xgboost --------------------------------------------------------"""

from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb


class djLGB(BaseEstimator, ClassifierMixin):
    """смесь lgb и xgb"""

    def __init__(self, seed=0, nest_lgb=1.0, nest_xgb=1.0, cbt=0.5, ss=0.5, alpha=0.5):
        """
        Инициализация
        seed - инициализация генератора псевдослучайных чисел
        nest_lgb, nest_xgb - сколько деревьев использовать (множитель)
        cbt, ss - процент признаков и объектов для сэмплирования
        alpha - коэффициент доверия XGB
        """
        print('LGB + XGB')
        self.models = [lgb.LGBMClassifier(num_leaves=2, learning_rate=0.07, n_estimators=int(1400 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=0 + seed),
                       lgb.LGBMClassifier(num_leaves=3, learning_rate=0.07, n_estimators=int(800 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=1 + seed),
                       lgb.LGBMClassifier(num_leaves=4, learning_rate=0.07, n_estimators=int(800 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=2 + seed),
                       lgb.LGBMClassifier(num_leaves=5, learning_rate=0.07, n_estimators=int(600 * nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=3 + seed, ),
                       xgb.XGBClassifier(max_depth=1,
                                         learning_rate=0.1,
                                         n_estimators=int(800 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=0 + seed),
                       xgb.XGBClassifier(max_depth=2,
                                         learning_rate=0.1,
                                         n_estimators=int(400 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=1 + seed),
                       xgb.XGBClassifier(max_depth=3,
                                         learning_rate=0.1,
                                         n_estimators=int(200 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=2 + seed),
                       xgb.XGBClassifier(max_depth=4,
                                         learning_rate=0.1,
                                         n_estimators=int(200 * nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=3 + seed)
                       ]
        self.weights = [(1 - alpha) * 1, (1 - alpha) * 1, (1 - alpha) * 1, (1 - alpha) * 0.5, alpha * 0.5, alpha * 1,
                        alpha * 1.5, alpha * 0.5]

    def fit(self, X, y=None):
        """
        обучение
        """
        for t, clf in enumerate(self.models):
            # print ('train', t)
            clf.fit(X, y)
        return self

    def predict(self, X):
        """
        определение вероятности
        """
        suma = 0.0
        for t, clf in enumerate(self.models):
            a = clf.predict_proba(X)[:, 1]
            suma += (self.weights[t] * a)
        return (suma / sum(self.weights))

    def predict_proba(self, X):
        """
        определение вероятности
        """
        return (self.predict(X))

#for t in range(N):
t = 1


#data['target'] = y

#pd.DataFrame(data).to_csv("prepearing_data_train.csv", index = False)
#pd.DataFrame(test).to_csv("prepearing_data_test.csv", index = False)

clf = djLGB(seed=2000 + 10*t, nest_lgb=1.3, nest_xgb=1.3)
clf.fit(data, y)




pred = clf.predict(test)

# Датафрейм для загрузки
#submission = test[['SK_ID_CURR']]
#submission['TARGET'] = pred

# Сохранение датафрейма
#submission.to_csv('mix_lightgbm_xgboost_full.csv', index = False)


