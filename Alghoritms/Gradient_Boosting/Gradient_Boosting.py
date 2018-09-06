import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from matplotlib import pyplot as plt

class GradientBoosting(BaseEstimator):

    #функция градиента
    def mse_grad(self, y, p):
        return 2 * (p - y.reshape([y.shape[0], 1])) / y.shape[0]

    def __init__(self, n_estimators = 10, learning_rate = 0.01,
                 max_depth = 3, random_state = 17, loss = 'mse', debug = False):
        #число итераций или деревьев
        self.n_estimators = n_estimators
        #максимальная длина дерева
        self.max_depth = max_depth
        #сид генерации псевдослучаных чисел для деревьев
        self.random_state = random_state
        #шаг градиентного спуска (по умолчанию 10^-2)
        self.learning_rate =learning_rate
        #название функции потерь
        self.loss_name = loss
        #инициализатор начального вектора предсказаний (нулевое дерево)
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0], 1])
        self.debug = debug

        if(loss == "mse"):
            self.objective = mean_squared_error
            self.objective_grad = self.mse_grad

        self.trees_ = []
        self.loss_by_iter = []

        if (self.debug):
            self.residuals = []
            self.temp_pred = []

    def fit(self, X, y):
        self.X = X
        self.y = y

        #инициализация первого дерева предсказаний
        b = self.initialization(y)

        predictions = b.copy()

        #цикл по количеству деревьев
        for t in range(self.n_estimators):

            #остатки на текущей итерации
            resid = -self.objective_grad(y, predictions)

            #чтобы разобраться с алгоритмом на игрушечных примерах (debag = True),
            #будем смотреть на остатки (антиградиент) на каждой итерации

            #если хотим следить за остатками на каждой итерации, то записывам остатки в специальный массив
            if self.debug:
                self.residuals.append(resid)

            #обучаем дерево-регрессор на остатки
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state)

            tree.fit(X, resid)

            b = tree.predict(X).reshape([X.shape[0], 1])

            #для отладки может пригодиться прогноз на каждой итерации
            if self.debug:
                self.temp_pred.append(b)

            self.trees_.append(tree)

            predictions += self.learning_rate * b

            self.loss_by_iter.append(self.objective(y, predictions))

        # сохраним прогноз алгоритма на той выборке, на которой он обучался
        self.train_pred = predictions

        return self

    #предсказание
    def predict_proba(self, X):
        #сначала прогноз - это просто вектор средних значений ответов на обучении
        pred = np.ones([X.shape[0], 1]) * np.mean(self.y)

        #добавляем прогнозы деревьев
        for t in range(self.n_estimators):
            pred += self.learning_rate * self.trees_[t].predict(X).reshape([X.shape[0], 1])

        return pred

    def predict(self, X):

        pred_probs = self.predict_proba(X)

        return pred_probs

#Вспомогательные функции
def get_1d_grid(data, eps=0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    return np.arange(x_min, x_max, eps)

def get_2d_grid(data, eps=0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, eps),
                         np.arange(y_min, y_max, eps))

#Регрессия с игрушечным примером
X_regr_toy = np.arange(7).reshape(-1, 1)
y_regr_toy = ((X_regr_toy - 3) ** 2).astype('float64')


xx  =  get_1d_grid(X_regr_toy)
plt.plot(xx, ((xx - 3) ** 2).astype('float64'), color='gray', linestyle='--')
plt.scatter(X_regr_toy, y_regr_toy)

plt.show()

boost_regr_mse = GradientBoosting(n_estimators=200, loss='mse', max_depth=3,
                                  learning_rate=0.1, debug=True)

boost_regr_mse.fit(X_regr_toy, y_regr_toy)

xx = get_1d_grid(X_regr_toy)
plt.plot(xx, ((xx - 3) ** 2), color='gray', linestyle='--')
plt.scatter(X_regr_toy, y_regr_toy)
plt.plot(xx, boost_regr_mse.predict(xx.reshape([xx.shape[0], 1])), color='red')

plt.show()

print(11)

