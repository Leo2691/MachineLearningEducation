import pandas as pd
import numpy as np
import math
import pylab
from matplotlib import mlab

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




X = [i for i in np.arange(-20, 20, 0.1)]
Y = [i for i in np.sin(X)]
X = np.array(X)
Y = np.array(Y)

noise = np.random.normal(Y, scale=0.3)

Y = Y + noise

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X=X, y=Y)

Y_pred = reg.predict(X)

score = mean_squared_error(Y, Y_pred)

# !!! Нарисуем одномерный график
pylab.plot(X, Y, X, Y_pred)

# !!! Покажем окно с нарисованным графиком
pylab.show()

print(score)




