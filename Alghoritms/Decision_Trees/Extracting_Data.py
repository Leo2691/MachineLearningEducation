import pandas as pd
import numpy as np


def extracting_data(path, n):
    data_with_NAN = pd.read_csv(path)

    data = data_with_NAN.dropna()
    data = data.reset_index()

    Y = np.array(pd.DataFrame.as_matrix(data['TARGET']))  # .values.tolist()
    Y = Y.reshape(len(Y), 1)

    X = np.array(pd.DataFrame.as_matrix(data.drop('TARGET', 1)))  # .values.tolist()

    X = np.append(X, Y, 1)

    return X[0:n]

