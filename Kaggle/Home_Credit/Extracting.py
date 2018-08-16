import pandas as pd

def Extracting(path, test = True):

    df = pd.read_csv(path)

    if(test):
        Y = df['TARGET']
        del df['TARGET']
        X = df
        return X, Y

    else:
        return df

#Extracting(path='D:/Machine_Learning_Competition/Home Credit/application_train.csv', test=True)