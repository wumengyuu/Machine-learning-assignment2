import numpy as np
import pandas as pd
import math
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from Datapreparation import dataprepare

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


def linearmodel(X_train, y_train):
    c = X_train[['Censored']].values
    X_train = X_train.drop(columns=['Censored'])

    pipe = Pipeline([('std', StandardScaler()),('estimator', LinearRegression())])
    pipe.fit(X_train, y_train)
    y_hat = pipe.predict(X_train)
    cMSE = error_metric(y_train, y_hat, c)
    return cMSE


df = pd.read_csv("train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
# print(X_train.shape, y_train.shape)
cMSE = linearmodel(X_train, y_train)
print(cMSE)