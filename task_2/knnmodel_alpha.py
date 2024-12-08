import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from Datapreparation import dataprepare
from error_metric import error_metric
from plot_y_yhat import plot_y_yhat, plot_error


def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 15)):
    c_train = X_train[['Censored']].values
    X_train = X_train.drop(columns=['Censored'])
    c_val = X_val[['Censored']].values
    X_val = X_val.drop(columns=['Censored'])

    all_errors: dict = {}
    best_model = None
    best_error = np.inf
    for k_val in k:
        knn = KNeighborsRegressor(n_neighbors=k_val)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        # plot_y_yhat(y_val, y_pred, str(k_val))
        #rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        cmse = error_metric(y_val, y_pred, c_val)[0]
        print(f'K: {k_val} got cMSE of value: {cmse}')
        all_errors[k_val] = cmse
        if cmse < best_error:
            best_error = cmse
            best_model = knn

    plot_error(all_errors, plot_title="Validation cMSE vs. k",
               xlabel="k", ylabel="cMSE")
    return best_model, best_error


df = pd.read_csv("../data/train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
k_range = range(1, 60)
top_model, top_error = validate_knn_regression(X_train, y_train, X_val, y_val, k=k_range)
print(f'\nBest model: {top_model}\nBest error: {top_error}')
