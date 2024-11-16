import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from Datapreparation import dataprepare
from plot_y_yhat import plot_y_yhat, plot_error


def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 15)):
    all_errors: dict = {}
    best_model = None
    best_error = np.inf
    for k_val in k:
        knn = KNeighborsRegressor(n_neighbors=k_val)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        # plot_y_yhat(y_val, y_pred, str(k_val))
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        print(f'K: {k_val} got RMSE of value: {rmse}')
        all_errors[k_val] = rmse
        if rmse < best_error:
            best_error = rmse
            best_model = knn

    plot_error(all_errors, plot_title="Validation RMSE vs. k",
               xlabel="k", ylabel="RMSE")
    return best_model, best_error


#df = pd.read_csv("train_data.csv")
#X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
#k_range = range(1, 60)
#top_model, top_error = validate_knn_regression(X_train, y_train, X_val, y_val, k=k_range)
#print(f'\nBest model: {top_model}\nBest error: {top_error}')
