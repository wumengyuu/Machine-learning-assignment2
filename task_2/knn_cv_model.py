import math

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from plot_y_yhat import plot_error, plot_all_error_stats


def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


def data_cleaning(data):
    df = pd.DataFrame(data)
    df_clean = df.dropna(subset=['SurvivalTime']) # drop row if SurvivalTime is missing
    df_clean = df_clean.dropna(axis=1, how='any') # drop missing columns
    return df_clean


def get_Xy(df_clean):
    X = df_clean[['Age', 'Gender', 'Stage', 'TreatmentType', 'Censored']]
    y = df_clean[['SurvivalTime']]
    return X, y

def validate_knn_regression(X, y, kcv=8, k=7):
    all_errors: list = []
    best_model = None
    best_error = np.inf
    for kcv_val in range(2, kcv):
        knn = KNeighborsRegressor(n_neighbors=k)
        kf = KFold(n_splits=kcv_val, shuffle=True)
        cmse_list = []

        for train_index, test_index in kf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            c_test = X_test[['Censored']].values
            X_train = X_train.drop(columns=['Censored'])
            X_test = X_test.drop(columns=['Censored'])

            knn.fit(X_train, y_train)
            y_pred_val = knn.predict(X_test)
            cMSE_test = error_metric(y_test, y_pred_val, c_test)
            # rmse = math.sqrt(np.mean((y_val - y_pred_val) ** 2))
            cmse_list.append(cMSE_test)

        avg_cmse = np.mean(cmse_list)
        all_errors.append(avg_cmse)

        if avg_cmse < best_error:
            best_error = avg_cmse
            best_model = knn

    #plot_error(all_errors, plot_title="Validation RMSE vs. k",
     #          xlabel="k", ylabel="RMSE")
    return best_model, best_error, all_errors

def get_best_knn_model_with_cv():
    df = pd.read_csv("../data/train_data.csv")
    # X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
    df_clean = data_cleaning(df)
    X, y = get_Xy(df_clean)
    top_model = None
    top_error = np.inf
    top_k = None

    complete_error = {}
    for k in range(1, 20):
        model, error, all_errors = validate_knn_regression(X, y, k=k)
        complete_error[k] = all_errors
        #print(f'For K={k}  Best error: {error}')
        if error < top_error:
            top_error = error
            top_model = model
            top_k = k

    print(f'\n\nBest model: {top_model}\nBest error: {top_error}\nBest k: {top_k}')
    plot_all_error_stats(complete_error, plot_title="KNearestNeighbors with KFold Cross Validation",
                         xlabel="Number of neighbors", ylabel="cMSE")
    return top_model


def get_submission():
    model = get_best_knn_model_with_cv()

    dftest = pd.read_csv("../data/test_data.csv")
    dftest = pd.DataFrame(dftest)
    #dftest = dftest.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'])
    X_dftest_test = dftest[['Age', 'Gender', 'Stage', 'TreatmentType']]

    #print(X_dftest_test.shape)
    yhat_dftest_test = model.predict(X_dftest_test)

    # save yhat_dftest_test as csv
    ids = np.arange(0, yhat_dftest_test.shape[0]).reshape(-1, 1)
    y_pred = np.hstack((ids, yhat_dftest_test))

    y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
    y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
    #print(y_pred_df[:3])

    y_pred_df.to_csv("submissions/Nonlinear-submission-11.csv", index=False)


get_submission()
