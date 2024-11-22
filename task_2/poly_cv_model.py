import math

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from plot_y_yhat import plot_error


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


def validate_poly_regression(X, y, k=5, regressor=None, degrees=range(1, 15),
                             max_features=None):
    all_errors: dict = {}
    best_model = None
    best_error = np.inf
    if regressor is None:
        alphas = [0.1, 1, 10, 100]
        regressor = RidgeCV(alphas=alphas)
        #regressor = LinearRegression()
    for deg in degrees:
        polyreg = make_pipeline(PolynomialFeatures(deg),
                                StandardScaler(),
                                regressor)
        #print(f"{polyreg}\n{best_model}")

        kf = KFold(n_splits=k, shuffle=True)
        cmse_list = []

        for train_index, test_index in kf.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            c_test = X_test[['Censored']].values
            X_train = X_train.drop(columns=['Censored'])
            X_test = X_test.drop(columns=['Censored'])

            polyreg.fit(X_train, y_train)
            y_pred_val = polyreg.predict(X_test)
            cMSE_test = error_metric(y_test, y_pred_val, c_test)
            cmse_list.append(cMSE_test)

        # compare RMSE
        avg_cmse = np.mean(cmse_list)
        if avg_cmse > 100:
            break

        all_errors[deg] = avg_cmse

        #print(f'Degree {deg}: Avg cMSE = {avg_cmse:.4f}')

        if avg_cmse < best_error:
            best_error = avg_cmse
            best_model = polyreg

    plot_error(all_errors, plot_title="Polynomial Regression with KFold Cross Validation for k="+str(k),
                     xlabel="Degree", ylabel="cMSE")
    return best_model, best_error


def get_best_poly_model_with_cv():
    df = pd.read_csv("../data/train_data.csv")
    # X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
    df_clean = data_cleaning(df)
    X, y = get_Xy(df_clean)
    top_model = None
    top_error = np.inf
    top_k = None
    for k in range(2, 10):
        model, error = validate_poly_regression(X, y, k, regressor=None)
        #print(f'For K={k}  Best error: {error}')
        if error < top_error:
            top_error = error
            top_model = model
            top_k = k

    best_degree = top_model.named_steps['polynomialfeatures'].degree
    res, err = validate_poly_regression(X, y, top_k, regressor=None, degrees=range(best_degree, best_degree+1))
    print(f'\n\nBest model: {top_model}\nBest error: {err}\nBest K: {top_k}')
    return res
    #print("get best model: ", top_model.n_features_in_)
    #return top_model


def get_submission():
    modell = get_best_poly_model_with_cv()

    dftest = pd.read_csv("../data/test_data.csv")
    dftest = pd.DataFrame(dftest)
    #dftest = dftest.drop(columns=['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse'])
    X_dftest_test = dftest[['Age', 'Gender', 'Stage', 'TreatmentType']]

    yhat_dftest_test = modell.predict(X_dftest_test)

    # save yhat_dftest_test as csv
    ids = np.arange(0, yhat_dftest_test.shape[0]).reshape(-1, 1)
    y_pred = np.hstack((ids, yhat_dftest_test))

    y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
    y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
    #print(y_pred_df[:3])

    y_pred_df.to_csv("submissions/Nonlinear-submission-06.csv", index=False)


get_submission()
