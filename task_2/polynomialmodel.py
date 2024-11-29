import math

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from Datapreparation import dataprepare
from error_metric import error_metric
from plot_y_yhat import plot_y_yhat, plot_error


def validate_poly_regression(X_train, y_train, X_val, y_val,
                             regressor=None, degrees=range(1, 50),
                             max_features=None):
    print(f'Validating polynomial regression with degrees: {degrees}')
    c_train = X_train[['Censored']].values
    X_train = X_train.drop(columns=['Censored'])
    c_val = X_val[['Censored']].values
    X_val = X_val.drop(columns=['Censored'])
    all_errors: dict = {}
    best_model = None
    best_error = np.inf
    if regressor is None:
        alphas = [0.01, 0.1, 1, 10]
        regressor = RidgeCV(alphas=alphas)
    for deg in degrees:
        polyreg = make_pipeline(PolynomialFeatures(deg),
                                StandardScaler(),
                                regressor)
        polyreg.fit(X_train, y_train)

        y_pred_val = polyreg.predict(X_val)

        # compare cMSE
        cmse = error_metric(y_val, y_pred_val, c_val)[0]


        all_errors[deg] = cmse

        # Check if this is the best model so far (based on validation RMSE)
        if cmse < best_error:
            best_error = cmse
            best_model = polyreg

    plot_error(all_errors, plot_title="Validation cMSE vs. Degree",
                     xlabel="Degree", ylabel="cMSE")
    return best_model, best_error


# df = pd.read_csv("../data/train_data.csv")
# X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
# top_model, top_error = validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None)
# print(top_model.named_steps['ridgecv'].alpha_)
# print(f'\nBest model: {top_model}\nBest error: {top_error}')
