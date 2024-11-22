import math

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from Datapreparation import dataprepare
from plot_y_yhat import plot_y_yhat, plot_error


def validate_poly_regression(X_train, y_train, X_val, y_val,
                             regressor=None, degrees=range(1, 15),
                             max_features=None):
    print(f'Validating polynomial regression with degrees: {degrees}')
    all_errors: dict = {}
    best_model = None
    best_error = np.inf
    if regressor is None:
        alphas = [0.0001, 0.001, 0.01, 0.1]
        regressor = RidgeCV(alphas=alphas)
    for deg in degrees:
        polyreg = make_pipeline(PolynomialFeatures(deg),
                                StandardScaler(),
                                regressor)
        polyreg.fit(X_train, y_train)

        # predict on training and validation dataset
        y_pred_val = polyreg.predict(X_val)
        y_pred_train = polyreg.predict(X_train)
        # plot_y_yhat(y_val, y_pred_val)

        # compare RMSE
        rmse_val = math.sqrt(np.square(np.subtract(y_val, y_pred_val)).mean().iloc[0])
        rmse_train = math.sqrt(np.square(np.subtract(y_train, y_pred_train)).mean().iloc[0])

        # Print the number of polynomial features and RMSE for both train and validation
        # print(f'Degree {deg}: Created {polyreg.named_steps["polynomialfeatures"].n_output_features_} features.')
        # print(f'Degree {deg}: Train RMSE = {rmse_train:.4f}, Validation RMSE = {rmse_val:.4f}')

        all_errors[deg] = rmse_val

        # Check if this is the best model so far (based on validation RMSE)
        if rmse_val < best_error:
            best_error = rmse_val
            best_model = polyreg

    # plot_error(all_errors, plot_title="Validation RMSE vs. Degree",
      #               xlabel="Degree", ylabel="RMSE")
    return best_model, best_error

df = pd.read_csv("../data/train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
top_model, top_error = validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None)
print(f'\nBest model: {top_model}\nBest error: {top_error}')
