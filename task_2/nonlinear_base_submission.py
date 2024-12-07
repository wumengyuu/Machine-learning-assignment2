import numpy as np
import pandas as pd

from Datapreparation import dataprepare
from polynomialmodel import validate_poly_regression
from task_2.knn_submission import validate_knn_regression

df = pd.read_csv("../data/train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)

#X_train = X_train.drop(columns=['Censored'])
#X_val = X_val.drop(columns=['Censored'])
#model, error = validate_poly_regression(X_train, y_train, X_val, y_val, degrees=range(35, 36))
model, error = validate_knn_regression(X_train, y_train, X_val, y_val, k=range(7, 8))

dftest = pd.read_csv("../data/test_data.csv")
dftest = pd.DataFrame(dftest)
dftest = dftest.drop(columns=['GeneticRisk','ComorbidityIndex', 'TreatmentResponse'])
X_dftest_test = dftest[['Age', 'Gender', 'Stage', 'TreatmentType']]

yhat_dftest_test = model.predict(X_dftest_test)

# save yhat_dftest_test as csv
ids = np.arange(0, yhat_dftest_test.shape[0]).reshape(-1, 1)
y_pred = np.hstack((ids, yhat_dftest_test))

y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
y_pred_df['id'] = y_pred_df['id'].astype(np.int32)

print("Created submission")
#y_pred_df.to_csv("submissions/Nonlinear-submission-12.csv", index=False)
