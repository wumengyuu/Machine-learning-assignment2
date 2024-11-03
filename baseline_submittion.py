import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from Datapreparation import dataprepare
from plot_y_yhat import plot_y_yhat

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def linearmodel(X_train, y_train, X_dftest_test ):

    X_train = X_train.drop(columns=['Censored'])
    pipe = Pipeline([('std', StandardScaler()),('estimator', LinearRegression())])
    pipe.fit(X_train, y_train)
 
    yhat_dftest_test = pipe.predict(X_dftest_test)
    
    # save yhat_dftest_test as csv
    ids = np.arange(0, yhat_dftest_test.shape[0]).reshape(-1, 1)
    y_pred = np.hstack((ids, yhat_dftest_test))

    y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
    y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
    print(y_pred_df[:3])

    y_pred_df.to_csv("baseline-submission-01.csv", index=False)
    return 


df = pd.read_csv("train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)

dftest = pd.read_csv("test_data.csv")
dftest = pd.DataFrame(dftest)
dftest = dftest.drop(columns=['GeneticRisk','ComorbidityIndex', 'TreatmentResponse'])
X_dftest_test = dftest[['Age', 'Gender', 'Stage', 'TreatmentType']]

linearmodel(X_train, y_train, X_dftest_test)

