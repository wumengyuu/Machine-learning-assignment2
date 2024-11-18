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

def linearmodel(X_train, y_train, X_val, X_test, y_val, y_test):

    c_train = X_train[['Censored']].values
    X_train = X_train.drop(columns=['Censored'])
    c_val = X_val[['Censored']].values
    X_val = X_val.drop(columns=['Censored'])
    c_test = X_test[['Censored']].values
    X_test = X_test.drop(columns=['Censored'])
    
    pipe = Pipeline([('std', StandardScaler()),('estimator', LinearRegression())])
    pipe.fit(X_train, y_train)
 
    y_hat_val = pipe.predict(X_val)
    cMSE_val = error_metric(y_val, y_hat_val, c_val) # cross validation

    y_hat_test = pipe.predict(X_test)
    cMSE_test = error_metric(y_test, y_hat_test, c_test) # cross validation
    
    # plot y, y_hat_val
    plot_y_yhat(y_val,y_hat_val)

    return cMSE_val, cMSE_test


df = pd.read_csv("train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
cMSE_val, cMSE_test = linearmodel(X_train, y_train, X_val, X_test, y_val, y_test)
print('cMSE_val:', cMSE_val)
print('cMSE_test:', cMSE_test)

