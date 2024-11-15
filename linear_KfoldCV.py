import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from plot_y_yhat import plot_y_yhat

def data_cleaning(data):
    df = pd.DataFrame(data)
    df_clean = df.dropna(subset=['SurvivalTime']) # drop row if SurvivalTime is missing
    df_clean = df_clean.dropna(axis=1, how='any') # drop missing columns
    return df_clean

def get_Xy(df_clean):
    X = df_clean[['Age', 'Gender', 'Stage', 'TreatmentType', 'Censored']]
    y = df_clean[['SurvivalTime']]
    print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))
    return X, y

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def linearmodel(X_train, y_train, X_test, y_test):

    X_train = X_train.drop(columns=['Censored'])
    c_test = X_test[['Censored']].values
    X_test = X_test.drop(columns=['Censored'])
    
    pipe = Pipeline([('std', StandardScaler()),('estimator', LinearRegression())])
    pipe.fit(X_train, y_train)

    y_hat_test = pipe.predict(X_test)
    cMSE_test = error_metric(y_test, y_hat_test, c_test) 
    
    return cMSE_test

# K-fold cross validation
def Kfold_CV(K, X, y):
    kf =KFold(n_splits=K, shuffle=True, random_state=42)
    cnt = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        cMSE_test = linearmodel(X_train, y_train, X_test, y_test)
        print(f'cMSE: {cMSE_test}')
        cnt += 1
    return

data = pd.read_csv("train_data.csv")
df_clean = data_cleaning(data)
X, y = get_Xy(df_clean)
K = 5
Kfold_CV(K, X, y)

# Gradient Descent on CMSE, maybe add Ridge regularization
# CMSE(y, c)+Ridge regularization