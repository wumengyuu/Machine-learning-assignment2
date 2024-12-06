import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def dataprepare(df):
    df = pd.DataFrame(df)
    # Drop missing columns
    df = df.drop(df.columns[0], axis=1)     # drop first column
    df = df.dropna(subset=['SurvivalTime']) # drop row if SurvivalTime is missing
    df_clean = df.dropna(axis=1, how='any') # drop missing columns
    return df_clean

def get_Xyc(df):
    # Get X, y, c
    X = df[['Age', 'Gender', 'Stage', 'TreatmentType']]
    y = df[['SurvivalTime']]
    c = df[['Censored']]
    return X, y, c

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def linearmodel(X_train, y_train, X_test, y_test, c):
    c = c.values
    # linear model
    pipe = Pipeline([('std', StandardScaler()),('estimator', LinearRegression())])
    pipe.fit(X_train, y_train)
    y_hat_test = pipe.predict(X_test)
    cMSE_test = error_metric(y_test, y_hat_test, c) 
    return cMSE_test

# K-fold cross validation
def Kfold_CV(K, X, y, c):
    kf =KFold(n_splits=K, shuffle=True, random_state=42)
    cnt = 1
    cmse_list = []
    for train_index, test_index in kf.split(X, y, c):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        c_train, c_test = c.iloc[train_index], c.iloc[test_index]
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        cMSE_test = linearmodel(X_train, y_train, X_test, y_test, c_test)
        print(f'cMSE: {cMSE_test}')
        cnt += 1
        cmse_list.append(cMSE_test)
    return cmse_list

data = pd.read_csv("train_data.csv")
df_clean = dataprepare(data)
X, y, c = get_Xyc(df_clean)

# K-fold cross validation
K = 5
cmse_list = Kfold_CV(K, X, y, c)
plt.plot(range(1, K+1), cmse_list, '-o')
plt.title('K-fold Cross Validation')
plt.xlabel('Fold')
plt.ylabel('CMSE')
plt.show()
