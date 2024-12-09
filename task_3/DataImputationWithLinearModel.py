import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

def plot_y_yhat(y_val,y_pred, plot_title = "plot"):
    labels = 'SurvivalTime'
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val),MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10,10))
    x0 = np.min(y_val)
    x1 = np.max(y_val)
    plt.scatter(y_val, y_pred)
    plt.xlabel('True ' + labels)
    plt.ylabel('Predicted ' + labels)
    plt.plot([x0, x1], [x0, x1], color='red')
    plt.axis('square')
    plt.show()
    
def ImputeData(df_raw):
    df_raw.drop(columns=df_raw.columns[0], axis=1, inplace=True)
    #df = df_raw
    df = df_raw.dropna(subset=['SurvivalTime'])
    # Univariate imputation
    #imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    #imp.fit(df)
    
    # Multivariate impuatation
    imp = IterativeImputer(missing_values=np.nan, 
                           initial_strategy='most_frequent',
                           imputation_order='ascending',
                           #estimator=BayesianRidge(),
                           estimator=RandomForestRegressor(n_estimators=10, random_state=0),
                           random_state=0,
                           #n_nearest_features=20,
                            max_iter=10,
                           sample_posterior=False)
    imp.fit(df)
    columns = ['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'SurvivalTime', 'Censored']
    df_imp = pd.DataFrame(imp.transform(df), columns=columns)

    # define X and Y
    X = df_imp[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'Censored']]
    y = df_imp[['SurvivalTime']]

    # split the data
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
    
    return X_train, y_train, X_val, X_test, y_val, y_test    


def linearmodel_imp(X_train, y_train, X_val, X_test, y_val, y_test):
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

    plot_y_yhat(y_val, y_hat_val)

    return cMSE_val, cMSE_test

df = pd.read_csv("train_data.csv")
X_train, y_train, X_val, X_test, y_val, y_test = ImputeData(df)

#dftest = pd.read_csv("test_data.csv")
#dftest = pd.DataFrame(dftest)
#dftest_clean = dftest.dropna(axis=0, how='any') # drop missing columns

#X_dftest_test = dftest_clean[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']]

cMSE_val, cMSE_test = linearmodel_imp(X_train, y_train, X_val, X_test, y_val, y_test)
print('cMSE_val:', cMSE_val)
print('cMSE_test:', cMSE_test)
