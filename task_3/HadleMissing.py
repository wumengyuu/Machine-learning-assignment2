import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer

def ImputeData(df):
    # Univariate imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    imp.fit(df)
    
    # Multivariate impuatation
    #imp = IterativeImputer(missing_values=np.nan, 
    #                       initial_strategy='most_frequent',
    #                       imputation_order='ascending',
                           #estimator=BayesianRidge(),
    #                       estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    #                       random_state=0,
                           #n_nearest_features=20,
    #                        max_iter=10,
    #                       sample_posterior=False)
    #imp.fit(df)
    columns = ['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'SurvivalTime', 'Censored']
    df_imp = pd.DataFrame(imp.transform(df), columns=columns)

    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'Censored']
    encoder = LabelEncoder()

    for feature in categorical_features:
        df_imp[feature] = encoder.fit_transform(df_imp[feature])
    
    return df_imp

def Impute_test_data(df):
    #df_raw.drop(columns=df_raw.columns[0], axis=1, inplace=True)
    #df = df_raw.dropna(subset=['SurvivalTime'])
    # Univariate imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    imp.fit(df)
    
    # Multivariate impuatation
    #imp = IterativeImputer(missing_values=np.nan, 
    #                       initial_strategy='most_frequent',
    #                       imputation_order='ascending',
                           #estimator=BayesianRidge(),
    #                       estimator=RandomForestRegressor(n_estimators=10, random_state=0),
    #                       random_state=0,
                           #n_nearest_features=20,
    #                        max_iter=10,
    #                       sample_posterior=False)
    #imp.fit(df)
    columns = ['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
    df_imp = pd.DataFrame(imp.transform(df), columns=columns)

    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
    encoder = LabelEncoder()

    for feature in categorical_features:
        df_imp[feature] = encoder.fit_transform(df_imp[feature])
    
    return df_imp

def interval_mae(y_true_lower, y_true_upper, y_pred):
    mae = np.where((y_true_lower <= y_pred) & (y_pred <= y_true_upper),
                   0,
                   np.minimum(np.abs(y_true_lower-y_pred),
                              np.abs(y_true_upper-y_pred))) 
    return mae.mean()

def predict_with_AFT(df_train, df_test):
    df_train['y_lower'] = df_train['SurvivalTime']
    df_train['y_upper'] = np.where(df_train['Censored'], df_train['SurvivalTime'], -1)
    
    stratifying_column = df_train['Censored']
    df_train = df_train.drop(['Censored', 'SurvivalTime'], axis=1)

    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']

    X = df_train[['Age'] + categorical_features]
    y = df_train[['y_lower', 'y_upper']]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=stratifying_column, random_state=32)

    X_test = df_test[['Age'] + categorical_features]

    
    train_pool = Pool(X_train, label=y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, label=y_val, cat_features=categorical_features)
    test_pool = Pool(X_test, cat_features=categorical_features)

    model_logistic = CatBoostRegressor(iterations=500,
                                      loss_function='SurvivalAft:dist=Logistic;scale=1.2',
                                      eval_metric='SurvivalAft',
                                      verbose=0)

    _ = model_logistic.fit(train_pool, eval_set=val_pool)

    y_hat_dftest = model_logistic.predict(test_pool, prediction_type='Exponent')
    y_hat_dftest = y_hat_dftest.reshape(-1, 1)
    
    ids = np.arange(0, y_hat_dftest.shape[0]).reshape(-1, 1)
    y_pred = np.hstack((ids, y_hat_dftest))
    
    y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
    y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
    print(y_pred_df[:3])
    y_pred_df.to_csv("handle-missing-submission-01.csv", index=False)
