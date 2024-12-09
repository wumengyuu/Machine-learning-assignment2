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
from catboost import CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def ImputeData(df_raw):
    df_raw.drop(columns=df_raw.columns[0], axis=1, inplace=True)
    #df = df_raw
    df = df_raw.dropna(subset=['SurvivalTime'])
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

def interval_mae(y_true_lower, y_true_upper, y_pred):
    mae = np.where((y_true_lower <= y_pred) & (y_pred <= y_true_upper),
                   0,
                   np.minimum(np.abs(y_true_lower-y_pred),
                              np.abs(y_true_upper-y_pred))) 
    return mae.mean()

def predict_with_AFT(df):
    df['y_lower'] = df['SurvivalTime']
    df['y_upper'] = np.where(df['Censored'], df['SurvivalTime'], -1)
    
    stratifying_column = df['Censored']
    df = df.drop(['Censored', 'SurvivalTime'], axis=1)

    X = df[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']]
    y = df[['y_lower', 'y_upper']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratifying_column, random_state=32)
    
    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']
  
    
    train_pool = Pool(X_train, label=y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, label=y_test, cat_features=categorical_features)

    model_normal = CatBoostRegressor(iterations=500,
                                    loss_function='SurvivalAft:dist=Normal',
                                    eval_metric='SurvivalAft',
                                    verbose=0)
    model_logistic = CatBoostRegressor(iterations=500,
                                      loss_function='SurvivalAft:dist=Logistic;scale=1.2',
                                      eval_metric='SurvivalAft',
                                      verbose=0)
    model_extreme = CatBoostRegressor(iterations=500,
                                      loss_function='SurvivalAft:dist=Extreme;scale=2',
                                      eval_metric='SurvivalAft',
                                      verbose=0)
    _ = model_normal.fit(train_pool, eval_set=test_pool)
    _ = model_logistic.fit(train_pool, eval_set=test_pool)
    _ = model_extreme.fit(train_pool, eval_set=test_pool)

    train_predictions = pd.DataFrame({
        'y_lower': y_train['y_lower'],
        'y_upper': y_train['y_lower'],
        'preds_normal': model_normal.predict(train_pool, prediction_type='Exponent'),
        'preds_logistic': model_logistic.predict(train_pool, prediction_type='Exponent'),
        'preds_extreme': model_extreme.predict(train_pool, prediction_type='Exponent')
    })
    
    train_predictions['y_upper'] = np.where(train_predictions['y_upper'] == -1, np.inf, train_predictions['y_upper'])

    test_predictions = pd.DataFrame({
        'y_lower': y_test['y_lower'],
        'y_upper': y_test['y_lower'],
        'preds_normal': model_normal.predict(test_pool, prediction_type='Exponent'),
        'preds_logistic': model_logistic.predict(test_pool, prediction_type='Exponent'),
        'preds_extreme': model_extreme.predict(test_pool, prediction_type='Exponent')
    })

    test_predictions['y_upper'] = np.where(test_predictions['y_upper'] == -1, np.inf, test_predictions['y_upper'])


    distributions = ['normal', 'logistic', 'extreme']
    print('Interval MAE')
    for dist in distributions:
        train_metric = interval_mae(train_predictions['y_lower'], train_predictions['y_upper'], train_predictions[f'preds_{dist}'])
        test_metric = interval_mae(test_predictions['y_lower'], test_predictions['y_upper'], test_predictions[f'preds_{dist}'])
        print(f'Train set. dist:{dist}: {train_metric:0.2f}')
        print(f'Test set. dist:{dist}: {test_metric:0.2f}')
        print('---------------------------')

def getTrainTest(df_raw):
    df_raw.drop(columns=df_raw.columns[0], axis=1, inplace=True)
    df = df_raw.dropna(subset=['SurvivalTime'])

    columns = ['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'SurvivalTime', 'Censored']


    # define X and Y
    X = df[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'Censored']]
    y = df[['SurvivalTime']]

    # split the data
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
    
    return X_train, y_train, X_val, X_test, y_val, y_test

def buildDecisionTree(X_train, y_train, X_val, X_test, y_val, y_test):
    c_train = X_train[['Censored']].values
    X_train = X_train.drop(columns=['Censored'])

    c_val = X_val[['Censored']].values
    X_val = X_val.drop(columns=['Censored'])

    c_test = X_test[['Censored']].values
    X_test = X_test.drop(columns=['Censored'])
        
    pipe = Pipeline([('std', StandardScaler()),
                     #('estimator', DecisionTreeRegressor())])
                     #('estimator', HistGradientBoostingRegressor(learning_rate=0.01,
                     #                                            max_iter=80,
                     #                                            #max_features=0.3 
                     #                                           )
                      ('estimator', CatBoostRegressor(learning_rate=0.01, iterations=700, 
                                                      depth=13, l2_leaf_reg=6,
                                                     verbose=10))])
    pipe.fit(X_train, y_train, estimator__eval_set=(X_val, y_val), estimator__early_stopping_rounds=80)
    #pipe.fit(X_train, y_train)
    catboost_model = pipe.named_steps['estimator']
    best_score = catboost_model.get_best_score()['validation']['RMSE']
    
    y_hat_val = pipe.predict(X_val).reshape(-1, 1)
    cMSE_val = error_metric(y_val, y_hat_val, c_val) # cross validation

    y_hat_test = pipe.predict(X_test).reshape(-1, 1)
    cMSE_test = error_metric(y_test, y_hat_test, c_test) # cross validation

    plot_y_yhat(y_val, y_hat_val)

   
    return cMSE_val, cMSE_test, best_score

df_raw = pd.read_csv("train_data.csv")
categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'Censored']

# Predict using Decision trees
#df_categorized = df_raw[categorical_features].astype('category')
#X_train, y_train, X_val, X_test, y_val, y_test = getTrainTest(df_raw)

#cMSE_val, cMSE_test, best_score = buildDecisionTree(X_train, y_train, X_val, X_test, y_val, y_test)
#print('cMSE_val:', cMSE_val)
#print('cMSE_test:', cMSE_test)
#print('RMSE test score: ', best_score)


# Predict using AFT

df = ImputeData(df_raw)
predict_with_AFT(df)
