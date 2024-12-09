from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from PlotsErrors import *

def prepare_for_AFT(df):
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
    
    plot_y_yhat(y_test['y_lower'], test_predictions['preds_logistic'], plot_title="y_hat")

def prepare_for_AFT_Unimputed(df):
    df['y_lower'] = df['SurvivalTime']
    df['y_upper'] = np.where(df['Censored'], df['SurvivalTime'], -1)

    df = df.dropna(subset=['y_lower', 'y_upper'])
    
    stratifying_column = df['Censored']
    df = df.drop(['Censored', 'SurvivalTime'], axis=1)

    X = df[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']]
    y = df[['y_lower', 'y_upper']]

    
    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse']

    for feature in categorical_features:
        X[feature] = X[feature].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratifying_column, random_state=32)

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

    plot_y_yhat(y_test['y_lower'], test_predictions['preds_normal'], plot_title="y_hat")

