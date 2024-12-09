import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

def ImputeData(df):
    #df_raw.drop(columns=df_raw.columns[0], axis=1, inplace=True)
    #df = df_raw.dropna(subset=['SurvivalTime'])
    
    # Univariate imputation
    #imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    #imp.fit(df)
    
    # Multivariate impuatation
    imp = IterativeImputer(missing_values=np.nan, 
    #                       initial_strategy='most_frequent',
    #                       imputation_order='ascending',
                           #estimator=BayesianRidge(),
                           estimator=RandomForestRegressor(n_estimators=10, random_state=0),
                           random_state=0,
                           #n_nearest_features=20,
                            max_iter=10,
                           sample_posterior=False)
    imp.fit(df)
    columns = ['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'SurvivalTime', 'Censored']
    df_imp = pd.DataFrame(imp.transform(df), columns=columns)

    categorical_features = ['Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex', 'TreatmentResponse', 'Censored']
    encoder = LabelEncoder()

    for feature in categorical_features:
        df_imp[feature] = encoder.fit_transform(df_imp[feature])
    
    return df_imp