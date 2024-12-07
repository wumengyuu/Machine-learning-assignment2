import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def semi_dataprepare(df):
    df = pd.DataFrame(df)

    ## Plot missing values 
    # msno.bar(df)
    # msno.heatmap(df)
    # msno.matrix(df)
    # msno.dendrogram(df)
    plt.show()

    # Drop missing columns
    df_clean = df.dropna(subset=['SurvivalTime'])  # drop row if SurvivalTime is missing
    #df_clean = df_clean.dropna(axis=1, how='any')  # drop missing columns

    # creat X, y
    X = df_clean[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'Censored',
                'ComorbidityIndex', 'TreatmentResponse']]

    y = df_clean[['SurvivalTime']]

    # split the data
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

    return X_train, y_train, X_test, y_test, X_val, y_val


## Dataframes
# df = pd.read_csv("train_data.csv")
# X_train, y_train, X_val, X_test, y_val, y_test = dataprepare(df)
# print(X_train.shape, X_val.shape)
