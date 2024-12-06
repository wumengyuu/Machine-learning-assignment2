import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

def dataprepare(df):
    df = pd.DataFrame(df)
    # Drop missing columns
    df = df.drop(df.columns[0], axis=1)     # drop first column
    df = df.dropna(subset=['SurvivalTime']) # drop row if SurvivalTime is missing
    df_clean = df.dropna(axis=1, how='any') # drop missing columns
    return df_clean

def missingno_plot(df):
    # Plot missing values 
    msno.bar(df)
    msno.matrix(df) 
    msno.dendrogram(df)
    plt.show()

def get_Xyc(df):
    # Get X, y, c
    X = df[['Age', 'Gender', 'Stage', 'TreatmentType']]
    y = df[['SurvivalTime']]
    c = df[['Censored']]
    return X, y, c

## Raw data
df = pd.read_csv("train_data.csv")
missingno_plot(df)

## Cleaned data
df_clean= dataprepare(df)
missingno_plot(df_clean)

## Get X, y, c
X, y, c = get_Xyc(df)
print(X.shape, y.shape, c.shape)

