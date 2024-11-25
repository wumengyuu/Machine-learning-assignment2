import numpy as np
import pandas as pd
from HadleMissing import *

df_train_raw = pd.read_csv("train_data.csv")
df_train_raw.drop(columns=df_train_raw.columns[0], axis=1, inplace=True)
df_train_raw1 = df_train_raw.dropna(subset=['SurvivalTime'])
df_train = ImputeData(df_train_raw1)

df_test_raw = pd.read_csv("test_data.csv")
df_test_raw.drop(columns=df_test_raw.columns[0], axis=1, inplace=True)
df_test = Impute_test_data(df_test_raw)

predict_with_AFT(df_train, df_test)