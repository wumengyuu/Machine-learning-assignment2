import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_y_yhat import plot_y_yhat
from task_3.task3_1 import ImputeData
from task_4.semi_datapreparation import semi_dataprepare
from task_4.frozen_transformer import FrozenTransformer

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

# Impute missing values with the best imputers from Task 3.1
imp = SimpleImputer()
# X is the union of the unsupervised and (train) supervised feature datasets
df = pd.read_csv("../data/train_data.csv")
X = df[['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk',
                'ComorbidityIndex', 'TreatmentResponse', 'Censored']]

X = imp.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Try different numbers of components.
iso = Isomap(n_components=2)
iso.fit(X)

pipe = make_pipeline(SimpleImputer(),
                     scaler,
                     FrozenTransformer(iso), # <- Here is the Frozen Isomap
                     LinearRegression())

X_train, y_train, X_val, X_test, y_val, y_test = ImputeData(df)
# (X_train, y_train) is the labeled, supervised data
pipe.fit(X_train, y_train)

y_hat_test = pipe.predict(X_test)
plot_y_yhat(y_test, y_hat_test)
cMSE_test = error_metric(y_test, y_hat_test, X_test[['Censored']].values)
print('cMSE_test:', cMSE_test)
