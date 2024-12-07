import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from error_metric import error_metric
from plot_y_yhat import plot_y_yhat, plot_error
from task_3.task3_1 import ImputeData
from task_4.semi_datapreparation import semi_dataprepare
from task_4.frozen_transformer import FrozenTransformer

# Impute missing values with the best imputers from Task 3.1
imp = SimpleImputer()
# X is the union of the unsupervised and (train) supervised feature datasets
df = pd.read_csv("../data/train_data.csv")
X = df[['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk',
                'ComorbidityIndex', 'TreatmentResponse']]

X = imp.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

#X_train, y_train, X_val, X_test, y_val, y_test = ImputeData(df)
X_train, y_train, X_test, y_test, X_val, y_val = semi_dataprepare(df)

c_test = X_test[['Censored']].values
X_test = X_test.drop(columns=['Censored'])
X_train = X_train.drop(columns=['Censored'])
X_val = X_val.drop(columns=['Censored'])

errors = {}
min_error = np.inf
min_error_model = None
min_n_of_components = 0
# Try different numbers of components.
for i in range(1, 20):
    iso = Isomap(n_components=i)
    iso.fit(X)

    pipe = make_pipeline(SimpleImputer(),
                         scaler,
                         FrozenTransformer(iso), # <- Here is the Frozen Isomap
                         LinearRegression())

    # (X_train, y_train) is the labeled, supervised data
    #print("train: ",X_train.columns)

    pipe.fit(X_train, y_train)

    y_hat_test = pipe.predict(X_test)
    #plot_y_yhat(y_test, y_hat_test)
    cMSE_test = error_metric(y_test, y_hat_test, c_test)
    #print(f'cMSE_test for {i} components: ', cMSE_test)
    errors[i] = cMSE_test[0]
    if cMSE_test[0] < min_error:
        min_error = cMSE_test[0]
        min_error_model = pipe
        min_n_of_components = i

plot_error(errors, plot_title='cMSE_test for different number of components', xlabel='Number of components', ylabel='cMSE_test')
#print minimal value from errors and its key
print("Minimal cMSE_test value: ", min_error, " was achieved using ", min_n_of_components, " components.")


dftest = pd.read_csv("../data/test_data.csv")
dftest = pd.DataFrame(dftest)
#dftest = dftest.drop(columns=['Id'])
X_dftest_test = dftest[['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType',
                'ComorbidityIndex', 'TreatmentResponse']]

yhat_dftest_test = min_error_model.predict(X_dftest_test)

# save yhat_dftest_test as csv
ids = np.arange(0, yhat_dftest_test.shape[0]).reshape(-1, 1)
y_pred = np.hstack((ids, yhat_dftest_test))

y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
print(y_pred_df[:3])

y_pred_df.to_csv("submissions/semisupervised-submission-04.csv", index=False)
