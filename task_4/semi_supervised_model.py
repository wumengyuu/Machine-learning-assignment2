import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from task_3.task3_1 import ImputeData
from task_4.visualise_data import plot_data_imputation
from task_4.frozen_transformer import FrozenTransformer

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def get_model():
    # Impute missing values with the best imputers from Task 3.1
    imp = SimpleImputer()
    df = pd.read_csv("../data/train_data.csv")

# I don't think X can include Censored values because they are not in
# test_data.csv
    X_raw = df[['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk',
                'ComorbidityIndex', 'TreatmentResponse']]
    y = df[['SurvivalTime']]

# Apparently, X is supposed to be the union of the unsupervised
# and (train) supervised feature datasets, so I put the whole X there
# I don't think that's correct tho
    X = imp.fit_transform(X_raw)

# I tried to see what the data looks like before and after imputation,
# but they are exactly the same probably because I don't understand how
# it works yet. The entire code is from chatgpt, so it's just for
# the development process


    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Try different numbers of components.
    iso = Isomap(n_components=2)
    iso.fit(X)

    pipe = make_pipeline(SimpleImputer(),
                         scaler,
                         FrozenTransformer(iso), # <- Here is the Frozen Isomap
                         LinearRegression())

    # (X_train, y_train) is the labeled, supervised data
    X_train, y_train, X_val, X_test, y_val, y_test = ImputeData(df)
    c_test = X_test[['Censored']].values
# Columns must be the same as the ones in the training data (line 24)
    X_test = X_test[['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk',
                'ComorbidityIndex', 'TreatmentResponse']]
    X_train = X_train[['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk',
                'ComorbidityIndex', 'TreatmentResponse']]

    plot_data_imputation(X_raw, X_train, 'Age', 'GeneticRisk', 'Gender')
    pipe.fit(X_train, y_train)

    y_hat_test = pipe.predict(X_test)
    cMSE_test = error_metric(y_test, y_hat_test, c_test)
    print('cMSE_test:', cMSE_test)
    return pipe

def create_submission():
    model = get_model()
    dftest = pd.read_csv("../data/test_data.csv")
    dftest = pd.DataFrame(dftest)
    #dftest = dftest.drop(columns=['Id'])
    X_dftest_test = dftest[['Age', 'Gender', 'Stage', 'TreatmentType', 'GeneticRisk',
                'ComorbidityIndex', 'TreatmentResponse']]

    yhat_dftest_test = model.predict(X_dftest_test)

    # save yhat_dftest_test as csv
    ids = np.arange(0, yhat_dftest_test.shape[0]).reshape(-1, 1)
    y_pred = np.hstack((ids, yhat_dftest_test))

    y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
    y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
    print(y_pred_df[:3])

    y_pred_df.to_csv("submissions/semisupervised-submission-02.csv", index=False)


create_submission()
