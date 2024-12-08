import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from task_4.plot_y_yhat import plot_y_yhat
from sklearn.preprocessing import StandardScaler


def dataprepare(df):
    df = pd.DataFrame(df)
    # Drop missing columns
    df = df.drop(df.columns[0], axis=1)     # drop first column
    df = df.dropna(subset=['SurvivalTime']) # drop row if SurvivalTime is missing
    df_clean = df.dropna(axis=1, how='any') # drop missing columns
    return df_clean

def get_Xyc(df):
    # Get X, y, c
    X = df[['Age', 'Gender', 'Stage', 'TreatmentType']]
    y = df[['SurvivalTime']]
    c = df[['Censored']]
    return X, y, c

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def xgboost_model(X_train, y_train, X_test, y_test, c_test):
    # Initialize XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_hat_test = model.predict(X_test)
    cMSE_test = error_metric(y_test.values, y_hat_test, c_test.values)
    return cMSE_test, y_hat_test

def Kfold_CV(K, X, y, c):
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    cnt = 0
    cmse_list = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        c_train, c_test = c.iloc[train_index], c.iloc[test_index]
        print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
        
        cMSE_test, y_hat_test = xgboost_model(X_train, y_train, X_test, y_test, c_test)
        print(f'cMSE: {cMSE_test}')
        cnt += 1
        cmse_list.append(cMSE_test)
        
        # Plot predictions
        title = f'Fold {cnt}'
        plot_y_yhat(y_test, y_hat_test, title)

    return cmse_list

def scaler(input):
    scaler = StandardScaler()
    fitinput = scaler.fit_transform(input)
    output = scaler.transform(fitinput) 
    return output

# Load and prepare the data
data = pd.read_csv("train_data.csv")
df_clean = dataprepare(data)
X, y, c = get_Xyc(df_clean)


# K-fold cross validation
K = 5
cmse_list = Kfold_CV(K, X, y, c)

# Plot K-fold performance
plt.plot(range(1, K+1), cmse_list, '-o')
plt.title('K-fold Cross Validation')
plt.xlabel('Fold')
plt.ylabel('CMSE')
plt.show()

# Final model evaluation
df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
X_train, y_train, c_train = get_Xyc(df_train)
X_test, y_test, c_test = get_Xyc(df_test)

cMSE_test, y_hat_test = xgboost_model(X_train, y_train, X_test, y_test, c_test)
print(f'Final Test cMSE: {cMSE_test}')
