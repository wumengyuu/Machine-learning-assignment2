import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def cMSE_gradient(X, y, theta, C, lam):

    N = X.shape[0]  # Number of samples
    # Initialize gradient
    gradient = np.zeros_like(theta)
    y_hat = X @ theta       # Compute predictions (theta^T * x_n)

    for n in range(N):
        if y[n] > y_hat[n]:  # Case where max(0, theta^T x_n - y_n) = 0
            gradient += 2 * X[n] * (1 - C[n]) * (y_hat[n] - y[n])
        else:  # Case where max(0, theta^T x_n - y_n) = (theta^T x_n - y_n)
            gradient += 2 * X[n] * (1 - C[n]) * (y_hat[n] - y[n]) + 2 * X[n] * C[n] * (y_hat[n] - y[n])
    
    # Add Ridge Regularization term
    gradient += 2 * lam * theta

    return (2 / N) * gradient



def gradient_descent(X, y, theta, C, lam, alpha, iterations, decay_rate=0.01):
    for i in range(iterations):
        gradient = cMSE_gradient(X, y, theta, C, lam)
        theta -= alpha * gradient
        
        # Decrease learning rate after each iteration (exponential decay)
        alpha *= (1 / (1 + decay_rate * i))
    
    return theta
################################################################
def data_cleaning(data):
    df = pd.DataFrame(data)
    df_clean = df.dropna(subset=['SurvivalTime']) # drop row if SurvivalTime is missing
    df_clean = df_clean.dropna(axis=1, how='any') # drop missing columns
    return df_clean

def get_Xy(df_clean):
    X = df_clean[['Age', 'Gender', 'Stage', 'TreatmentType']].values
    y = df_clean[['SurvivalTime']].values.ravel()  
    c = df_clean[['Censored']].values.ravel()     

    # print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))
    return X, y, c

def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

def scaler(input):
    scaler = StandardScaler()
    fitinput = scaler.fit_transform(input)
    output = scaler.transform(fitinput) 
    return output

################################################################
# Data (X, y)
data = pd.read_csv("train_data.csv")
df_clean = data_cleaning(data)
X, y, c = get_Xy(df_clean)

X = scaler(X)

# Parameter()
d = X.shape[1] # feature 
theta = np.random.randn(d).astype(np.float64)   # Make sure theta is a float type
lambda_reg = 0.001
learning_rate = 0.01
num_iterations = 500

# run Gradient Descent
optimized_theta = gradient_descent(X, y, theta, c, lambda_reg, learning_rate, num_iterations)

## run in train data
y_pred = X @ optimized_theta 
# print("Predictions for train data:", y_pred)

# compute error
cmse_test = error_metric(y, y_pred, c)
print("CMSE for test data:", cmse_test)

################################################################
# lambda_vals = [0.001, 0.01, 0.1, 1]
# learning_rates = [0.001, 0.01, 0.1]
# num_iterations = [100, 200, 500]

# best_error = float("inf")
# best_params = {}

# for lam in lambda_vals:
#     for lr in learning_rates:
#         for iter in num_iterations:
#             optimized_theta = gradient_descent(X, y, theta, c, lam, lr, iter)
#             validation_error = error_metric(y, X @ optimized_theta, c)
            
#             if validation_error < best_error:
#                 best_error = validation_error
#                 best_params = {'lambda': lam, 'learning_rate': lr, 'iterations': iter}

# print("Best Parameters:", best_params)

################################################################
# test data
data_test = pd.read_csv("test_data.csv")
data_test = pd.DataFrame(data_test)
data_test = data_test.dropna(axis=1, how='any') # drop missing columns
X_test = data_test[['Age', 'Gender', 'Stage', 'TreatmentType']].values

X_test = scaler(X_test)

# Predict SurvivalTime for new data
y_test_pred = X_test @ optimized_theta 
# print("Predictions for new data:", y_test_pred)

## save to csv file
def save_submission(data):
    ids = np.arange(0, data.shape[0]).reshape(-1, 1)
    data = data.reshape(-1, 1)
    data = np.hstack((ids, data))
    data = pd.DataFrame(data, columns=["id", "SurvivalTime"])
    data['id'] = data['id'].astype(np.int32)
    return data

y_test_pred = save_submission(y_test_pred)
# y_test_pred.to_csv("cMSE-baseline-submission-04.csv", index=False)
print(y_test_pred.head)

# check null in y_test_pred
print(y_test_pred.isnull().sum())
