import numpy as np
import pandas as pd


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


def gradient_descent(X, y, theta, C, lam, alpha, iterations):

    for _ in range(iterations):
        gradient = cMSE_gradient(X, y, theta, C, lam)
        theta -= alpha * gradient  # Update theta

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

################################################################
# Data (X, y)
data = pd.read_csv("train_data.csv")
df_clean = data_cleaning(data)
X, y, c = get_Xy(df_clean)


# Parameter()
d = X.shape[1] # feature 
theta = np.random.randn(d).astype(np.float64)* 0.1  # Make sure theta is a float type
lambda_reg = 0.1
learning_rate = 0.01
num_iterations = 100

# run Gradient Descent
optimized_theta = gradient_descent(X, y, theta, c, lambda_reg, learning_rate, num_iterations)
print("Optimized theta:", optimized_theta)


################################################################
# test data
data_test = pd.read_csv("test_data.csv")
data_test = pd.DataFrame(data_test)
data_test = data_test.dropna(axis=1, how='any') # drop missing columns
X_test = data_test[['Age', 'Gender', 'Stage', 'TreatmentType']].values

# Predict SurvivalTime for new data
y_pred = X_test @ optimized_theta 
print("Predictions for new data:", y_pred)

## save to csv file
ids = np.arange(0, y_pred.shape[0]).reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)
y_pred = np.hstack((ids, y_pred))

y_pred_df = pd.DataFrame(y_pred, columns=["id", "SurvivalTime"])
y_pred_df['id'] = y_pred_df['id'].astype(np.int32)
print(y_pred_df.head)
y_pred_df.to_csv("cMSE-baseline-submission-02.csv", index=False)



# import matplotlib.pyplot as plt

# # Plot the distribution of SurvivalTime
# plt.hist(y, bins=50)
# plt.title("Distribution of SurvivalTime")
# plt.show()