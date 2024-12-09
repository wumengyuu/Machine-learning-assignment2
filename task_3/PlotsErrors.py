import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def plot_y_yhat(y_val,y_pred, plot_title = "plot"):
    labels = 'SurvivalTime'
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val),MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10,10))
    x0 = np.min(y_val)
    x1 = np.max(y_val)
    plt.scatter(y_val, y_pred)
    plt.xlabel('True ' + labels)
    plt.ylabel('Predicted ' + labels)
    plt.plot([x0, x1], [x0, x1], color='red')
    plt.axis('square')
    plt.show()
    
def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]    

def interval_mae(y_true_lower, y_true_upper, y_pred):
    mae = np.where((y_true_lower <= y_pred) & (y_pred <= y_true_upper),
                   0,
                   np.minimum(np.abs(y_true_lower-y_pred),
                              np.abs(y_true_upper-y_pred))) 
    return mae.mean()

def plot_error(error: dict, plot_title = "Plot", xlabel = "X", ylabel = "Y"):
    plt.plot(list(error.keys()), list(error.values()))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.show()