import numpy as np
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