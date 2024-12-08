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

    def plot_error(error: dict, plot_title = "Plot", xlabel = "X", ylabel = "Y"):
    plt.plot(list(error.keys()), list(error.values()))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.show()


def plot_all_error_stats(data: dict, plot_title = "Plot", xlabel = "X", ylabel = "Y"):
    keys = list(data.keys())
    means = [np.mean(values) for values in data.values()]
    std_devs = [np.std(values) for values in data.values()]
    mins = [np.min(values) for values in data.values()]
    maxs = [np.max(values) for values in data.values()]
    x = range(len(keys))
    plt.figure(figsize=(10, 6))
    plt.plot(x, means, label='Mean', color='blue', marker='o', linewidth=2)
    plt.plot(x, mins, label='Min', color='red', marker='o', linewidth=2)
    plt.plot(x, maxs, label='Max', color='red', marker='o', linewidth=2)

    plt.fill_between(x,
                     [m - s for m, s in zip(means, std_devs)],
                     [m + s for m, s in zip(means, std_devs)],
                     color='blue', alpha=0.2, label='± Std Dev')

    plt.xticks(x, keys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()

    plt.tight_layout()
    plt.show()
