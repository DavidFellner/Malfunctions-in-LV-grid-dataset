import numpy as np
import matplotlib.pyplot as plt
import importlib

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def plot_sample(Y, x=None, label = None, title=None):

    fig, ax = plt.subplots()
    marker = '.'
    if config.sample_length <= 96:
        markersize = 2
    elif config.sample_length <= 7*96:
        markersize = 1
    else:
        markersize = 0.5

    if not x:
        if isinstance(Y, np.ndarray):
            x = np.linspace(0, len(Y[0]), len(Y[0]))
            for i in range(len(Y)):
                if label:
                    ax.plot(x, Y[i], marker, label = label[i], markersize=markersize)
                else:
                    ax.plot(x, Y[i], marker, markersize=markersize)
        else:
            x = np.linspace(0, len(Y), len(Y))
            if label:
                ax.plot(x, Y, marker, label=label, markersize=markersize)
            else:
                ax.plot(x, Y, marker, markersize=markersize)
    else:
        if isinstance(Y, np.ndarray):
            for i in range(len(Y)):
                if label:
                    ax.plot(x, Y[i], marker, label=label[i], markersize=markersize)
                else:
                    ax.plot(x, Y[i], marker, markersize=markersize)
        else:
            if label:
                ax.plot(x, Y, marker, label=label, markersize=markersize)
            else:
                ax.plot(x, Y, marker, markersize=markersize)

    fig.show()
    if label:
        plt.legend(loc="best", markerscale=10)
    if title:
        plt.title(title)

    return ax

def plot_2D(y, x=None, labels=None, title=None, x_label=None, y_label=None):

    fig, ax = plt.subplots()

    if not x:
        if type(y[0]) == list:
            x = np.linspace(0, len(y[0]), len(y[0]))
        else:
            x = np.linspace(0, len(y), len(y))

    if labels:
        if type(y[0]) == list:
            for i in list(range(len(y))):
                ax.plot(x, y[i], label=labels[i])
        else:
            ax.plot(x, y, label=labels)
    else:
        if type(y[0]) == list:
            for i in list(range(len(y))):
                ax.plot(x, y[i])
        else:
            ax.plot(x, y)

    fig.show()
    if labels:
        plt.legend(loc="best", markerscale=10)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    return ax