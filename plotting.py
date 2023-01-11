import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config


def plot_sample(Y, x=None, label = None, title=None, save=False, figname=None):

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
    if save:
        plt.savefig(figname + '.png')

    return ax

def plot_2D(y, x=None, labels=None, title=None, x_label=None, y_label=None, save=False, figname=None, style='-'):

    fig, ax = plt.subplots()

    if not x:
        if type(y[0]) == list:
            x = np.linspace(0, len(y[0])+1, len(y[0]))
        else:
            x = np.linspace(0, len(y)+1, len(y))

    if labels:
        if type(y[0]) == list:
            for i in list(range(len(y))):
                ax.plot(x, y[i], style, label=labels[i])
        else:
            ax.plot(x, y, style, label=labels)
    else:
        if type(y[0]) == list:
            for i in list(range(len(y))):
                ax.plot(x, y[i], style)
        else:
            ax.plot(x, y, style)

    fig.show()
    if labels:
        plt.legend(loc="best", markerscale=10)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if save:
        plt.savefig(figname + '.png')
        plt.savefig(figname + '.pdf', dpi=fig.dpi, bbox_inches='tight', format='pdf')

    return ax

def plot_time_sweep():
    path = os.path.join(config.models_folder, learning_config['classifier'])
    file = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_training_time_sweep" + "_" + 'result.txt')
    figure_name = os.path.join(path, learning_config["dataset"] + "_training_time_sweep")

    f = open(file, 'r')
    text = f.read()
    F_scores = [float(i.split('\n')[0][:4]) for i in text.split('FScore: ')[1:]]
    Precisions = [float(i.split('\n')[0][:5]) for i in text.split('Precision: ')[1:]]
    Recalls = [float(i.split('\n')[0][:5]) for i in text.split('Recall: ')[1:]]
    plot_2D([F_scores, Precisions, Recalls], x=list(range(1,learning_config['number of epochs']+1)), labels=['F-score', 'Precision', 'Recall'], x_label='Training time', y_label = 'Scores', save=True, figname=figure_name)


def plot_hyp_para_tuning():
    path = os.path.join(config.models_folder, learning_config['classifier'])
    file = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_hyper_parameter_analysis_on_" + learning_config["hyperparameter tuning"][0] + "_" + 'result.txt')
    figure_name = os.path.join(path, learning_config["dataset"] + "_hyper_parameter_analysis_on_" + learning_config["hyperparameter tuning"][0])

    f = open(file, 'r')
    text = f.read()
    F_scores = [float(i.split('\n')[0][:4]) for i in text.split('FScore: ')[1:]]
    Precisions = [float(i.split('\n')[0][:5]) for i in text.split('Precision: ')[1:]]
    Recalls = [float(i.split('\n')[0][:5]) for i in text.split('Recall: ')[1:]]
    plot_2D([F_scores, Precisions, Recalls], x=learning_config["hyperparameter tuning"][1], labels=['F-score', 'Precision', 'Recall'], x_label=learning_config["hyperparameter tuning"][0] + ' Values', y_label = 'Scores', save=True, figname=figure_name) #'Number of RNN Attention Blocks Values' #learning_config["hyperparameter tuning"][0] +

def plot_grid_search():
    path = os.path.join(config.models_folder, learning_config['classifier'])
    file = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_gridsearch_on_" + learning_config["grid search"][0]
                                + "_" + 'result.txt')
    figure_name = os.path.join(path, learning_config["dataset"] + "_gridsearch_on_" + learning_config["grid search"][0])

    f = open(file, 'r')
    text = f.read()
    F_scores = [float(i.split('\n')[0][:4]) for i in text.split('FScore: ')[1:]]
    Precisions = [float(i.split('\n')[0][:5]) for i in text.split('Precision: ')[1:]]
    Recalls = [float(i.split('\n')[0][:5]) for i in text.split('Recall: ')[1:]]
    plot_2D([F_scores, Precisions, Recalls], x=learning_config["grid search"][1], labels=['F-score', 'Precision', 'Recall'], x_label=learning_config["grid search"][0] + ' Values', y_label = 'Scores', save=True, figname=figure_name) #learning_config["grid search"][0] + ' Values'

def plot_estimate_vs_target_by_load(y, y_pred_nn, y_pred_lr, style='-', phase='phase1', setup='A'):
    path = config.load_estimation_folder

    for load in list(y.columns)[0::2]:
        load_name = load.split('_')[0]
        figure_name = os.path.join(path, f'{phase}_setup_{setup}_estimation_{load_name}')
        y_load_P = list(y[load_name + '_P'].values)
        y_load_Q = list(y[load_name + '_Q'].values)
        y_pred_nn_P = list(y_pred_nn[load_name + '_P'].values)
        y_pred_nn_Q = list(y_pred_nn[load_name + '_Q'].values)
        y_pred_lr_P = list(y_pred_lr[load_name + '_P'].values)
        y_pred_lr_Q = list(y_pred_lr[load_name + '_Q'].values)
        #labels=['Target P', 'Target Q', 'Prediction NN P', 'Prediction NN Q', 'Prediction LR P', 'Prediction LR Q']
        labels_P = ['Target P', 'Estimate NN P', 'Estimate LR P']
        labels_Q = ['Target Q', 'Estimate NN Q', 'Estimate LR Q']

        #plot_2D([y_load_P, y_load_Q,y_pred_nn_P, y_pred_nn_Q,y_pred_lr_P, y_pred_lr_Q], labels=labels, title='Load estimation vs. target values', x_label='Timestep', y_label='Load [kW]', save=True, figname=figure_name)
        plot_2D([y_load_P, y_pred_nn_P, y_pred_lr_P ], labels=labels_P,
                title='Load estimation vs. target values', x_label='Timestep', y_label='Load [kW]', save=True,
                figname=figure_name + '_P', style=style)
        plot_2D([y_load_Q, y_pred_nn_Q, y_pred_lr_Q], labels=labels_Q,
                title='Load estimation vs. target values', x_label='Timestep', y_label='Load [kVA]', save=True,
                figname=figure_name + '_Q', style=style)


