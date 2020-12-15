"""
Author:
    David Fellner
Description:
    Set settings for QDS and elements and save results to file to create a dataset executing a QDS. At first the grid is
    prepared and scenario settings are set. Then samples are created from raw data. These samples are timeseries of voltages
    at points of conenction of households and photovoltaics (PVs) of a low voltage distribution network.
    Finally a deep learning approach is compared to a linear classifier to either determine if a sample is from a term
    with PV (class 1) or no PV (class 0) or from a term with a regularly behaving PV (class 0) or a PV with a
    malfunctioning reactive power control curve (class 1). Additionally a dummy dataset can be created that only
    consists of samples that are constant over the entire timeseries (class 0) and samples that are not (class 1).
    Randomly chosen samples of either classes are plotted along with execution at default.
    See framework diagrams for a better overview.

    Choose experiment (dataset and learning settings) in experiment_config.py
    Predefined experiments vary the dataset type (dummy, PV vs no PV, regular PV vs malfunctioning PV) as well as the
    timeseries length of samples (1 day vs 1 week) and the number of samples (too little, 'just enough', sufficient to
    produce a meaningful output after training with the basic network design used, i.e. no Fscore ill defined because only
    always one class predicted in any run of cross validation; note that 1 day vs 7 days also means increasing the amount
    of data points, therefore redundant experiments (i.e. increasing the sample number even more for 1 day timeseries
    experiments was neglected to allow for a better orientation between experiments)

    Metrics: Deep learning approach should perform better than linear classifier (which just guesses between 0 and 1 class)
             meaning that a higher Fscore should be achieved
             Experiment configs state if this goal can be fulfilled with the experiment settings

                    Task      Dataset collection  ANN design  ANN tuning  Results     Report      Presentation
      Time planned: (Hours)   15                  7.5         15          7.5         10          4
      Time spent:   (Hours)   ~15                 25          ~15         5             to be seen
      Conclusion:   It took much longer than planned to actually get the RNN running and producing meaningful outputs
"""
import importlib

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config
import plotting
if not config.raw_data_available:
    from start_powerfactory import start_powerfactory
    from grid_preparation import prepare_grid
    from data_creation import create_data
from create_instances import create_samples
from malfunctions_in_LV_grid_dataset import MlfctinLVdataset
from PV_noPV_dataset import PVnoPVdataset
from dummy_dataset import Dummydataset
from RNN import RNN
from Transformer import Transformer

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler

import pandas as pd
import os

def generate_raw_data():

    if config.raw_data_available == False:
        for file in os.listdir(config.data_folder):
            if os.path.isdir(config.data_folder + file):

                print('Creating data using the grid %s' % file)
                app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
                curves = prepare_grid(app, file, o_ElmNet)

                create_data(app, o_ElmNet, curves, study_case_obj, file)
                print('Done with grid %s' % file)

        print('Done with all grids')

    return

def create_dataset():

    if config.dataset_available == False:
        print(
            "Dataset %s is created from raw data" % learning_config['dataset'])
        if (1/config.share_of_positive_samples).is_integer():
            df = pd.DataFrame()
            results_folder = config.results_folder + config.raw_data_set_name + '_raw_data' + '\\'
            for dir in os.listdir(results_folder):
                if os.path.isdir(results_folder + dir):
                    combinations_already_in_dataset = []  # avoid having duplicate samples (i.e. data of terminal with malfunction at same terminal and same terminals having a PV)
                    files = os.listdir(results_folder + dir)[0:int(config.simruns)]
                    for file in files:
                        samples, combinations_already_in_dataset = create_samples(results_folder + dir, file, combinations_already_in_dataset,
                                                                               len(df.columns))
                        df = pd.concat([df, samples], axis=1, sort=False)
            return df
        else:
            print("Share of malfunctioning samples wrongly chosen, please choose a value that yields a real number as an inverse i.e. 0.25 or 0.5")

def save_dataset(df):

    if config.dataset_available == False:
        df.to_csv(config.results_folder + learning_config['dataset'] + '.csv', header=True, sep=';', decimal='.', float_format='%.' + '%sf' % config.float_decimal)
        print(
            "Dataset %s saved" % learning_config['dataset'])

    return

def load_dataset(dataset=None):

    if not dataset:
        if learning_config['dataset'][:7] == 'PV_noPV':
            dataset = PVnoPVdataset(config.results_folder + learning_config["dataset"] + '.csv')
        elif learning_config['dataset'][:31] == 'malfunctions_in_LV_grid_dataset':
            dataset = MlfctinLVdataset(config.results_folder + learning_config["dataset"] + '.csv')
        else:
            dataset = Dummydataset(config.results_folder + learning_config["dataset"] + '.csv')


    else:
        if learning_config['dataset'][:7] == 'PV_noPV':
            dataset = PVnoPVdataset(config.test_data_folder + 'PV_noPV.csv')
        else:
            dataset = MlfctinLVdataset(config.test_data_folder + 'malfunctions_in_LV_grid_dataset.csv')

    X = dataset.get_x()
    y = dataset.get_y()
    return dataset, X, y

def cross_val(X, y, model):
    kf = KFold(n_splits=learning_config['k folds'])
    best_clfs = []
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = list(np.array(y)[train_index]), list(np.array(y)[test_index])

        X_train, X_test = model.preprocess(X_train, X_test)

        clfs, losses, lrs = model.fit(X_train, y_train, X_test, y_test, early_stopping=learning_config['early stopping'], control_lr=learning_config['LR adjustment'])
        best_model = choose_best(clfs)
        best_clfs.append(best_model)
        model.state_dict = best_model[0]
        scores.append(model.eval(X_test, y_test) + [best_model[1]])

    very_best_model = choose_best(best_clfs)
    model.state_dict = very_best_model[0]
    best_score = model.eval(X_test, y_test) + [very_best_model[1]]

    scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores], 'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores], 'Lowest validation loss': [i[2] for i in scores]}

    return model, scores_dict, best_score


def choose_best(models_and_losses):
    index_best = [i[1] for i in models_and_losses].index(min([i[1] for i in models_and_losses]))
    return models_and_losses[index_best]

def baseline(X, y):
    clf_baseline = SGDClassifier()
    scores = cross_validate(clf_baseline, X, y, scoring=learning_config["metrics"], cv=10, n_jobs=1)
    print("########## Linear Baseline: 10-fold Cross-validation ##########")
    for metric in learning_config["cross_val_metrics"]:
        print("%s: %0.2f (+/- %0.2f)" % (metric, scores[metric].mean(), scores[metric].std() * 2))

    return

def plot_samples(X, y):

    X_zeromean = np.array([x - x.mean() for x in X])  # deduct it's own mean from every sample
    X_train, X_test, y_train, y_test = train_test_split(X_zeromean, y, random_state=0, test_size=100)


    maxabs_scaler = MaxAbsScaler().fit(X_train)  # fit scaler as to scale training data between -1 and 1

    # samples = [y.index(0), y.index(1)]
    samples = [random.sample([i for i, x in enumerate(y) if x == 0], 1)[0],
               random.sample([i for i, x in enumerate(y) if x == 1], 1)[0]]
    print("Samples shown: #{0} of class 0; #{1} of class 1".format(samples[0], samples[1]))
    X_maxabs = maxabs_scaler.transform(X_zeromean[samples])
    plotting.plot_sample(X[samples], label=[y[i] for i in samples], title='Raw samples')
    plotting.plot_sample(X_zeromean[samples], label=[y[i] for i in samples], title='Zeromean samples')
    plotting.plot_sample(X_maxabs, label=[y[i] for i in samples], title='Samples scaled to -1 to 1')

if __name__ == '__main__':  #see config file for settings

    generate_raw_data()
    dataset = create_dataset()
    save_dataset(dataset)

    print("\n########## Configuration ##########")
    for key, value in learning_config.items():
        print(key, ' : ', value)
    print("number of samples : %d" % config.number_of_samples)

    dataset, X, y = load_dataset()
    if learning_config["plot samples"]:
        plot_samples(X, y)

    if learning_config['baseline']:
        baseline(X, y)

    print('X data with zero mean per sample and scaled between -1 and 1 based on training samples used')

    if learning_config['classifier'] == 'RNN':
        model = RNN(learning_config['RNN model settings'][0],  learning_config['RNN model settings'][1],
                    learning_config['RNN model settings'][2], learning_config['RNN model settings'][3])

    if not learning_config["cross_validation"]:

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=learning_config['train test split'])
        X_train, X_test = model.preprocess(X_train, X_test)
        print("\n########## Training ##########")
        clfs, losses, lrs = model.fit(X_train, y_train, X_test, y_test, early_stopping=learning_config['early stopping'], control_lr=learning_config['LR adjustment'])
        plotting.plot_2D([losses, [i[1] for i in clfs]], labels=['Training loss', 'Validation loss'], title='Losses after each epoch', x_label='Epoch', y_label='Loss')   #plot training loss for each epoch
        plotting.plot_2D(lrs, labels='learning rate', title='Learning rate for each epoch', x_label='Epoch',
                         y_label='Learning rate')
        clf = choose_best(clfs)
        model.state_dict = clf[0]                           #pick weights of best model found
        score = model.eval(X_test, y_test) + [clf[1]]
        print("\n########## Metrics ##########")
        print(
            "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\nLowest validation loss: {4}".format(score[0], score[1][0], score[1][1], score[1][2], score[2]))

    if learning_config["cross_validation"]:
        print("\n########## k-fold Cross-validation ##########")
        model, scores, best_score = cross_val(X, y, model)
        print("########## Metrics ##########")
        for score in scores:
            print("%s: %0.2f (+/- %0.2f)" % (score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))
        print("\n########## Best Model found ##########")
        print(
            "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\nlowest validation loss: {4}".format(best_score[0],best_score[1][0], best_score[1][1],
                                                                                                   best_score[1][2], best_score[2]))

    a = 1






