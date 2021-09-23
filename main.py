"""
Author:
    David Fellner (no Software engineer by training, so please don't get enraged)
Description:
    Set settings for QDS (quasi dynamic load flow simulation in a power grid) and elements and save results to file to
    create a data set executing a QDS.
    At first the grid is prepared and scenario settings are set. Then samples are created from raw data.
    These samples are time series of voltages at points of connection of households and photovoltaics (PVs) of a low
    voltage distribution network. Finally a deep learning approach is compared to a linear classifier to either
    determine if a sample is from a term with PV (class 1) or no PV (class 0) or from a term with a regularly
    behaving PV (class 0) or a PV with a malfunctioning reactive power control curve (class 1).
    Additionally a dummy dataset can be created that only consists of samples that are constant over the entire
    timeseries (class 0) and samples that are not (class 1). Randomly chosen samples of either classes are plotted
    along with execution at default.
    See framework diagrams for a better overview.
    Test files are in the project folder.

    Choose experiment (dataset and learning settings) in experiment_config.py
    Predefined experiments vary the dataset type (dummy, PV vs no PV, regular PV vs malfunctioning PV) as well as the
    timeseries length of samples (1 day vs 1 week) and the number of samples (too little, 'just enough', sufficient to
    produce a meaningful output after training with the basic network design used, i.e. no Fscore ill defined because only
    always one class predicted in any run of cross validation; note that 1 day vs 7 days also means increasing the amount
    of data points, therefore redundant experiments (i.e. increasing the sample number even more for 1 day timeseries
    experiments was neglected to allow for a better orientation between experiments)
    The experiment also defines the network architecture (in the predefined experiments this is a simple 2 layer Elman
    RNN with 6 hidden nodes in each layer). Multiple options are available such as changing the mini batch size, early
    stopping, warm up, controlling the learning rate...

    Metrics: Deep learning approach should perform better than linear classifier (which just guesses between 0 and 1 class)
             meaning that a higher Fscore should be achieved
             Experiment configs state if this goal can be fulfilled with the experiment settings

                    Task      Dataset collection  ANN design  ANN tuning  Results     Report      Presentation
      Time planned: (Hours)   15                  7.5         15          7.5         10          4
      Time spent:   (Hours)   ~20                 25          ~15         5             to be seen
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
from util import load_model, export_model, save_model, load_data, plot_samples, model_exists, choose_best, save_result

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import logging, sys
import torch
import h5py

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
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    if config.dataset_available == False:
        print(
            "Dataset %s is created from raw data" % learning_config['dataset'])
        if (1 / config.share_of_positive_samples).is_integer():

            results_folder = config.results_folder + config.raw_data_set_name + '_raw_data' + '\\'
            for dir in os.listdir(results_folder):
                if os.path.isdir(results_folder + dir):
                    combinations_already_in_dataset = []  # avoid having duplicate samples (i.e. data of terminal with malfunction at same terminal and same terminals having a PV)
                    files = os.listdir(results_folder + dir)[0:int(config.simruns)]
                    for file in files:
                        train_samples, test_samples, combinations_already_in_dataset = create_samples(
                            results_folder + dir, file, combinations_already_in_dataset,
                            len(train_set.columns) + len(test_set.columns))
                        train_set = pd.concat([train_set, train_samples], axis=1, sort=False)
                        test_set = pd.concat([test_set, test_samples], axis=1, sort=False)
            return train_set, test_set
        else:
            print(
                "Share of malfunctioning samples wrongly chosen, please choose a value that yields a real number as an inverse i.e. 0.25 or 0.5")
            return train_set, test_set


def save_dataset(df, type='train', scaler=None):
    if config.dataset_available == False:
        if config.dataset_format == 'HDF':
            from sklearn.preprocessing import MaxAbsScaler
            from util import fit_scaler, preprocessing

            path = config.results_folder + learning_config['dataset'] + '\\' + type + '\\'
            if not os.path.isdir(path):
                os.makedirs(path)

            data_raw = df[:-1].astype(np.float32)
            label = df.iloc[-1].copy()[:].astype(int)

            if type == 'train':
                scaler = fit_scaler(data_raw)
            data_preprocessed = preprocessing(data_raw, scaler).transpose()

            with h5py.File(path + learning_config['dataset'] + '_' + type + '.hdf5', 'w') as hdf:
                if int(len(data_raw.columns) / len(label)) > 1:
                    dset_data = hdf.create_dataset('x_raw_' + type, data=data_raw, shape=(
                        len(data_raw.columns), len(data_raw), int(len(data_raw.columns) / len(label))),
                                                   compression='gzip',
                                                   chunks=True)
                    dset_data_pre = hdf.create_dataset('x_' + type, data=data_preprocessed, shape=(
                        len(data_preprocessed.columns), len(data_preprocessed),
                        int(len(data_preprocessed.columns) / len(label))), compression='gzip', chunks=True)
                else:
                    dset_data = hdf.create_dataset('x_raw_' + type, data=data_raw,
                                                   shape=(len(data_raw.columns), len(data_raw)), compression='gzip',
                                                   chunks=True)
                    dset_data_pre = hdf.create_dataset('x_' + type, data=data_preprocessed,
                                                       shape=(len(data_preprocessed.columns), len(data_preprocessed)),
                                                       compression='gzip', chunks=True)

                dset_label = hdf.create_dataset('y_' + type, data=label, shape=(len(label), 1), compression='gzip',
                                                chunks=True)
                hdf.close()
                return scaler
        else:
            df.to_csv(config.results_folder + learning_config['dataset'] + '.csv', header=True, sep=';', decimal='.',
                      float_format='%.' + '%sf' % config.float_decimal)
    print(
        "Dataset %s saved" % learning_config['dataset'])
    return 0


def cross_val(X, y, model):
    kf = KFold(n_splits=learning_config['k folds'])
    best_clfs = []
    scores = []

    for train_index, test_index in kf.split(X):
        print('Split #%d' % (len(scores) + 1))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = list(np.array(y)[train_index]), list(np.array(y)[test_index])

        X_train, X_test = model.preprocess(X_train, X_test)

        clfs, losses, lrs = model.fit(X_train, y_train, X_test, y_test,
                                      early_stopping=learning_config['early stopping'],
                                      control_lr=learning_config['LR adjustment'])
        best_model = choose_best(clfs)
        best_clfs.append(best_model)
        model.state_dict = best_model[0]
        y_pred, outputs = model.predict(X_test)
        scores.append(model.score(y_test, y_pred) + [best_model[1]])

    very_best_model = choose_best(best_clfs)
    model.state_dict = very_best_model[0]

    scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores],
                   'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores],
                   'Lowest validation loss': [i[2] for i in scores]}

    return model, scores_dict


def baseline(X, y):
    clf_baseline = SGDClassifier()
    scores = cross_validate(clf_baseline, X, y, scoring=learning_config["metrics"], cv=10, n_jobs=1)
    print("########## Linear Baseline: 10-fold Cross-validation ##########")
    for metric in learning_config["cross_val_metrics"]:
        print("%s: %0.2f (+/- %0.2f)" % (metric, scores[metric].mean(), scores[metric].std() * 2))

    return


def init():
    level = 'INFO'
    logger = logging.getLogger('main')
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device}.")

    return logger, device


if __name__ == '__main__':  # see config file for settings

    generate_raw_data()
    if config.dataset_available == False:
        train_set, test_set = create_dataset()
        scaler = save_dataset(train_set, 'train')
        save_dataset(test_set, 'test', scaler=scaler)

    print("\n########## Configuration ##########")
    for key, value in learning_config.items():
        print(key, ' : ', value)
    print("number of samples : %d" % config.number_of_samples)

    logger, device = init()

    # Load data
    logger.info("Loading Data ...")
    if learning_config["mode"] == 'train':
        train_loader = load_data('train')
    test_loader = load_data('test')
    logger.info(f"Loaded data.")

    # dataset, X, y = load_dataset()
    if learning_config["plot samples"] and learning_config["mode"] == 'train':
        for i, (X, y, X_raw) in enumerate(train_loader):
            plot_samples(X_raw, y, X)
            break

    # if learning_config['baseline']:
    # baseline(X, y)

    print('X data with zero mean per sample and scaled between -1 and 1 based on training samples used')

    path = os.path.join(config.models_folder, learning_config['classifier'])
    model, epoch, loss = load_model(learning_config)

    if not learning_config["cross_validation"]:

        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=learning_config['train test split'])
        # X_train, X_test = model.preprocess(X_train, X_test)

        print("\n########## Training ##########")
        if learning_config["do grid search"]:
            logger.info("Grid search on hyperparameter {}".format(learning_config["grid search"][0]))
            runs = len(learning_config["grid search"][1])
        else:
            runs = 1
        for i in range(runs):
            if learning_config["mode"] == 'train':
                logger.info("Training classifier ..")
                if learning_config["do grid search"]: logger.info(
                    "Value of {}: {}".format(learning_config["grid search"][0], learning_config["grid search"][1][i]))

                clfs, losses, lrs = model.fit(train_loader, test_loader,
                                              early_stopping=learning_config['early stopping'],
                                              control_lr=learning_config['LR adjustment'], prev_epoch=epoch,
                                              prev_loss=loss, grid_search_parameter = learning_config["grid search"][1][i])

                logger.info("Training finished!")
                logger.info('Finished Training')
                plotting.plot_2D([losses, [i[1] for i in clfs]], labels=['Training loss', 'Validation loss'],
                                 title='Losses after each epoch', x_label='Epoch',
                                 y_label='Loss')  # plot training loss for each epoch
                plotting.plot_2D(lrs, labels='learning rate', title='Learning rate for each epoch', x_label='Epoch',
                                 y_label='Learning rate')
                clf, epoch = choose_best(clfs)
                model.state_dict = clf[0]  # pick weights of best model found

            y_pred, outputs, y_test = model.predict(test_loader=test_loader)
            if learning_config["mode"] == 'eval':
                clf = model
                score = model.score(y_test, y_pred)
                print("\n########## Metrics ##########")
                print(
                    "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}".format(score[0],
                                                                                     score[1][
                                                                                         0],
                                                                                     score[1][
                                                                                         1],
                                                                                     score[1][
                                                                                         2], ))
            else:
                score = model.score(y_test, y_pred) + [clf[1]]
                print("\n########## Metrics ##########")
                print(
                    "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\nLowest validation loss: {4}".format(score[0],
                                                                                                                  score[1][
                                                                                                                      0],
                                                                                                                  score[1][
                                                                                                                      1],
                                                                                                                  score[1][
                                                                                                                      2],
                                                                                                                  score[2]))
            if learning_config["save_model"] and learning_config["mode"] == 'train':
                save_model(model, epoch, clf[1], i)

            if learning_config["save_result"]:
                save_result(score, i)

            if learning_config["export_model"]:
                export_model(model, learning_config, i)

        plotting.plot_grid_search()

    if learning_config["cross_validation"]:
        print("\n########## k-fold Cross-validation ##########")
        model, scores = cross_val(X, y, model)
        print("########## Metrics ##########")
        for score in scores:
            print("%s: %0.2f (+/- %0.2f)" % (score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

        if learning_config["save_model"] and learning_config["mode"] == 'train':
            save_model(model, epoch, clf[1], learning_config)

        if learning_config["save_result"]:
            save_result(scores, learning_config)

        if learning_config["export_model"]:
            export_model(model, learning_config)
