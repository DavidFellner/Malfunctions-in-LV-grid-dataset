import os

import Dataset
import detection_method_settings
from HDF5Dataset import HDF5Dataset
from Dataset import Deep_learning_dataset, Raw_Dataset, PCA_Dataset, Combined_Dataset, Complete_Dataset
import plotting
import torch
from torch.utils import data
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import importlib

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

if config.deeplearning:
    from algorithms.RNN import RNN
    from algorithms.LSTM import LSTM
    from algorithms.GRU import GRU
    from algorithms.Transformer import Transformer
    from algorithms.RTransformer import RT


def model_exists(full_path):
    if learning_config["do grid search"] and learning_config["mode"] == "train":
        return False
    else:
        return os.path.exists(os.path.join(full_path, learning_config["dataset"] + "_" + learning_config["type"] + "_" + 'model.pth'))


def load_model(learning_config, run):
    path = os.path.join(config.models_folder, learning_config['classifier'])
    if learning_config['classifier'] == 'RNN':
        load_saved = model_exists(path)
        model = RNN(learning_config['RNN model settings'][0], learning_config['RNN model settings'][1],
                    learning_config['RNN model settings'][2], learning_config['RNN model settings'][3])
    elif learning_config['classifier'] == 'LSTM':
        load_saved = model_exists(path)
        model = LSTM(learning_config['LSTM model settings'][0], learning_config['LSTM model settings'][1],
                     learning_config['LSTM model settings'][2], learning_config['LSTM model settings'][3])
    elif learning_config['classifier'] == 'GRU':
        load_saved = model_exists(path)
        model = GRU(learning_config['GRU model settings'][0], learning_config['GRU model settings'][1],
                    learning_config['GRU model settings'][2], learning_config['GRU model settings'][3])
    elif learning_config['classifier'] == 'Transformer':
        load_saved = model_exists(path)
        model = Transformer(learning_config['Transformer model settings'][0],
                            learning_config['Transformer model settings'][1],
                            learning_config['Transformer model settings'][2],
                            learning_config['Transformer model settings'][3],
                            learning_config['Transformer model settings'][4],
                            learning_config['Transformer model settings'][5])
    elif learning_config['classifier'] == 'RTransformer':
        load_saved = model_exists(path)
        if learning_config["do hyperparameter sensitivity analysis"]:
            if learning_config["hyperparameter tuning"][0] == 'n_layers':
                model = RT(learning_config['R-Transformer model settings'][0],
                           learning_config['R-Transformer model settings'][1],
                           learning_config['R-Transformer model settings'][2],
                           learning_config['R-Transformer model settings'][3],
                           learning_config['R-Transformer model settings'][4],
                           learning_config['R-Transformer model settings'][5],
                           learning_config["hyperparameter tuning"][1][run],
                           learning_config['R-Transformer model settings'][7],
                           learning_config['R-Transformer model settings'][8],
                           learning_config['R-Transformer model settings'][9])
            if learning_config["hyperparameter tuning"][0] == 'key_size':
                model = RT(learning_config['R-Transformer model settings'][0],
                           learning_config['R-Transformer model settings'][1],
                           learning_config['R-Transformer model settings'][2],
                           learning_config['R-Transformer model settings'][3],
                           learning_config['R-Transformer model settings'][4],
                           learning_config["hyperparameter tuning"][1][run],
                           learning_config['R-Transformer model settings'][6],
                           learning_config['R-Transformer model settings'][7],
                           learning_config['R-Transformer model settings'][8],
                           learning_config['R-Transformer model settings'][9])
            if learning_config["hyperparameter tuning"][0] == 'n_heads':
                model = RT(learning_config['R-Transformer model settings'][0],
                           learning_config['R-Transformer model settings'][1],
                           learning_config['R-Transformer model settings'][2],
                           learning_config["hyperparameter tuning"][1][run],
                           learning_config['R-Transformer model settings'][4],
                           learning_config['R-Transformer model settings'][5],
                           learning_config['R-Transformer model settings'][6],
                           learning_config["hyperparameter tuning"][1][run],
                           learning_config['R-Transformer model settings'][8],
                           learning_config['R-Transformer model settings'][9])
            if learning_config["hyperparameter tuning"][0] == 'heads':
                model = RT(learning_config['R-Transformer model settings'][0],
                           learning_config['R-Transformer model settings'][1],
                           learning_config['R-Transformer model settings'][2],
                           learning_config["hyperparameter tuning"][1][run],
                           learning_config['R-Transformer model settings'][4],
                           learning_config['R-Transformer model settings'][5],
                           learning_config['R-Transformer model settings'][6],
                           learning_config['R-Transformer model settings'][7],
                           learning_config['R-Transformer model settings'][8],
                           learning_config['R-Transformer model settings'][9])
            if learning_config["hyperparameter tuning"][0] == 'RNN_heads':
                model = RT(learning_config['R-Transformer model settings'][0],
                           learning_config['R-Transformer model settings'][1],
                           learning_config['R-Transformer model settings'][2],
                           learning_config['R-Transformer model settings'][3],
                           learning_config['R-Transformer model settings'][4],
                           learning_config['R-Transformer model settings'][5],
                           learning_config['R-Transformer model settings'][6],
                           learning_config["hyperparameter tuning"][1][run],
                           learning_config['R-Transformer model settings'][8],
                           learning_config['R-Transformer model settings'][9])
            else:
                print('paramerter not defined for hyper parameter optimization!')
        else:
            model = RT(learning_config['R-Transformer model settings'][0],
                       learning_config['R-Transformer model settings'][1],
                       learning_config['R-Transformer model settings'][2],
                       learning_config['R-Transformer model settings'][3],
                       learning_config['R-Transformer model settings'][4],
                       learning_config['R-Transformer model settings'][5],
                       learning_config['R-Transformer model settings'][6],
                       learning_config['R-Transformer model settings'][7],
                       learning_config['R-Transformer model settings'][8],
                       learning_config['R-Transformer model settings'][9])
    else:
        print('Invalid model type entered!')
        return None

    device = model.choose_device()

    if load_saved:
        print('Loading model ..')

        try:
            checkpoint = torch.load(os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_" + 'model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            model.to(device)
            print('Model successfully loaded!')
            return model, epoch, loss
        except RuntimeError:
            print('Improper model loaded (different architecture), fresh model used')
            pass
    else:
        print('No saved Model found. Fresh model used')

    model.to(device)

    return model, None, None


def export_model(model, learning_config, grid_search_run, hyper_para):
    dummy_input = torch.randn(1, 672, 1)
    out = model(dummy_input)
    input_names = ["input"]  # + ["learned_%d" % i for i in range(3)]
    output_names = ["output"]
    name = learning_config['dataset'] + '.onnx'

    model.eval()
    torch.onnx.export(torch.jit.trace_module(model, {'forward': dummy_input}), dummy_input, name, example_outputs=out,
                      export_params=True, verbose=True,
                      input_names=input_names, output_names=output_names)


def save_model(model, epoch, loss, grid_search_run, hyp_search_run):
    path = os.path.join(config.models_folder, learning_config['classifier'])

    if not os.path.exists(path):
        os.makedirs(path)

    base_name = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"])

    if learning_config["do grid search"]:
        grid_search = "_gridsearch_on_" + learning_config["grid search"][0] + "_value_" + str(learning_config["grid search"][1][grid_search_run])
    else: grid_search = ''
    if learning_config["do hyperparameter sensitivity analysis"]:
        hyper_para = "_hyper_parameter_analysis_on_" + learning_config["hyperparameter tuning"][0] + "_value_" + str(learning_config["hyperparameter tuning"][1][hyp_search_run])
    else: hyper_para = ''

    name = base_name + grid_search + hyper_para + '_model.pth'

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, name)
    except TypeError:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict,
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, name)

    # torch.save(model.state_dict(), path + ".model")
    # torch.save(model.optimizer.state_dict(), path + ".optimizer")


def save_result(score, grid_search_run, hyp_search_run, epoch):
    path = os.path.join(config.models_folder, learning_config['classifier'])

    if not os.path.exists(path):
        os.makedirs(path)

    base_name = os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"])

    os.path.join(path, learning_config["dataset"] + "_" + learning_config["type"] + "_gridsearch_on_" +
                 learning_config["grid search"][0]
                 + "_" + 'result.txt')

    if learning_config["do grid search"]:
        grid_search = "_gridsearch_on_" + learning_config["grid search"][0]
    else:
        grid_search = ''
    if learning_config["do hyperparameter sensitivity analysis"]:
        hyper_para = "_hyper_parameter_analysis_on_" + learning_config["hyperparameter tuning"][0]
    else:
        hyper_para = ''
    if learning_config["training time sweep"]:
        training_time_sweep = "_training_time_sweep"
    else:
        training_time_sweep = ''

    name = base_name + grid_search + hyper_para + training_time_sweep + '_result.txt'

    f = open(name, "a")
    if grid_search_run == 0 or hyp_search_run == 0 or epoch == 0 :
        f.write("Configuration:\n")
        for key, value in learning_config.items():
            f.write("\n" + key + ' : ' + str(value))
    if learning_config["do grid search"]:
        f.write("\nGrid search on: " + learning_config["grid search"][0] +
                "; value:" + str(learning_config["grid search"][1][grid_search_run]))
    if learning_config["do hyperparameter sensitivity analysis"]:
        f.write("\nHyperparameter sensitivity analysis on: " + learning_config["hyperparameter tuning"][0] +
                "; value:" + str(learning_config["hyperparameter tuning"][1][hyp_search_run]))
    if learning_config["training time sweep"]:
        f.write("\nTraining time seep; epoch:" + str(epoch + 1))
    f.write("\n########## Metrics ##########")
    f.write(
        "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(score[0],
                                                                         score[1][
                                                                             0],
                                                                         score[1][
                                                                             1],
                                                                         score[1][
                                                                             2], ))

    f.close()


def plot_samples(X, y, X_pre=None):
    if type(X) == torch.Tensor:
        X = np.array(X)
        y = np.array(y)

    X_zeromean = np.array([x - x.mean() for x in X])  # deduct it's own mean from every sample
    if X_pre is None:
        X_train, X_test, y_train, y_test = train_test_split(X_zeromean, y, random_state=0, test_size=100)
        maxabs_scaler = MaxAbsScaler().fit(X_train)  # fit scaler as to scale training data between -1 and 1

    # samples = [y.index(0), y.index(1)]
    samples = [random.sample([i for i, x in enumerate(y) if x == 0], 1)[0],
               random.sample([i for i, x in enumerate(y) if x == 1], 1)[0]]
    print("Samples shown: #{0} of class 0; #{1} of class 1".format(samples[0], samples[1]))
    plotting.plot_sample(X[samples], label=[y[i] for i in samples], title='Raw samples')
    plotting.plot_sample(X_zeromean[samples], label=[y[i] for i in samples], title='Zeromean samples')
    if X_pre is None:
        X_maxabs = maxabs_scaler.transform(X_zeromean[samples])
        plotting.plot_sample(X_maxabs, label=[y[i] for i in samples], title='Samples scaled to -1 to 1')
    else:
        plotting.plot_sample(np.array(X_pre[samples]), label=[y[i] for i in samples], title='Samples scaled to -1 to 1')


def load_data(type):
    path = os.path.join(config.datasets_folder, learning_config['dataset'], type)
    file = learning_config['dataset'] + '_' + learning_config['type'] + '_' + type + '.hdf5'
    # dataset = HDF5Dataset(path, recursive=True, load_data=False,
    # data_cache_size=4, transform=None)

    dataset = HDF5Dataset(os.path.join(path, file), type)

    # pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'train' + '.h5', key = 'train/data', mode='a')
    # pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'train' + '.h5', key = 'train/label', mode='a')

    if type == 'test':
        # data_info = dataset.get_data_infos('data')
        # loader_params = {'batch_size': data_info[0]['shape'][0], 'shuffle': True, 'num_workers': 1}       #load all test data at once?

        # loader_params = {'batch_size': dataset.__len__(), 'shuffle': False, 'num_workers': 1}       #load all test data at once?
        if dataset.__len__() < 1000:
            loader_params = {'batch_size': dataset.__len__(), 'shuffle': False,
                             'num_workers': 1}  # load all test data at once
        else:
            loader_params = {'batch_size': 100, 'shuffle': False, 'num_workers': 1}  # load 100 test samples at once

    else:
        loader_params = {'batch_size': learning_config['mini batch size'], 'shuffle': True, 'num_workers': 1}
    data_loader = data.DataLoader(dataset, **loader_params)

    return data_loader

def load_dataset():

    X = None
    Y = None

    for type in ['train', 'test']:
        path = os.path.join(config.datasets_folder, learning_config['dataset'], type)
        file = learning_config['dataset'] + '_' + learning_config['type'] + '_' + type + '.hdf5'


        dataset = HDF5Dataset(os.path.join(path, file), type)

        if X is None and Y is None:
            X = dataset[:][0]
            y = dataset[:][1]
        else:
            np.concatenate((X, dataset[:][0]), axis=0)
            np.concatenate((y, dataset[:][1]), axis=0)

    return X, y

def detection_method_dl(module, X, y):

    classifier_combos = detection_method_settings.Classifier_Combos().classifier_combos['general_combined_dataset']
    data = Dataset.Reduced_Combined_Dataset(X,y)

    for classifiers in classifier_combos:
        scores = cross_val_ml_dl(module, data, classifiers_and_parameters=classifiers)
        print(f"\n########## Metrics for traditional ML classifier on deep learnign data using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes correct and wrong ##########")
        for score in scores:
            print("%s: %0.2f (+/- %0.2f)" % (
                score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))
            print('Note: A score of 0.5 means guessing in this case!')


def cross_val_ml_dl(module, data, classifiers_and_parameters=None):

    if classifiers_and_parameters is None:
        classifiers_and_parameters = {'SVM': {'poly': [8]}, 'NuSVM': {'linear': [9], 'poly': [11], 'rbf': [2]},
                                      'kNN': {3: [18, 'uniform']}}
    X = data.X
    y = data.y
    kf = KFold(n_splits=3, shuffle=True)

    scores = []

    for train_index, test_index in kf.split(X, y):
        # print('Split #%d' % (len(scores) + 1))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        y_pred, y_test = module.assembly_learner_combined_dataset(module, [X_train, X_test, y_train.ravel(), y_test.ravel()], classifiers_and_parameters, cross_val=True)
        scores.append(module.scoring(module, y_test, y_pred))

    scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores],
                   'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores]}

    return scores_dict

"""def load_dataset(dataset=None):
    '''
        deprecated
    '''

    if not dataset:
        if learning_config['dataset'][:7] == 'PV_noPV':
            dataset = PVnoPVdataset(config.raw_data_folder + learning_config["dataset"] + '.csv')
        elif learning_config['dataset'][:31] == 'malfunctions_in_LV_grid_dataset':
            dataset = MlfctinLVdataset(config.raw_data_folder + learning_config["dataset"] + '.csv')
        else:
            dataset = Dummydataset(config.raw_data_folder + learning_config["dataset"] + '.csv')


    else:
        if learning_config['dataset'][:7] == 'PV_noPV':
            dataset = PVnoPVdataset(config.test_data_folder + 'PV_noPV.csv')
        else:
            dataset = MlfctinLVdataset(config.test_data_folder + 'malfunctions_in_LV_grid_dataset.csv')

    X = dataset.get_x()
    y = dataset.get_y()
    return dataset, X, y"""


def preprocess(X_train, X_test):
    scaler = fit_scaler(X_train)
    X_train = preprocessing(X_train, scaler)
    X_test = preprocessing(X_test, scaler)
    return X_train, X_test


def fit_scaler(X):
    X_zeromean = np.array(X - X.mean())  # deduct it's own mean from every sample
    maxabs_scaler = MaxAbsScaler().fit(X_zeromean.reshape(X_zeromean.shape[1], X_zeromean.shape[
        0]))  # fit scaler as to scale training data between -1 and 1
    return maxabs_scaler


def preprocessing(X, scaler):
    X_zeromean = np.array(X - X.mean())
    X = scaler.transform(X_zeromean.reshape(X_zeromean.shape[1], X_zeromean.shape[0]))
    return pd.DataFrame(data=X)


def choose_best(models_and_losses):
    index_best = [i[1] for i in models_and_losses].index(min([i[1] for i in models_and_losses]))
    epoch = index_best + 1
    return models_and_losses[index_best], epoch


def get_weights_copy(model):
    weights_path = 'weights_temp.pt'
    torch.save(model.state_dict, weights_path)
    return torch.load(weights_path)


def create_dataset(type='raw', data=None, variables=None, name=None, classes=None, bay='F2', Setup='A', labelling='classification',trafo_point=None):
    '''
    creates either  a deep learning dataset or the specified dataset for detection methods
    :param type: which type of detection methods dataset is needed? Options: raw, pca, combined
    :return: dataset object of desired dataset type
    '''

    if config.deeplearning:
        dataset = Deep_learning_dataset(config=config)
    elif config.detection_methods:
        if type=='raw':
            dataset = Raw_Dataset(data, name, classes=classes, bay=bay, Setup=Setup, labelling=labelling)
        if type=='pca':
            dataset =PCA_Dataset(data, name, classes=classes, bay=bay, Setup=Setup, labelling=labelling)
        if type=='combined':
            dataset = Combined_Dataset(data, variables, name, classes=classes, bay=bay, setup=Setup, labelling=labelling)
    elif config.disaggregation:
        if type=='complete':
            dataset = Complete_Dataset(data, variables, name, trafo_point=trafo_point, classes=classes, bay=bay, setup=Setup,
                                       labelling=labelling)
    else:
        print('Dataset in config: either deeplearning, detection_methods and/or disaggregation')
        dataset = None
        return dataset

    dataset.create_dataset()
    dataset.dataset_info()

    if config.deeplearning:
        scaler = dataset.save_dataset(dataset.train_set, 'train')
        dataset.save_dataset(dataset.test_set, 'test', scaler=scaler)

    return dataset

'''def load_test_data(config):

    pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'test' + '.h5', key = 'test/data', mode='a')
    pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'test' + '.h5', key = 'test/label', mode='a')
    test_loader, is_clean = load_gtsrb(config, test=True)
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.numpy(), y_test.numpy()
    return x_test, y_test, is_clean'''
