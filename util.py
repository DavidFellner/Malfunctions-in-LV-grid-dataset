import os
from RNN import RNN
from LSTM import LSTM
from GRU import GRU
from Transformer import Transformer
from RTransformer import RT
from HDF5Dataset import HDF5Dataset
import plotting
import torch
from torch.utils import data
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd


import importlib

from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config


def model_exists(full_path):
    return os.path.exists(full_path + ".model")

def load_model(learning_config):
    path = config.models_folder + learning_config['classifier']
    if learning_config['classifier'] == 'RNN':
        load_saved = model_exists(path)
        model = RNN(learning_config['RNN model settings'][0],  learning_config['RNN model settings'][1],
                    learning_config['RNN model settings'][2], learning_config['RNN model settings'][3])
    elif learning_config['classifier'] == 'LSTM':
        load_saved = model_exists(path)
        model = LSTM(learning_config['LSTM model settings'][0],  learning_config['LSTM model settings'][1],
                     learning_config['LSTM model settings'][2], learning_config['LSTM model settings'][3])
    elif learning_config['classifier'] == 'GRU':
        load_saved = model_exists(path)
        model = GRU(learning_config['GRU model settings'][0],  learning_config['GRU model settings'][1],
                    learning_config['GRU model settings'][2], learning_config['GRU model settings'][3])
    elif learning_config['classifier'] == 'Transformer':
        load_saved = model_exists(path)
        model = Transformer(learning_config['Transformer model settings'][0],  learning_config['Transformer model settings'][1], learning_config['Transformer model settings'][2], learning_config['Transformer model settings'][3], learning_config['Transformer model settings'][4], learning_config['Transformer model settings'][5])
    elif learning_config['classifier'] == 'RTransformer':
        load_saved = model_exists(path)
        model = RT(learning_config['R-Transformer model settings'][0],  learning_config['R-Transformer model settings'][1], learning_config['R-Transformer model settings'][2], learning_config['R-Transformer model settings'][3], learning_config['R-Transformer model settings'][4], learning_config['R-Transformer model settings'][5], learning_config['R-Transformer model settings'][6], learning_config['R-Transformer model settings'][7], learning_config['R-Transformer model settings'][8], learning_config['R-Transformer model settings'][9])
    else:
        print('Invalid model type entered!')
        return None

    device = model.choose_device()

    if load_saved:
        print('Loading model ..')

        checkpoint = torch.load(path + '\\model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.to(device)
        model.optimizer.to(device)
        return model, model.optimizer

    model.to(device)

    return model

def export_model(model, learning_config):
    dummy_input = torch.randn(1, 672, 1)
    out = model(dummy_input)
    input_names = ["input"]  # + ["learned_%d" % i for i in range(3)]
    output_names = ["output"]
    name = learning_config['dataset'] + '.onnx'

    model.eval()
    torch.onnx.export(torch.jit.trace_module(model, {'forward': dummy_input}), dummy_input, name, example_outputs=out, export_params=True, verbose=True,
                      input_names=input_names, output_names=output_names)


def save_model(model, epoch, loss):
    path = config.models_folder + learning_config['classifier']

    if not os.path.exists(config.models_folder + learning_config['classifier']):
        os.makedirs(config.models_folder + learning_config['classifier']
                    )

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, path + '\\model.pth')
    except TypeError:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict,
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, path + '\\model.pth')

    #torch.save(model.state_dict(), path + ".model")
    #torch.save(model.optimizer.state_dict(), path + ".optimizer")

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

    path = config.results_folder + learning_config['dataset'] + '\\' + type + '\\'
    file  = learning_config['dataset'] + '_' + type + '.hdf5'
    #dataset = HDF5Dataset(path, recursive=True, load_data=False,
                          #data_cache_size=4, transform=None)

    dataset = HDF5Dataset(path + file, type)

    #pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'train' + '.h5', key = 'train/data', mode='a')
    #pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'train' + '.h5', key = 'train/label', mode='a')

    if type == 'test':
        #data_info = dataset.get_data_infos('data')
        #loader_params = {'batch_size': data_info[0]['shape'][0], 'shuffle': True, 'num_workers': 1}       #load all test data at once?

        #loader_params = {'batch_size': dataset.__len__(), 'shuffle': False, 'num_workers': 1}       #load all test data at once?
        if dataset.__len__() < 1000:
            loader_params = {'batch_size': dataset.__len__(), 'shuffle': False, 'num_workers': 1}       #load all test data at once
        else:
            loader_params = {'batch_size': 1000, 'shuffle': False, 'num_workers': 1}       #load 1000 test samples at once

    else:
        loader_params = {'batch_size': learning_config['mini batch size'], 'shuffle': True, 'num_workers': 1}
    data_loader = data.DataLoader(dataset, **loader_params)

    return data_loader

def load_dataset(dataset=None):
    '''
        deprecated
    '''

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

def preprocess(X_train, X_test):
    scaler = fit_scaler(X_train)
    X_train = preprocessing(X_train, scaler)
    X_test = preprocessing(X_test, scaler)
    return X_train, X_test

def fit_scaler(X):
    X_zeromean = np.array(X - X.mean())             # deduct it's own mean from every sample
    maxabs_scaler = MaxAbsScaler().fit(X_zeromean.reshape(X_zeromean.shape[1], X_zeromean.shape[0]))                          # fit scaler as to scale training data between -1 and 1
    return maxabs_scaler

def preprocessing(X, scaler):
    X_zeromean = np.array(X - X.mean())
    X = scaler.transform(X_zeromean.reshape(X_zeromean.shape[1], X_zeromean.shape[0]))
    return pd.DataFrame(data=X)


'''def load_test_data(config):

    pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'test' + '.h5', key = 'test/data', mode='a')
    pd.read_hdf(config.results_folder + learning_config['dataset'] + '_' + 'test' + '.h5', key = 'test/label', mode='a')
    test_loader, is_clean = load_gtsrb(config, test=True)
    x_test, y_test = next(iter(test_loader))
    x_test, y_test = x_test.numpy(), y_test.numpy()
    return x_test, y_test, is_clean'''
