import os
import sys
import importlib
import numpy as np
from sklearn.model_selection import train_test_split
sys._called_from_test = True
test_folder = os.getcwd() + '\\test\\'

spec = importlib.util.spec_from_file_location('malfunctions_in_LV_grid_dataset_test', test_folder + 'malfunctions_in_LV_grid_dataset_test.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

experiment = 'malfunctions_in_LV_grid_dataset_test'
f = open(test_folder + "experiment.txt", "w")
f.write(experiment)
f.close()

from main import load_dataset
from RNN import RNN

def test_load_dataset():
    '''
    Tests if malfunctions_in_LV_grid_dataset is correctly imported
    '''

    dataset, X, y = load_dataset()

    assert len(X[0]) == 96
    assert len(X) == 1998
    assert len(y) == 1998
    assert sum(y) == 999

def test_preprocessing():
    '''
    Tests if preprocessing is done correctly (zero meaning every sample by itself and max abs scaling fitted on the training set)
    '''

    X_train = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]])
    X_test = np.array([[2,3,4,5], [0,6,7,8]])
    model = RNN(config.learning_config['RNN model settings'][0],  config.learning_config['RNN model settings'][1],config.learning_config['RNN model settings'][2], config.learning_config['RNN model settings'][3])

    X_train, X_test = model.preprocess(X_train, X_test)

    assert all([i.mean() == 0 for i in X_train])
    assert max([i.max() for i in X_train]) == 1
    assert max([i.min() for i in X_train]) == -1
    assert max([i.max() for i in X_test]) > 1
    assert min([i.min() for i in X_test]) < -1

def test_training():
    dataset, X, y = load_dataset()
    model = RNN(config.learning_config['RNN model settings'][0], config.learning_config['RNN model settings'][1],
                config.learning_config['RNN model settings'][2], config.learning_config['RNN model settings'][3])


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=config.learning_config['train test split'])
    X_train, X_test = model.preprocess(X_train, X_test)
    clfs, losses, lrs = model.fit(X_train, y_train, X_test, y_test, early_stopping=config.learning_config['early stopping'],
                                  control_lr=config.learning_config['LR adjustment'])

    assert type(clfs[-1][0]()) == type(model.state_dict())
    assert len(losses) == 2

def test_prediction():
    dataset, X, y = load_dataset()
    model = RNN(config.learning_config['RNN model settings'][0], config.learning_config['RNN model settings'][1],
                config.learning_config['RNN model settings'][2], config.learning_config['RNN model settings'][3])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                        test_size=config.learning_config['train test split'])
    X_train, X_test = model.preprocess(X_train, X_test)
    model.fit(X_train, y_train, X_test, y_test, early_stopping=config.learning_config['early stopping'],
              control_lr=config.learning_config['LR adjustment'])

    y_pred, outputs = model.predict(X_test)

    assert len(X_test) == len(y_pred)
    assert len(X_test[0]) == len(outputs[0])


def test_postprocessing():
    model = RNN(config.learning_config['RNN model settings'][0], config.learning_config['RNN model settings'][1],
                config.learning_config['RNN model settings'][2], config.learning_config['RNN model settings'][3])

    y_test = [1,0,1,0,1,0]
    y_pred = [0,0,0,0,0,0]

    accuracy, metrics = model.eval(y_test, y_pred)
    precision = metrics[0]
    recall = metrics[1]
    fscore = metrics[2]

    assert accuracy == 0.5
    assert precision == 0.25
    assert recall == 0.5
    assert fscore == 1/3


