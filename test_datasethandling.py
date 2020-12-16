import os
import sys
import importlib
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

    X_train = [[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]]
    X_test = [[2,3,4,5], [0,6,7,8]]

    X_train, X_test = RNN.preprocess(X_train, X_test)

    assert all([i.mean() == 0 for i in X_train])
    assert all([i.mean() == 0 for i in X_test])
    assert max([i.max() for i in X_train]) == 1
    assert max([i.min() for i in X_train]) == -1
    assert max([i.max() for i in X_train]) > 1
    assert max([i.min() for i in X_train]) < -1


def test_postprocessing():



    return

