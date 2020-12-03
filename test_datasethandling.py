import os
import pandas as pd

from main import load_dataset
import main
import config

def test_load_dataset():
    '''
    Tests if malfunctions_in_LV_grid_dataset is correctly imported
    '''

    dataset, X, y = load_dataset(dataset='test')

    assert X.size == 1405400 or X.size == 2108100
    assert len(X) == 40 or len(X) == 60
    assert len(y) == 40 or len(y) == 60
    assert sum(y) == 10 or sum(y) == 15

