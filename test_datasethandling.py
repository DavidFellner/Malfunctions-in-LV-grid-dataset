import os
import pandas as pd

from main import load_dataset
import main
import config

def test_load_dataset():
    '''
    Tests if malfunctions_in_LV_grid_dataset is correctly imported
    '''

    dataset, X, y = load_dataset()

    assert X.size == 1405400
    assert len(X) == 40
    assert len(y) == 40

