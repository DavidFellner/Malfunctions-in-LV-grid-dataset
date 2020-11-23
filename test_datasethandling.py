import os
import pandas as pd

from train import load_dataset
import main
import config

def test_load_dataset():
    '''
    Tests if a dataset of correct size and with teh correct number of postive targets is created
    '''

    dataset, X, y = load_dataset()

    assert X.size == 1405400
    assert y.size == 40

