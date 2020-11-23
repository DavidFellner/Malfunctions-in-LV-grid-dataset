import os
import pandas as pd

from create_instances import create_samples
from create_instances import extract_data
import main
import config

def test_create_dataset():
    '''
    Tests if a dataset of correct size and with teh correct number of postive targets is created
    '''

    df = main.create_dataset()
    num_of_positive_samples = (df.iloc[-1] == 1).value_counts()[True]

    assert num_of_positive_samples == config.number_of_samples * config.percentage_of_malfunction_samples
    assert len(df.columns) == config.number_of_samples

def test_create_samples():
    '''
    Tests if the correct amount of samples is extracted per file and if duplicate sample listing works
    '''

    dir = os.getcwd() + '\\test\\output\\'
    file = 'result_run#0.csv'
    terminals_already_in_dataset = []
    samples, terminals_already_in_dataset = create_samples(dir, file, terminals_already_in_dataset,
                                                           0)
    assert len(samples.columns) == 1 / config.percentage_of_malfunction_samples
    assert len(terminals_already_in_dataset) > 0

def test_extract_data():
    '''
    Tests if the correct number of samples of each label are extracted from a results file
    '''

    df = pd.read_csv(os.getcwd() + '\\test\\output\\' + 'result_run#0.csv', header=[0, 1, 2], sep=';')
    metainfo = df[('metainfo', 'in the first', 'few indices')]
    number_of_positive_samples = len([i for i in metainfo.iloc[3].split("'") if 'Bus' in i])

    df_treated, terminals_already_in_dataset = extract_data(df, [], 0)
    number_of_positive_samples_extracted = (df_treated.iloc[-1] == 1).value_counts()[True]
    assert number_of_positive_samples_extracted == number_of_positive_samples
    assert len(df_treated.columns) == (1 / config.percentage_of_malfunction_samples)

