import os
import pandas as pd

import sys
sys._called_from_test = True

import importlib
test_folder = os.path.join(os.getcwd(), 'test')

spec = importlib.util.spec_from_file_location('dummy_test',  os.path.join(test_folder, 'dummy_test.py'))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

experiment = 'dummy_test'
f = open( os.path.join(test_folder, "experiment.txt"), "w")
f.write(experiment)
f.close()

from create_instances import create_samples
from create_instances import extract_malfunction_data
import main

def test_generate_dummy_raw_data():
    '''
    Test if correct number of result files is created as raw data for mlfct dataset
    '''

    main.generate_raw_data()

    results_folder = os.path.join(config.results_folder, config.raw_data_set_name + '_raw_data')
    file_folder = os.path.join(results_folder, '1-LV-semiurb4--0-sw')
    count = len([name for name in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder + name))])

    assert count == 1


def test_create_dummy_dataset():
    '''
    #Tests if a malfct dataset with the correct number of positive targets is created
    '''

    df = main.create_dataset()
    num_of_positive_samples = (df.iloc[-1] == 1).value_counts()[True]

    assert num_of_positive_samples == len(df.columns) / 2

def test_create_samples():
    '''
    #Tests if the correct amount of samples is extracted per file and if duplicate sample listing works
    '''

    dir = os.path.join(config.results_folder, config.raw_data_set_name + '_raw_data', '1-LV-semiurb4--0-sw')
    file = 'result_run#0.csv'
    terminals_already_in_dataset = []
    samples, terminals_already_in_dataset = create_samples(dir, file, terminals_already_in_dataset,
                                                           0)
    assert len(samples.columns) == 2000
    assert len(terminals_already_in_dataset) > 0

def test_extract_malfunction_data():
    '''
    #Tests if the correct number of samples of each label are extracted from a results file
    '''

    df = pd.read_csv(os.path.join(config.results_folder, config.raw_data_set_name + '_raw_data', '1-LV-semiurb4--0-sw', 'result_run#0.csv'), header=[0, 1, 2], sep=';')

    df_treated, terminals_already_in_dataset = extract_malfunction_data(df, [], 0)
    number_of_positive_samples_extracted = (df_treated.iloc[-1] == 1).value_counts()[True]
    assert number_of_positive_samples_extracted == 1000
    assert len(df_treated.columns) == 2000
