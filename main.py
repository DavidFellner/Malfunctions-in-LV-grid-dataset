"""
Author:
    David Fellner
Description:
    Set settings for QDS and elements and save results to file to create a dataset executing a QDS. At first the grid is
    prepared and scenario settings are set.
"""

import config
from start_powerfactory import start_powerfactory
from grid_preparation import prepare_grid
from data_creation import create_data
from create_instances import create_samples
from train import train

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

    if config.dataset_available == False:
        if (1/config.share_of_malfunction_samples).is_integer():
            df = pd.DataFrame()
            for dir in os.listdir(config.results_folder):
                if os.path.isdir(config.results_folder + dir):
                    terminals_already_in_dataset = []  # avoid having duplicate samples (data of terminal with malfunction at same terminal and same terminals having a PV)
                    files = os.listdir(config.results_folder + dir)[0:int(config.simruns)]
                    for file in files:
                        samples, terminals_already_in_dataset = create_samples(config.results_folder + dir, file, terminals_already_in_dataset,
                                                                               len(df.columns))
                        df = pd.concat([df, samples], axis=1, sort=False)
            return df
        else:
            print("Share of malfunctioning samples wrongly chosen, please choose a value that yields a real number as an inverse i.e. 0.25 or 0.5")

def save_dataset(df):

    if config.dataset_available == False:
        df.to_csv(config.results_folder + config.data_set_name, header=True, sep=';', decimal='.', float_format='%.3f')

    return

if __name__ == '__main__':  #see config file for settings

    generate_raw_data()
    dataset = create_dataset()
    save_dataset(dataset)

    train()





