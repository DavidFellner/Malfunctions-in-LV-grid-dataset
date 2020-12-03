"""
Author:
    David Fellner
Description:
    Set settings for QDS and elements and save results to file to create a dataset executing a QDS. At first the grid is
    prepared and scenario settings are set.
"""

import config
from config import learning_config
import plotting
from start_powerfactory import start_powerfactory
from grid_preparation import prepare_grid
from data_creation import create_data
from create_instances import create_samples
from malfunctions_in_LV_grid_dataset import MlfctinLVdataset
from PV_noPV_dataset import PVnoPVdataset
from RNN import RNN
from Transformer import Transformer

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import skorch

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
        print(
            "Dataset %s is created from raw data" % learning_config['dataset'])
        if (1/config.share_of_positive_samples).is_integer():
            df = pd.DataFrame()
            results_folder = config.results_folder + config.data_set_name + '_raw_data' + '\\'
            for dir in os.listdir(results_folder):
                if os.path.isdir(results_folder + dir):
                    terminals_already_in_dataset = []  # avoid having duplicate samples (data of terminal with malfunction at same terminal and same terminals having a PV)
                    files = os.listdir(results_folder + dir)[0:int(config.simruns)]
                    for file in files:
                        samples, terminals_already_in_dataset = create_samples(results_folder + dir, file, terminals_already_in_dataset,
                                                                               len(df.columns))
                        df = pd.concat([df, samples], axis=1, sort=False)
            return df
        else:
            print("Share of malfunctioning samples wrongly chosen, please choose a value that yields a real number as an inverse i.e. 0.25 or 0.5")

def save_dataset(df):

    if config.dataset_available == False:
        df.to_csv(config.results_folder + config.data_set_name + '.csv', header=True, sep=';', decimal='.', float_format='%.' + '%sf' % config.float_decimal)
        print(
            "Dataset %s saved" % learning_config['dataset'])

    return

def load_dataset(dataset=None):

    if not dataset:
        if learning_config['dataset'] == 'PV_noPV':
            dataset = PVnoPVdataset(learning_config["PV_noPV"])
        else:
            dataset = MlfctinLVdataset(learning_config["malfunction_in_LV_grid_data"])

    else:
        if learning_config['dataset'] == 'PV_noPV':
            dataset = PVnoPVdataset(learning_config["test_data_set2"])
        else:
            dataset = MlfctinLVdataset(learning_config["test_data_set1"])

    X = dataset.get_x()
    y = dataset.get_y()
    return dataset, X, y

if __name__ == '__main__':  #see config file for settings

    generate_raw_data()
    dataset = create_dataset()
    save_dataset(dataset)

    print("\n########## Configuration ##########")
    for key, value in learning_config.items():
        print(key, ' : ', value)

    dataset, X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    if learning_config['classifier'] == 'RNN':
        model = RNN(learning_config['RNN model settings'][0],  learning_config['RNN model settings'][1],
                    learning_config['RNN model settings'][2], learning_config['RNN model settings'][3])

    if not learning_config["cross_validation"]:
        clf = model.fit(X_train, y_train)
        scaler = model.get_scaler()
        y_pred = clf.predict(scaler.transform(X_test))
        metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        print("########## Metrics ##########")
        print(
            "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}".format(accuracy, metrics[0], metrics[1],
                                                                             metrics[2]))

    if learning_config["cross_validation"]:
        scores = cross_validate(model, X, y, scoring=learning_config["metrics"], cv=10, n_jobs=1)
        print("########## 10-fold Cross-validation ##########")
        for metric in learning_config["cross_val_metrics"]:
            print("%s: %0.2f (+/- %0.2f)" % (metric, scores[metric].mean(), scores[metric].std() * 2))






