"""
Author:
    David Fellner
Description:
    Set settings for QDS and elements and save results to file to create a dataset executing a QDS. At first the grid is
    prepared and scenario settings are set.
"""

import config
from config import learning_config
from start_powerfactory import start_powerfactory
from grid_preparation import prepare_grid
from data_creation import create_data
from create_instances import create_samples
from train import train
from malfunctions_in_LV_grid_dataset import MlfctinLVdataset
from RNN import RNN

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

def load_dataset():

    if learning_config['dataset'] == 'some_other_dataset':
        dataset = MlfctinLVdataset(learning_config["some_other_dataset"])
    else:
        dataset = MlfctinLVdataset(learning_config["malfunction_in_LV_grid_data"])    #default is created dataset

    X = dataset.get_x()
    y = dataset.get_y()
    return dataset, X, y

if __name__ == '__main__':  #see config file for settings

    generate_raw_data()
    dataset = create_dataset()
    save_dataset(dataset)

    print("########## Configuration ##########")
    for key, value in learning_config.items():
        print(key, ' : ', value)

    dataset, X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #clf = train(dataset, X_train, y_train)

    #or
    if learning_config['classifier'] == 'RNN':
        model = RNN(learning_config['RNN model settings'][0],  learning_config['RNN model settings'][1],
                    learning_config['RNN model settings'][2], learning_config['RNN model settings'][3])

    if not learning_config["cross_validation"]:
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        print("########## Metrics ##########")
        print(
            "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}".format(accuracy, metrics[0], metrics[1],
                                                                             metrics[2]))

    if learning_config["cross_validation"]:
        scores = cross_validate(model, X, y, scoring=learning_config["metrics"], cv=10)
        print("########## 10-fold Cross-validation ##########")
        for metric in learning_config["cross_val_metrics"]:
            print("%s: %0.2f (+/- %0.2f)" % (metric, scores[metric].mean(), scores[metric].std() * 2))






