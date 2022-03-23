import pandas as pd
import numpy as np
import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

import importlib
from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

from dataset_creation.create_instances import create_samples

class Deep_learning_dataset:

    def __init__(self, config):

        self.config = config
        self.learning_config = self.config.learning_config

        train_set, test_set = self.create_dataset()

        scaler = self.save_dataset(train_set, 'train')
        self.save_dataset(test_set, 'test', scaler=scaler)

    def dataset_info(self):
        print(
            f'Dataset containing {len(self.train_set.columns) + len(self.test_set.columns)} samples, {sum(self.train_set.loc["label"]) + sum(self.test_set.loc["label"])} of which positive, created')
        print(f'Test set: {len(self.test_set.columns)} samples, of which {sum(self.test_set.loc["label"])} positive')
        print(f'Training set: {len(self.train_set.columns)} samples, of which {sum(self.train_set.loc["label"])} positive')

    def create_dataset(self):
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        if self.config.dataset_available == False:
            print(
                f"Dataset {self.learning_config['dataset']} with a {self.config.type} malfunction is created from raw data")
            if (1 / self.config.share_of_positive_samples).is_integer():

                penetrations = ''
                for key, value in self.config.percentage.items():
                    if value > 0:
                        penetrations += '_' + key + '(' + str(value) + ')'

                results_folder = os.path.join(self.config.raw_data_folder,
                                              (self.config.raw_data_set_name + penetrations + '_raw_data'))
                print(f'Creating dataset using simulation data with the following penetrations: {penetrations}')
                # results_folder = config.raw_data_folder + config.raw_data_set_name + '_raw_data' + '\\'

                for dir in os.listdir(results_folder):
                    if os.path.isdir(os.path.join(results_folder, dir)):
                        combinations_already_in_dataset = []  # avoid having duplicate samples (i.e. data of terminal with malfunction at same terminal and same terminals having a PV)
                        files = os.listdir(os.path.join(results_folder, dir))[0:int(self.config.simruns)]
                        for file in files:
                            try:
                                postive_test_samples = int(sum(self.test_set.loc['label']))
                            except KeyError:
                                postive_test_samples = 0
                            train_samples, test_samples, combinations_already_in_dataset = create_samples(
                                os.path.join(results_folder, dir), file, combinations_already_in_dataset,
                                len(self.train_set.columns) + len(self.test_set.columns), postive_test_samples,
                                len(self.test_set.columns))
                            self.train_set = pd.concat([self.train_set, train_samples], axis=1, sort=False)
                            self.test_set = pd.concat([self.test_set, test_samples], axis=1, sort=False)

                return self.train_set, self.test_set
            else:
                print(
                    "Share of malfunctioning samples wrongly chosen, please choose a value that yields a real number as an inverse i.e. 0.25 or 0.5")
                return self.train_set, self.test_set

    def save_dataset(self, df, type='train', scaler=None):
        if self.config.dataset_available == False:
            if self.config.dataset_format == 'HDF':
                from sklearn.preprocessing import MaxAbsScaler
                from util import fit_scaler, preprocessing

                path = os.path.join(self.config.datasets_folder, self.learning_config['dataset'], type)
                if not os.path.isdir(path):
                    os.makedirs(path)

                data_raw = df[:-1].astype(np.float32)
                label = df.iloc[-1].copy()[:].astype(int)

                if type == 'train':
                    scaler = fit_scaler(data_raw)
                data_preprocessed = preprocessing(data_raw, scaler).transpose()

                with h5py.File(
                        os.path.join(path, self.learning_config['dataset'] + '_' + self.config.type + '_' + type + '.hdf5'),
                        'w') as hdf:
                    if int(len(data_raw.columns) / len(label)) > 1:
                        dset_data = hdf.create_dataset('x_raw_' + type, data=data_raw, shape=(
                            len(data_raw.columns), len(data_raw), int(len(data_raw.columns) / len(label))),
                                                       compression='gzip',
                                                       chunks=True)
                        dset_data_pre = hdf.create_dataset('x_' + type, data=data_preprocessed, shape=(
                            len(data_preprocessed.columns), len(data_preprocessed),
                            int(len(data_preprocessed.columns) / len(label))), compression='gzip', chunks=True)
                    else:
                        dset_data = hdf.create_dataset('x_raw_' + type, data=data_raw,
                                                       shape=(len(data_raw.columns), len(data_raw)), compression='gzip',
                                                       chunks=True)
                        dset_data_pre = hdf.create_dataset('x_' + type, data=data_preprocessed,
                                                           shape=(
                                                           len(data_preprocessed.columns), len(data_preprocessed)),
                                                           compression='gzip', chunks=True)

                    dset_label = hdf.create_dataset('y_' + type, data=label, shape=(len(label), 1), compression='gzip',
                                                    chunks=True)
                    hdf.close()
                    return scaler
            else:
                df.to_csv(self.config.raw_data_folder + self.learning_config['dataset'] + '_' + self.config.type + '.csv', header=True,
                          sep=';', decimal='.',
                          float_format='%.' + '%sf' % self.config.float_decimal)
        print(
            "Dataset %s saved" % self.learning_config['dataset'])
        return 0

class PCA_Dataset:
    '''
    PCA has been already perfomed on samples of this dataset; the features are variances explained by PCs
    '''

    def __init__(self, data, name, classes=None, bay='F2', Setup='A', labelling='classification'):

        self.name = name
        self.data = data
        self.labelling = labelling
        self.bay = bay
        self.Setup = Setup

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']


    def create_dataset(self):

        self.X = np.array([self.data[key][1] for key in self.data if
                           key[-2:] == self.name.split('_')[2] and key.split(' ')[3] == self.name.split('_')[1] and
                           key.split(' ')[0] in self.classes])
        self.y = []
        self.labels = {'correct': 0, 'wrong': 0, 'inversed': 0}
        for key in self.data:
            if key[-2:] == self.bay and key.split(' ')[3] == self.Setup:
                if key.split(' ')[0] in self.classes and self.labelling == 'classification':
                    self.y = self.y + [self.classes.index(key.split(' ')[0])]
                    self.labels['correct'] = self.y.count(0)
                    self.labels['wrong'] = self.y.count(1)
                    self.labels['inversed'] = self.y.count(2)
                elif self.labelling == 'detection':
                    if key.split(' ')[0] == 'correct':
                        self.y = self.y + [0]
                        self.labels['correct'] = self.labels['correct'] + 1
                    elif key.split(' ')[0] == 'wrong':
                        self.y = self.y + [1]
                        self.labels['wrong'] = self.labels['wrong'] + 1
                    elif key.split(' ')[0] == 'inversed':
                        self.y = self.y + [1]
                        self.labels['wrong'] = self.labels['wrong'] + 1
        self.y = np.array(self.y)

    def dataset_info(self):
        print(
            f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created')


class Raw_Dataset:
    '''
        raw measurements
    '''

    def __init__(self, data, name, classes=None, bay='F2', Setup='A', labelling='classification'):

        self.name = name
        self.data = data
        self.labelling = labelling
        self.bay = bay
        self.Setup = Setup

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

    def create_dataset(self):

        self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])
        self.y = []
        self.labels = {'correct': 0, 'wrong': 0, 'inversed': 0}
        for measurement in self.X:
            if measurement.name[-2:] == self.bay and measurement.name.split(' ')[3] == self.Setup:
                if measurement.name.split(' ')[0] in self.classes and self.labelling == 'classification':
                    self.y = self.y + [self.classes.index(measurement.name.split(' ')[0])]
                    self.labels['correct'] = self.y.count(0)
                    self.labels['wrong'] = self.y.count(1)
                    self.labels['inversed'] = self.y.count(2)
                elif self.labelling == 'detection':
                    if self.data[measurement].name.split(' ')[0] == 'correct':
                        self.y = self.y + [0]
                        self.labels['correct'] = self.labels['correct'] + 1
                    elif self.data[measurement].name.split(' ')[0] == 'wrong':
                        self.y = self.y + [1]
                        self.labels['wrong'] = self.labels['wrong'] + 1
                    elif self.data[measurement].name.split(' ')[0] == 'inversed':
                        self.y = self.y + [1]
                        self.labels['wrong'] = self.labels['wrong'] + 1
        self.y = np.array(self.y)

    def dataset_info(self):
        print(
            f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created')

class Combined_Dataset:
    '''
    combines all data in one dataframe with the measurements as an index and the variables at timesteps as columns (as in ('var1',t1) ('var2',t1), ('var3',t1), ('var1',t2), ('var2',t2) ...)
    then pca is done reduce to the most important variables on the most important timesteps
    '''

    def __init__(self, data, variables, name, classes=None, bay='F2', setup='A', labelling='classification'):

        self.name = name
        self.variables = variables
        self.labelling = labelling
        self.bay = bay
        self.setup = setup

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

        self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in [data[measurement] for measurement in data if
                                                               measurement[-2:] == name.split('_')[2] and
                                                               measurement.split(' ')[3] == name.split('_')[1] and
                                                               measurement.split(' ')[0] in self.classes]}

        measurements = {}
        for measurement in data:
            if measurement[-2:] == name.split('_')[2] and measurement.split(' ')[3] == name.split('_')[1] and measurement.split(' ')[0] in self.classes:
                reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                                                     data=data[measurement].data[variables].values,
                                                                     columns=variables)
                measurements[measurement] = Combined_Dataset.flatten_df_into_row(self, reduced_measurement)

        self.combined_data = pd.DataFrame(index=[self.data[measurement].name for measurement in self.data], data=[measurements[measurement].values[0] for measurement in measurements], columns=measurements[list(measurements.keys())[0]].columns)


    def create_dataset(self):

        scaled_data = Combined_Dataset.scale(self)
        pca_data = Combined_Dataset.PCA(self, n_components=learning_config['components'])
        labelled_data = Combined_Dataset.label(self)


    def dataset_info(self):
        print(
            f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created')

    def label(self):

        self.X = np.array(self.principalComponents_selection)

        """self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])"""
        self.y = []
        self.labels = {'correct': 0, 'wrong': 0, 'inversed': 0}
        for measurement in self.data:
            if measurement[-2:] == self.bay and measurement.split(' ')[3] == self.setup:
                if measurement.split(' ')[0] in self.classes and self.labelling == 'classification':
                    self.y = self.y + [self.classes.index(measurement.split(' ')[0])]
                    self.labels['correct'] = self.y.count(0)
                    self.labels['wrong'] = self.y.count(1)
                    self.labels['inversed'] = self.y.count(2)
                elif self.labelling == 'detection':
                    if self.data[measurement].name.split(' ')[0] == 'correct':
                        self.y = self.y + [0]
                        self.labels['correct'] = self.labels['correct'] + 1
                    elif self.data[measurement].name.split(' ')[0] == 'wrong':
                        self.y = self.y + [1]
                        self.labels['wrong'] = self.labels['wrong'] + 1
                    elif self.data[measurement].name.split(' ')[0] == 'inversed':
                        self.y = self.y + [1]
                        self.labels['wrong'] = self.labels['wrong'] + 1
        self.y = np.array(self.y)

        return self.X, self.y

    def PCA(self, n_components=0.99):

        pca = PCA(n_components=n_components)

        self.principalComponents_selection = pca.fit_transform(self.combined_data_scaled)
        self.explained_variance = pca.explained_variance_ratio_

        return self.principalComponents_selection

    def scale(self):

        combined_data_scaled = StandardScaler().fit_transform(self.combined_data)
        self.combined_data_scaled = pd.DataFrame(index=self.combined_data.index, data=combined_data_scaled, columns=self.combined_data.columns)

        return self.combined_data_scaled

    def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v