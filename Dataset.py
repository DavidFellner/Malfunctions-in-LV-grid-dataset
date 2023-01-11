import pandas
import pandas as pd
import numpy as np
import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

import importlib

import detection_method_settings
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
        print(
            f'Training set: {len(self.train_set.columns)} samples, of which {sum(self.train_set.loc["label"])} positive')

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
                        os.path.join(path,
                                     self.learning_config['dataset'] + '_' + self.config.type + '_' + type + '.hdf5'),
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
                df.to_csv(
                    self.config.raw_data_folder + self.learning_config['dataset'] + '_' + self.config.type + '.csv',
                    header=True,
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
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] ==
                           self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])
        self.y = []
        if config.use_case == 'DSM':
            self.labels = {'no DSM': 0, 'DSM': 0}
            for measurement in self.X:
                if self.data[measurement].name.split(' ')[0] == 'DSM':
                    self.y = self.y + [0]
                    self.labels['DSM'] = self.labels['DSM'] + 1
                elif self.data[measurement].name.split(' ')[0] == 'no':
                    self.y = self.y + [1]
                    self.labels['no DSM'] = self.labels['no DSM'] + 1
        else:
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

        if config.use_case == 'DSM':
            self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in
                         [data[measurement] for measurement in data if
                          measurement[-2:] == name.split('_')[2] and
                          (measurement.split(' ')[2] == name.split('_')[1] or measurement.split(' ')[3] ==
                           name.split('_')[1])]}

            measurements = {}
            for measurement in data:
                if measurement[-2:] == name.split('_')[2] and (
                        measurement.split(' ')[2] == name.split('_')[1] or measurement.split(' ')[3] == name.split('_')[
                    1]):
                    reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                                       data=data[measurement].data[variables].values,
                                                       columns=variables)
                    measurements[measurement] = Combined_Dataset.flatten_df_into_row(self, reduced_measurement)
        else:
            self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in
                         [data[measurement] for measurement in data if
                          measurement[-2:] == name.split('_')[2] and
                          measurement.split(' ')[3] == name.split('_')[1] and
                          measurement.split(' ')[0] in self.classes]}
            measurements = {}
            for measurement in data:
                if measurement[-2:] == name.split('_')[2] and measurement.split(' ')[3] == name.split('_')[1] and \
                        measurement.split(' ')[0] in self.classes:
                    reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                                       data=data[measurement].data[variables].values,
                                                       columns=variables)
                    measurements[measurement] = Combined_Dataset.flatten_df_into_row(self, reduced_measurement)

        self.combined_data = pd.DataFrame(index=[self.data[measurement].name for measurement in self.data],
                                          data=[measurements[measurement].values[0] for measurement in measurements],
                                          columns=measurements[list(measurements.keys())[0]].columns).replace(np.nan,
                                                                                                              0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others

    def create_dataset(self):

        scaled_data = Combined_Dataset.scale(self)
        pca_data = Combined_Dataset.PCA(self, n_components=learning_config['components'])
        labelled_data = Combined_Dataset.label(self)

    def dataset_info(self):
        if config.use_case == 'DSM':
            print(
                f'Dataset containing {len(self.X)} samples, {self.labels["DSM"]} of which implementing DSM and {self.labels["no DSM"]} of which not implementing DSM')
        else:
            print(
                f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created (if only 2 classes inversed is called wrong)')

    def label(self):

        self.X = np.array(self.principalComponents_selection)

        """self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])"""
        self.y = []
        if config.use_case == 'DSM':
            self.labels = {'no DSM': 0, 'DSM': 0}
            for measurement in self.data:
                if self.data[measurement].name.split(' ')[0] == 'DSM':
                    self.y = self.y + [0]
                    self.labels['DSM'] = self.labels['DSM'] + 1
                elif self.data[measurement].name.split(' ')[0] == 'no':
                    self.y = self.y + [1]
                    self.labels['no DSM'] = self.labels['no DSM'] + 1
        else:
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
        self.combined_data_scaled = pd.DataFrame(index=self.combined_data.index, data=combined_data_scaled,
                                                 columns=self.combined_data.columns)

        return self.combined_data_scaled

    def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v


class Reduced_Combined_Dataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y

        pca = PCA(n_components=0.99)
        self.X = pca.fit_transform(self.X)


class Complete_Dataset:
    '''
    combines all data in one dataframe
    '''

    def __init__(self, data, variables, name, trafo_point='F2', classes=None, bay='F2', setup='A',
                 labelling='classification'):

        self.name = name
        self.variables = variables
        self.labelling = labelling
        self.bay = bay
        self.setup = setup

        self.trafo = trafo_point

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['trafo', 'participant']

        self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in
                     [data[measurement] for measurement in data] if
                     (applicable_measurements.name.split(' ')[3] or applicable_measurements.name.split(' ')[2]) ==
                     learning_config['setup_chosen'].split('_')[1]}

        measurements = {}
        samples = {}
        participant_data = []
        if config.use_case == 'DSM':
            self.num_participants = 3
        else:
            self.num_participants = 2
        counter = 1

        for measurement in self.data:

            scenario = int(measurement.split(':')[0].split(' ')[-1])
            vars = variables[measurement[-2:]]
            reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                               data=data[measurement].data[vars].values,
                                               columns=vars)

            # measurements[measurement] = Combined_Dataset.flatten_df_into_row(self, reduced_measurement)

            if measurement[-2:] == self.trafo:
                measurement_input_data = reduced_measurement
                measurement_output_data = reduced_measurement.drop(measurement_input_data.columns, axis=1)
                measurement_input_data.columns = [(name, measurement[-2:]) for name in measurement_input_data.columns]

                measurements[measurement] = [measurement_input_data, measurement_output_data]
                # measurements[measurement].index = [self.trafo]
                trafo_data = measurements[measurement]
            else:
                # measurements[measurement].index = [measurement[-2:]]

                measurement_input_data = reduced_measurement[[var for var in reduced_measurement.columns if
                                                              var.split(' ')[
                                                                  0] in variables['inputs']]]
                measurement_output_data = reduced_measurement.drop(measurement_input_data.columns, axis=1)
                measurement_input_data.columns = [(name, measurement[-2:]) for name in measurement_input_data.columns]
                measurement_output_data.columns = [(name, measurement[-2:]) for name in measurement_output_data.columns]
                measurements[measurement] = [measurement_input_data, measurement_output_data]

                participant_data.append(measurements[measurement])

            if counter % (self.num_participants + 1) == 0:
                # sample_data = (trafo_data, participant_data)
                scenario_data = [trafo_data]
                scenario_data += participant_data

                sample_X = pd.concat([data[0] for data in scenario_data], axis=1)
                sample_y = pd.concat([data[1] for data in scenario_data], axis=1)
                # sample_dict = {' '.join(measurement.split(':')[0].split(' ')[-4:]): {'X': sample_X, 'y': sample_y}}
                sample_dict = {'X': sample_X, 'y': sample_y}

                """sample_X = sample_X.replace(np.nan, 0)
                sample_y = sample_y.replace(np.nan, 0)

                sample_X_trans = sample_X.transpose()
                sample_X_trans['var'] = sample_X_trans.index
                sample_y_trans = sample_y.transpose()
                sample_y_trans['var'] = sample_y_trans.index

                sample_X_melt = sample_X_trans.melt(id_vars='var')
                sample_y_melt = sample_y_trans.melt(id_vars='var')

                sample_dict = {'X': sample_X, 'y': sample_y, 'X_melt': sample_X_melt, 'y_melt': sample_y_melt}"""

                sample_data = pd.concat(sample_dict, axis=1).replace(
                    np.nan, 0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others
                """sample_data = pd.DataFrame(index=' '.join(measurement.split(':')[0].split(' ')[-4:]),
                                             data=[sample_X, sample_y],
                                             columns=['X', 'y'])"""

                # sample_data = Complete_Dataset.combine_dfs_into_row(self, sample_df, num_participants)
                samples[' '.join(measurement.split(':')[0].split(' ')[-4:])] = sample_data
                participant_data = []

            counter += 1

        # build dataset as in 14 days as train, 1 day as test and then rotate days; this would make 15 'runs' for 'CV'
        # this way one run is one df
        # later add more features?? like lag + mean etc...

        # as dict
        self.complete_dataset = {}
        self.X = {}
        self.y = {}
        self.dfs = {}
        for sample in samples:
            self.complete_dataset[sample] = {'data': pd.concat([samples[sample]['X'], samples[sample]['y']], axis=1),
                                             'X': samples[sample]['X'], 'y': samples[sample]['y']}
            self.dfs[sample] = pd.concat([samples[sample]['X'], samples[sample]['y']], axis=1)
            self.X[sample] = np.array(samples[sample]['X'])
            self.y[sample] = np.array(samples[sample]['y'])

        self.splits = {}
        for day in self.dfs:
            self.splits[day.split(' ')[-1]] = {'training_data': pd.concat(self.removekey(self.dfs, day)),
                                               'testing_data': self.dfs[day]}

        """self.complete_dataset = pd.DataFrame(index=[measurement for measurement in samples],
                                             data=[samples[sample].values for sample in
                                                   samples],
                                             columns=[samples[sample] for sample in
                                                   samples][0].columns).replace(
            np.nan,
            0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others"""

        # DATASET: ONE SAMPLE OF INPUT IS ONE TIME POINT OF TRAFO DATA AND ONE SAMPLE OF OUTPUT IS ONE POINT OF ALL LOADS/GENS?

        """self.complete_dataset = pd.DataFrame(index=[self.data[measurement].name for measurement in self.data],
                                             data=[measurements[measurement].values[0] for measurement in measurements],
                                             columns=measurements[list(measurements.keys())[0]].columns).replace(np.nan,
                                                                                                                 0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others
"""

    def removekey(self, d, key):
        r = dict(d)
        del r[key]
        return r

    def create_dataset(self):

        # scaled_data = Complete_Dataset.scale(self)
        Complete_Dataset.label(self)

    def dataset_info(self):
        if config.disaggregation:
            print(
                f'Dataset containing {len(self.X.keys())} samples, containing {len(self.X.keys()) * 1} measurements at a transformer measurement point, and {len(self.X.keys()) * self.num_participants} measurements at grid participants measurement points')
        else:
            if config.use_case == 'DSM':
                print(
                    f'Dataset containing {len(self.X)} samples, {self.labels["DSM"]} of which implementing DSM and {self.labels["no DSM"]} of which not implementing DSM')
            else:
                print(
                    f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created (if only 2 classes inversed is called wrong)')

    def label(self):

        # self.X = np.array(self.data)

        """self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])"""
        # self.y = []
        self.labels = {'trafo': 0, 'participant': 0}
        for measurement in self.data:
            if measurement[
               -2:] == self.trafo:  # and (measurement.split(' ')[3] or measurement.split(' ')[2]) == self.setup:
                # self.y = self.y + [0]
                self.labels['trafo'] = self.labels['trafo'] + 1
            else:
                # self.y = self.y + [1]
                self.labels['participant'] = self.labels['participant'] + 1
        # self.y = np.array(self.y)

        # return self.X, self.y

    def scale(self):

        combined_data_scaled = StandardScaler().fit_transform(self.combined_data)
        self.combined_data_scaled = pd.DataFrame(index=self.combined_data.index, data=combined_data_scaled,
                                                 columns=self.combined_data.columns)

        return self.combined_data_scaled

    def combine_dfs_into_row(self, df, num_of_participants):

        # v = df.unstack().to_frame().sort_index(level=1).T
        # v.columns = v.columns.map(str)

        v = df.unstack().to_frame().T

        v.columns = v.columns.map(str)

        counter = 1
        entry = []
        entries = []
        for column in v.columns:
            entry.append(v[column])
            if counter % (num_of_participants + 1) == 0:
                entries.append(entry)
                entry = []
            counter += 1

        new_df = pandas.DataFrame(index=v.index, data=entries,
                                  columns=df.unstack().to_frame().T.droplevel(1, axis=1).columns[::num_of_participants])

        return new_df

    """def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v"""


class Sensor_Dataset:
    '''
    combines all data in one dataframe with the measurements as an index and the variables at timesteps as columns (as in ('var1',t1) ('var2',t1), ('var3',t1), ('var1',t2), ('var2',t2) ...)
    then pca is done reduce to the most important variables on the most important timesteps
    '''

    def __init__(self, data, phase_info, variables, name, classes=None, setup='A', labelling='classification'):

        self.name = name
        self.variables = variables
        self.labelling = labelling
        self.setup = setup
        self.phase = phase_info[0].split('_')[-1]
        self.trafo_point = phase_info[1][0]
        self.test_bays = config.test_bays_dict[self.phase]

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

        if self.phase == 'phase2':
            self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in
                         [data[measurement] for measurement in data if
                          measurement[-2:] == name.split('_')[2] and
                          (measurement.split(' ')[2] == name.split('_')[1] or measurement.split(' ')[3] ==
                           name.split('_')[1])]}

            measurements = {}
            for measurement in data:
                if measurement[-2:] == name.split('_')[2] and (
                        measurement.split(' ')[2] == name.split('_')[1] or measurement.split(' ')[3] == name.split('_')[
                    1]):
                    reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                                       data=data[measurement].data[variables].values,
                                                       columns=variables)
                    measurements[measurement] = Sensor_Dataset.flatten_df_into_row(self, reduced_measurement)
        else:
            self.trafo_data_correct = {applicable_measurements.name: data[applicable_measurements.name] for
                                       applicable_measurements in
                                       [data[measurement] for measurement in data if
                                        measurement[-2:] == self.trafo_point and
                                        measurement.split(' ')[3] == setup and
                                        measurement.split(' ')[0] == 'correct']}

            self.load_data_correct = {applicable_measurements.name: data[applicable_measurements.name] for
                                      applicable_measurements in
                                      [data[measurement] for measurement in data if not
                                      measurement[-2:] == self.trafo_point and
                                       measurement.split(' ')[3] == setup and
                                       measurement.split(' ')[0] == 'correct']}

            self.trafo_data_wrong = {applicable_measurements.name: data[applicable_measurements.name] for
                                     applicable_measurements in
                                     [data[measurement] for measurement in data if
                                      measurement[-2:] == self.trafo_point and
                                      measurement.split(' ')[3] == setup and
                                      measurement.split(' ')[0] == 'wrong']}

            self.load_data_wrong = {applicable_measurements.name: data[applicable_measurements.name] for
                                    applicable_measurements in
                                    [data[measurement] for measurement in data if not
                                    measurement[-2:] == self.trafo_point and
                                     measurement.split(' ')[3] == setup and
                                     measurement.split(' ')[0] == 'wrong']}

            data_dict = {'trafo_correct': self.trafo_data_correct, 'trafo_correct_unflat': self.trafo_data_correct,
                         'load_correct': self.load_data_correct,
                         'load_correct_unflat': self.load_data_correct, 'trafo_wrong': self.trafo_data_wrong,
                         'load_wrong': self.load_data_wrong}
            for data in data_dict:
                measurements = {}
                for measurement in data_dict[data]:
                    reduced_measurement = pd.DataFrame(index=data_dict[data][measurement].data.index,
                                                       data=data_dict[data][measurement].data[
                                                           variables[measurement[-2:]][1]].values,
                                                       columns=variables[measurement[-2:]][1])
                    if data == 'trafo_correct_unflat' or data == 'load_correct_unflat':
                        measurements[measurement] = reduced_measurement
                    else:
                        measurements[measurement] = Sensor_Dataset.flatten_df_into_row(self, reduced_measurement)

                if data == 'trafo_correct_unflat':
                    index_length = len(
                        [data_dict[data][measurement].data.index for measurement in data_dict[data]][0]) * len(
                        [data_dict[data][measurement].data.index for measurement in data_dict[data]])
                    data_values = np.concatenate([measurements[measurement].values for measurement in measurements])
                    columns = [column + ' ' + self.trafo_point for column in measurements[list(measurements.keys())[0]].columns]
                    data_dict[data] = pd.DataFrame(
                        index=range(index_length),
                        data=data_values,
                        columns=columns).replace(np.nan,
                                                 0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others
                elif data == 'load_correct_unflat':
                    index_length = int((len(
                        [data_dict[data][measurement].data.index for measurement in data_dict[data]][0]) * len(
                        [data_dict[data][measurement].data.index for measurement in data_dict[data]])) / (len(self.test_bays)-1))
                    columns = []
                    data_values = []
                    for test_bay in self.test_bays:
                        if test_bay == self.trafo_point:
                            continue
                        else:
                            columns += (Sensor_Dataset.deliver_load_column_names(self, measurements[list(measurements.keys())[0]].columns, test_bay))
                            data_values.append(np.concatenate([measurements[measurement].values for measurement in measurements if measurement.split(' ')[-1] == test_bay]))
                    data_values = np.concatenate([i for i in data_values], axis=1)
                    data_dict[data] = pd.DataFrame(
                        index=range(index_length),
                        data=data_values,
                        columns=columns).replace(np.nan,
                                                 0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others

                else:
                    data_dict[data] = pd.DataFrame(
                        index=[data_dict[data][measurement].name for measurement in data_dict[data]],
                        data=[measurements[measurement].values[0] for measurement in measurements],
                        columns=measurements[list(measurements.keys())[0]].columns).replace(np.nan,
                                                                                            0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others

            self.trafo_data_correct = data_dict['trafo_correct']
            self.trafo_data_correct_unflattened = data_dict['trafo_correct_unflat']
            self.load_data_correct = data_dict['load_correct']
            self.load_data_correct_unflattened = data_dict['load_correct_unflat']
            self.trafo_data_wrong = data_dict['trafo_wrong']
            self.load_data_wrong = data_dict['load_wrong']

    def create_dataset(self):

        scaled_data = Sensor_Dataset.scale(self)
        pca_data = Sensor_Dataset.PCA(self, n_components=learning_config['components'])
        labelled_data = Sensor_Dataset.label(self)

    def dataset_info(self):
        if config.use_case == 'DSM':
            print(
                f'Dataset containing {len(self.X)} samples, {self.labels["DSM"]} of which implementing DSM and {self.labels["no DSM"]} of which not implementing DSM')
        else:
            print(
                f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created (if only 2 classes inversed is called wrong)')

    def label(self):

        self.X = np.array(self.principalComponents_selection)

        """self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])"""
        self.y = []
        if config.use_case == 'DSM':
            self.labels = {'no DSM': 0, 'DSM': 0}
            for measurement in self.data:
                if self.data[measurement].name.split(' ')[0] == 'DSM':
                    self.y = self.y + [0]
                    self.labels['DSM'] = self.labels['DSM'] + 1
                elif self.data[measurement].name.split(' ')[0] == 'no':
                    self.y = self.y + [1]
                    self.labels['no DSM'] = self.labels['no DSM'] + 1
        else:
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

        # APPLY SAME SCALER TO WRONG DATA AS THE ONE FITTED TO CORRECT DATA > bei vollem Dataset dann
        combined_data_scaled = StandardScaler().fit_transform(self.combined_data)
        self.combined_data_scaled = pd.DataFrame(index=self.combined_data.index, data=combined_data_scaled,
                                                 columns=self.combined_data.columns)

        return self.combined_data_scaled

    def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v

    def deliver_load_column_names(self, data, test_bay):

        column_names = [column_name + ' ' + test_bay for column_name in data]

        return column_names


# assemble 14 + 14 + 1 (or 2?) dataset here and then scale all individually first (14c real + 1w real + 1c real; 14w sim) and then all together (maybe have to cut dimensions first) and then do PCA together; then label oc
# put all combinations in one dataset, or create 15 different ones? > whatever is easier to rotate samples with / have results to store (maybe one dataset per phase?)
class Application_Dataset:
    '''
    combines all data in one dataframe with the measurements as an index and the variables at timesteps as columns (as in ('var1',t1) ('var2',t1), ('var3',t1), ('var1',t2), ('var2',t2) ...)
    then pca is done reduce to the most important variables on the most important timesteps
    '''

    def __init__(self, data, phase_info, variables, name, classes=None, setup='A', labelling='classification'):

        self.name = name
        self.variables = variables
        self.labelling = labelling
        self.setup = setup
        self.phase = phase_info[0].split('_')[-1]
        self.trafo_point = phase_info[1][0]
        self.test_bays = config.test_bays_dict[self.phase]
        #self.trafo_data_correct = data[0].trafo_data_correct
        #self.trafo_data_wrong = data[0].trafo_data_wrong
        self.trafo_data_correct = Application_Dataset.df_from_meas_dict(self, data[0], self.variables)
        self.trafo_data_wrong = Application_Dataset.df_from_meas_dict(self, data[1], self.variables)
        self.sim_data_wrong = None
        self.scenario_combos_data = {}

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

        sim_data_df = Application_Dataset.df_from_meas_dict(self, data[2], self.variables)
        self.sim_data_wrong = sim_data_df

    def create_dataset(self):

        self.scenario_combos_data = {}

        for scenario in self.trafo_data_correct.index:
            # SCALE CORRECT HISTORIC REAL WORLD DATA FIRST AND THEN USE SAME SCALER ON ALL OTHER DATA > THEN PCA TO HAVE SAME NUMBER OF PCS > THEN CONCAT > THEN LABEL > THEN SPLIT IN TRAIN/TEST > THEN USE ALL THE OBTAINED COMBINATIONS

            combo = scenario.split(' ')[5][:-1]

            trafo_correct_hist = self.trafo_data_correct.drop(self.trafo_data_correct.index[int(combo) - 1],
                                                              axis=0)  # use all but 'latest' one

            trafo_correct_hist_scaled, scaler = Application_Dataset.scale(self, trafo_correct_hist)

            """trafo_correct_hist_pca_data = Application_Dataset.PCA(self, trafo_correct_hist_scaled,
                                                                  n_components=learning_config['components'])"""

            trafo_correct_test = self.trafo_data_correct.iloc[int(combo) - 1]  # use only the 'latest' one
            trafo_wrong_test = self.trafo_data_wrong.iloc[int(combo) - 1]  # use only the 'latest' one
            trafo_test = pd.concat([trafo_correct_test, trafo_wrong_test], axis=1).transpose()

            trafo_test_scaled, scaler = Application_Dataset.scale(self, trafo_test,
                                                                  scaler=scaler)  # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

            trafo_pca_data = Application_Dataset.PCA(self, pd.concat([trafo_correct_hist_scaled, trafo_test_scaled]),
                                                     n_components=len(
                                                         trafo_correct_hist))  # HAS TO BE DONE TOGETHER WITH REST OF REAL DATA, AS MIN NUM SAMPLES DETERMINES PCs
            trafo_test_pca_data = trafo_pca_data[-2:]
            trafo_correct_hist_pca_data = trafo_pca_data[:-2]

            """trafo_wrong_scaled, scaler = Application_Dataset.scale(self, trafo_wrong_test, scaler=scaler)
            trafo_wrong_pca_data = Application_Dataset.PCA(self, trafo_wrong_scaled,
                                                           n_components=len(trafo_correct_hist_pca_data.columns()))"""

            sim_wrong = self.sim_data_wrong.drop(self.sim_data_wrong.index[int(combo) - 1],
                                                 axis=0)  # use only 'historic' ones
            sim_wrong_scaled, scaler = Application_Dataset.scale(self,
                                                                 sim_wrong)  # scale with its own scaler, since dmesnions differ due to different number of varioables (30 for sim, 84 for real data)
            sim_wrong_pca_data = Application_Dataset.PCA(self, sim_wrong_scaled,
                                                         n_components=len(trafo_correct_hist_pca_data))

            combined_training_data = np.concatenate((trafo_correct_hist_pca_data,
                                                     sim_wrong_pca_data))  # df of data of 1 of 15 combos (historic)
            combined_testing_data = trafo_test_pca_data  # df of data of 1 of 15 combos (samples to classify/test on)

            labelled_training_data = {'X': combined_training_data,
                                      'y': [0] * len(trafo_correct_hist) + [1] * len(sim_wrong)}            #ALTER LABELLING IN CASE NOT JUST CORRECT ADN WRONG > AS FOR DETECTION
            labelled_test_data = {'X': combined_testing_data, 'y': [0, 1]}

            self.scenario_combos_data['combination_' + combo] = {'training': labelled_training_data,
                                                                 'testing': labelled_test_data}  # training: all historic data (14 real correct + 14 sim wrong samples); testing data : 1 real correct and 1 real wrong sample

        print(f'Datasets of {len(self.scenario_combos_data.keys())} combinations assembled')

        return self.scenario_combos_data

    def dataset_info(self):
        print(
            f'Dataset containing {len(self.scenario_combos_data)} combinations of 28 historic samples (14 correct and 14 corresponding wrong ones) for training and 2 testing samples (one correct one wrong) created')

    def PCA(self, data, n_components=0.99):

        pca = PCA(n_components=n_components)

        principalComponents_selection = pca.fit_transform(data)
        explained_variance = pca.explained_variance_ratio_

        return principalComponents_selection

    def scale(self, data, scaler=None):

        # APPLY SAME SCLAER TO WRONG DATA AS THE ONE FITTED TO CORRECT DATA
        if not scaler:
            scaler = StandardScaler().fit(data)

        data_scaled = scaler.transform(data)

        data_scaled = pd.DataFrame(index=data.index, data=data_scaled,
                                   columns=data.columns)

        return data_scaled, scaler

    def df_from_meas_dict(self, data_dict, variables):

        measurements = {}
        for measurement in data_dict:
            reduced_measurement = pd.DataFrame(index=data_dict[measurement].data.index,
                                               data=data_dict[measurement].data[
                                                   variables[measurement[-2:]][1]].values,
                                               columns=variables[measurement[-2:]][1])
            measurements[measurement] = Sensor_Dataset.flatten_df_into_row(self, reduced_measurement)

        data_df = pd.DataFrame(index=[data_dict[measurement].name for measurement in data_dict],
                               data=[measurements[measurement].values[0] for measurement in measurements],
                               columns=measurements[list(measurements.keys())[0]].columns).replace(np.nan,
                                                                                                   0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others

        return data_df

    def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v


class Complete_Dataset_orig:
    '''
    combines all data in one dataframe
    '''

    def __init__(self, data, variables, name, trafo_point='F2', classes=None, bay='F2', setup='A',
                 labelling='classification'):

        self.name = name
        self.variables = variables
        self.labelling = labelling
        self.bay = bay
        self.setup = setup

        self.trafo = trafo_point

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['trafo', 'participant']

        self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in
                     [data[measurement] for measurement in data]}

        measurements = {}
        samples = {}
        participant_data = []
        if config.use_case == 'DSM':
            self.num_participants = 3
        else:
            self.num_participants = 2
        counter = 1

        for measurement in data:

            scenario = int(measurement.split(':')[0].split(' ')[-1])
            vars = variables[measurement[-2:]]
            reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                               data=data[measurement].data[vars].values,
                                               columns=vars)

            # measurements[measurement] = Combined_Dataset.flatten_df_into_row(self, reduced_measurement)

            if measurement[-2:] == self.trafo:
                measurement_input_data = reduced_measurement
                measurement_output_data = reduced_measurement.drop(measurement_input_data.columns, axis=1)
                measurement_input_data.columns = [(name, measurement[-2:]) for name in measurement_input_data.columns]

                measurements[measurement] = [measurement_input_data, measurement_output_data]
                # measurements[measurement].index = [self.trafo]
                trafo_data = measurements[measurement]
            else:
                # measurements[measurement].index = [measurement[-2:]]

                measurement_input_data = reduced_measurement[[var for var in reduced_measurement.columns if
                                                              var.split(' ')[
                                                                  0] in variables['inputs']]]
                measurement_output_data = reduced_measurement.drop(measurement_input_data.columns, axis=1)
                measurement_input_data.columns = [(name, measurement[-2:]) for name in measurement_input_data.columns]
                measurement_output_data.columns = [(name, measurement[-2:]) for name in measurement_output_data.columns]
                measurements[measurement] = [measurement_input_data, measurement_output_data]

                participant_data.append(measurements[measurement])

            if counter % (self.num_participants + 1) == 0:
                # sample_data = (trafo_data, participant_data)
                scenario_data = [trafo_data]
                scenario_data += participant_data

                sample_X = pd.concat([data[0] for data in scenario_data], axis=1)
                sample_y = pd.concat([data[1] for data in scenario_data], axis=1)
                # sample_dict = {' '.join(measurement.split(':')[0].split(' ')[-4:]): {'X': sample_X, 'y': sample_y}}
                sample_dict = {'X': sample_X, 'y': sample_y}
                sample_data = pd.concat(sample_dict, axis=1).replace(
                    np.nan, 0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others
                """sample_data = pd.DataFrame(index=' '.join(measurement.split(':')[0].split(' ')[-4:]),
                                             data=[sample_X, sample_y],
                                             columns=['X', 'y'])"""

                # sample_data = Complete_Dataset.combine_dfs_into_row(self, sample_df, num_participants)
                samples[' '.join(measurement.split(':')[0].split(' ')[-4:])] = sample_data
                participant_data = []

            counter += 1

        # as dict
        self.complete_dataset = {}
        self.X = {}
        self.y = {}
        for sample in samples:
            self.complete_dataset[sample] = {'X': samples[sample]['X'], 'y': samples[sample]['y']}
            self.X[sample] = np.array(samples[sample]['X'])
            self.y[sample] = np.array(samples[sample]['y'])

        """self.complete_dataset = pd.DataFrame(index=[measurement for measurement in samples],
                                             data=[samples[sample].values for sample in
                                                   samples],
                                             columns=[samples[sample] for sample in
                                                   samples][0].columns).replace(
            np.nan,
            0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others"""

        # DATASET: ONE SAMPLE OF INPUT IS ONE TIME POINT OF TRAFO DATA AND ONE SAMPLE OF OUTPUT IS ONE POINT OF ALL LOADS/GENS?

        """self.complete_dataset = pd.DataFrame(index=[self.data[measurement].name for measurement in self.data],
                                             data=[measurements[measurement].values[0] for measurement in measurements],
                                             columns=measurements[list(measurements.keys())[0]].columns).replace(np.nan,
                                                                                                                 0)  # NaN values filled up with 0; NaN can occur when a measurement is shorter than others
"""

    def create_dataset(self):

        # scaled_data = Complete_Dataset.scale(self)
        Complete_Dataset.label(self)

    def dataset_info(self):
        if config.disaggregation:
            print(
                f'Dataset containing {len(self.X.keys())} samples, containing {len(self.X.keys()) * 1} measurements at a transformer measurement point, and {len(self.X.keys()) * self.num_participants} measurements at grid participants measurement points')
        else:
            if config.use_case == 'DSM':
                print(
                    f'Dataset containing {len(self.X)} samples, {self.labels["DSM"]} of which implementing DSM and {self.labels["no DSM"]} of which not implementing DSM')
            else:
                print(
                    f'Dataset containing {len(self.X)} samples, {self.labels["correct"]} of which correct, {self.labels["wrong"]} of which wrong, and {self.labels["inversed"]} of which inversed created (if only 2 classes inversed is called wrong)')

    def label(self):

        # self.X = np.array(self.data)

        """self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])"""
        # self.y = []
        self.labels = {'trafo': 0, 'participant': 0}
        for measurement in self.data:
            if measurement[
               -2:] == self.trafo:  # and (measurement.split(' ')[3] or measurement.split(' ')[2]) == self.setup:
                # self.y = self.y + [0]
                self.labels['trafo'] = self.labels['trafo'] + 1
            else:
                # self.y = self.y + [1]
                self.labels['participant'] = self.labels['participant'] + 1
        # self.y = np.array(self.y)

        # return self.X, self.y

    def scale(self):

        combined_data_scaled = StandardScaler().fit_transform(self.combined_data)
        self.combined_data_scaled = pd.DataFrame(index=self.combined_data.index, data=combined_data_scaled,
                                                 columns=self.combined_data.columns)

        return self.combined_data_scaled

    def combine_dfs_into_row(self, df, num_of_participants):

        # v = df.unstack().to_frame().sort_index(level=1).T
        # v.columns = v.columns.map(str)

        v = df.unstack().to_frame().T

        v.columns = v.columns.map(str)

        counter = 1
        entry = []
        entries = []
        for column in v.columns:
            entry.append(v[column])
            if counter % (num_of_participants + 1) == 0:
                entries.append(entry)
                entry = []
            counter += 1

        new_df = pandas.DataFrame(index=v.index, data=entries,
                                  columns=df.unstack().to_frame().T.droplevel(1, axis=1).columns[::num_of_participants])

        return new_df

    """def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v"""
