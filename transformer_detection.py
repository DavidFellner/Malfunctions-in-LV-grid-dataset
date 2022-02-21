from plot_measurements import plot_scenario_test_bay, plot_scenario_case, plot_pca, plot_grid_search
from detection_method_settings import measurements
from Measurement import Measurement
from Clustering import Clustering
from util import create_dataset

import os
import pandas as pd

class Transformer_detection:

    def __init__(self, config, learning_config):

        self.data_path = config.data_path
        self.test_bays = config.test_bays
        self.scenario = config.scenario
        self.plotting_variables = config.plotting_variables
        self.variables = config.variables
        self.sampling_step_size_in_seconds = config.sampling_step_size_in_seconds
        self.setups = config.setups
        self.plot_data = config.plot_data

        self.setup_chosen = learning_config['setup_chosen']
        self.mode = learning_config['detection']
        self.data_mode = learning_config['combined_data']

        self.approach = learning_config['PCA+clf']

    def plotting_data(self):
        fgs_test_bay, axs_test_bay = self.scenario_plotting_test_bay(self.variables, plot_all=False, vars=self.plotting_variables,
                                                                sampling=self.sampling_step_size_in_seconds)
        fgs_case, axs_case = self.scenario_plotting_case(self.variables, plot_all=False, vars=self.plotting_variables,
                                                    sampling=self.sampling_step_size_in_seconds)

    def scenario_plotting_test_bay(self, variables, plot_all=True, scenario=1, vars=None, sampling=None):
        if vars is None:
            vars = {'B1': 'Vrms ph-n AN Avg', 'F1': 'Vrms ph-n AN Avg', 'F2': 'Vrms ph-n L1N Avg'}
        fgs = {}
        axs = {}

        try:
            var_numbers = [variables[i][0].index(vars[i]) + 1 for i in
                           vars.keys()]  # +1 bc first column of data is useless and therefore not in variable list
        except ValueError:
            print(f"The variable  defined is not available")
            return fgs, axs

        vars = {'B1': (vars['B1'], var_numbers[0]), 'F1': (vars['F1'], var_numbers[1]),
                'F2': (vars['F2'], var_numbers[2])}
        if plot_all:
            for scenario in range(1, 16):
                relevant_measurements = self.load_data(scenario, sampling=sampling)
                fgs, axs = plot_scenario_test_bay(relevant_measurements, fgs, axs, vars)
        else:
            relevant_measurements = self.load_data(scenario, sampling=sampling)
            fgs, axs = plot_scenario_test_bay(relevant_measurements, fgs, axs, vars)

        return fgs, axs

    def scenario_plotting_case(self, variables, plot_all=True, scenario=1, vars=None, sampling=None):
        if vars is None:
            vars = {'B1': 'Vrms ph-n AN Avg', 'F1': 'Vrms ph-n AN Avg', 'F2': 'Vrms ph-n AN Avg'}
        fgs = {}
        axs = {}

        try:
            var_numbers = [variables[i][0].index(vars[i]) + 1 for i in
                           vars.keys()]  # +1 bc first column of data is useless and therefore not in variable list
        except ValueError:
            print(f"The variable  defined is not available")
            return fgs, axs

        vars = {'B1': (vars['B1'], var_numbers[0]), 'F1': (vars['F1'], var_numbers[1]),
                'F2': (vars['F2'], var_numbers[2])}
        if plot_all:
            for scenario in range(1, 16):
                relevant_measurements = self.load_data(scenario, sampling=sampling)
                fgs, axs = plot_scenario_case(relevant_measurements, fgs, axs, vars)
        else:
            relevant_measurements = self.load_data(scenario, sampling=sampling)
            fgs, axs = plot_scenario_case(relevant_measurements, fgs, axs, vars)

        return fgs, axs

    def load_data(self, scenario=None, sampling=None):
        print('Data loaded with sampling of ' + str(sampling))
        relevant_measurements = {}
        if scenario:
            # to get data of entire scenario
            for measurement in measurements:
                for test_bay in self.test_bays:
                    full_path = os.path.join(self.data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                    data = pd.read_csv(os.path.join(full_path, measurements[measurement][scenario - 1] + '.csv'),
                                       sep=',',
                                       decimal=',', low_memory=False)
                    data = data[
                           2 * 60 * 4:]  # cut off the first 2 minutes because this is where laods / PV where started up
                    data = data[
                           :6000]  # cut off after 25 minutes (25*60*4 bc 4 samples per second) because measurements were not turned off at same time
                    data['new_index'] = range(len(data))
                    data = data.set_index('new_index')

                    if sampling:
                        data = self.sample(data, sampling)

                    name = str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(test_bay)
                    relevant_measurements[
                        str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(
                            test_bay)] = Measurement(
                        data, name)
        else:
            # get all data
            for measurement in measurements:
                for scenario in measurements[measurement]:
                    for test_bay in self.test_bays:
                        full_path = os.path.join(self.data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                        data = pd.read_csv(os.path.join(full_path, scenario + '.csv'),
                                           sep=',',
                                           decimal=',', low_memory=False)
                        data = data[
                               2 * 60 * 4:]  # cut off the first 2 minutes because this is where laods / PV where started up
                        data = data[
                               :6000]  # cut off after 25 minutes (25*60*4 bc 4 samples per second) because measurements were not turned off at same time
                        data['new_index'] = range(len(data))
                        data = data.set_index('new_index')

                        if sampling:
                            data = self.sample(data, sampling)

                        name = str(measurement)[13:] + ' Scenario ' + str(
                            measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(test_bay)
                        relevant_measurements[
                            str(measurement)[13:] + ' Scenario ' + str(
                                measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(
                                test_bay)] = Measurement(
                            data, name)

        return relevant_measurements

    def sample(self, data, sampling):
        datetimeindex = pd.DataFrame(columns=['Datetime'], data=pd.to_datetime(data['Datum'] + ' ' + data['Zeit']))
        data = pd.concat((data, datetimeindex), axis=1)
        data = data.set_index('Datetime')
        # data = data.drop(['Datum', 'Zeit'], axis=1)
        data.resample(str(sampling) + 'S')
        index = pd.DataFrame(index=datetimeindex['Datetime'], columns=['new_index'], data=range(len(data)))
        data = pd.concat((data, index), axis=1)
        sampled_data = data.set_index('new_index')

        return sampled_data

    def clustering(self):
        data = self.load_data(sampling=self.sampling_step_size_in_seconds)
        dataset = create_dataset(type='raw', data=data, name=self.setup_chosen, classes=self.setups[self.setup_chosen], bay=self.setup_chosen.split('_')[2],
                              Setup=self.setup_chosen.split('_')[1], labelling=self.mode)

        Clustering_ward = Clustering(data=dataset, variables=self.variables[self.setup_chosen.split('_')[2]][1],
                                     metric='euclidean', method='ward', num_of_clusters=len(self.setups[self.setup_chosen]))
        cluster_assignments = Clustering_ward.cluster()
        score = Clustering_ward.rand_score()
        print(f'Score reached: {score}')
