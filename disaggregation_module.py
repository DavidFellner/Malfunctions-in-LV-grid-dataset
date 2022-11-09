from plot_measurements import plot_scenario_test_bay, plot_scenario_case, plot_pca, plot_grid_search
from Measurement import Measurement
from Clustering import Clustering
from util import create_dataset
from detection_method_settings import Variables

v = Variables()

import importlib
from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

try:
    try:
        if config.use_case == 'DSM':
            from detection_method_settings import measurements_DSM as measurements
    except AttributeError:
        if config.extended and learning_config['data_source'] == 'simulation':
            from detection_method_settings import measurements_extended as measurements
        else:
            from detection_method_settings import measurements as measurements
except AttributeError:
    pass

import os
import pandas as pd
import numpy as np
import statistics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import math


class Disaggregation:

    def __init__(self, config, learning_config):

        self.sim_data_path = config.sim_data_path
        self.pf_file = config.pf_file
        self.data_path = config.data_path
        self.test_bays = config.test_bays
        self.scenario = config.scenario
        self.plot_all = config.plot_all
        self.plotting_variables = config.plotting_variables
        self.variables = config.disaggregation_variables
        self.sampling_step_size_in_seconds = config.sampling_step_size_in_seconds
        self.setups = config.setups
        self.plot_data = config.plot_data

        self.data_source = learning_config['data_source']
        self.setup_chosen = learning_config['setup_chosen']
        self.mode = learning_config['mode']
        self.data_mode = learning_config['data_mode']
        self.selection = learning_config['selection']
        self.disagg_algo = learning_config['disaggregation algorithm']
        self.kernels = learning_config['kernels']
        self.gammas = learning_config['gammas']
        self.degrees = learning_config['degrees']
        self.neighbours = learning_config['neighbours']
        self.weights = learning_config['weights']

        self.approach = learning_config['approach']

    def plotting_data(self):
        fgs_test_bay, axs_test_bay = self.scenario_plotting_test_bay(self.variables, plot_all=self.plot_all,
                                                                     scenario=self.scenario,
                                                                     vars=self.plotting_variables,
                                                                     sampling=self.sampling_step_size_in_seconds)
        fgs_case, axs_case = self.scenario_plotting_case(self.variables, plot_all=self.plot_all, scenario=self.scenario,
                                                         vars=self.plotting_variables,
                                                         sampling=self.sampling_step_size_in_seconds)

        if config.save_figures:
            if not os.path.isdir(os.path.join(config.raw_data_folder, 'Graphs')):
                os.mkdir(os.path.join(config.raw_data_folder, 'Graphs'))

            if config.extended:
                extended = '_extended'
            else:
                extended = ''
            for fig in fgs_test_bay:
                fgs_test_bay[fig].set_size_inches(12, 12, forward=True)
                fgs_test_bay[fig].subplots_adjust(hspace=0.275, top=0.925)
                fgs_test_bay[fig].savefig(os.path.join(config.raw_data_folder, 'Graphs',
                                                       'scenario_' + fig + '_test_bay_' + learning_config[
                                                           'data_source'] + extended), dpi=fgs_test_bay[fig].dpi,
                                          bbox_inches='tight')
                fgs_test_bay[fig].savefig(os.path.join(config.raw_data_folder, 'Graphs',
                                                       'scenario_' + fig + '_test_bay_' + learning_config[
                                                           'data_source'] + extended + '.pdf'), dpi=fgs_test_bay[fig].dpi,
                                          bbox_inches='tight', format='pdf')
            for fig in fgs_case:
                fgs_case[fig].set_size_inches(12, 12, forward=True)
                fgs_case[fig].subplots_adjust(hspace=0.275, top=0.925)
                fgs_case[fig].savefig(os.path.join(config.raw_data_folder, 'Graphs',
                                                   'scenario_' + fig + '_case_' + learning_config['data_source'] + extended),
                                      dpi=fgs_case[fig].dpi, bbox_inches='tight')
                fgs_case[fig].savefig(os.path.join(config.raw_data_folder, 'Graphs',
                                                   'scenario_' + fig + '_case_' + learning_config[
                                                       'data_source'] + extended + '.pdf'),
                                      dpi=fgs_case[fig].dpi, bbox_inches='tight', format='pdf')

    def scenario_plotting_test_bay(self, variables, plot_all=True, scenario=1, vars=None, sampling=None):
        if vars is None:
            vars = {'B1': 'Vrms ph-n AN Avg', 'F1': 'Vrms ph-n AN Avg', 'F2': 'Vrms ph-n L1N Avg'}
        fgs = {}
        axs = {}

        if learning_config['data_source'] == 'simulation':
            vars_in_data = self.load_data(1, sampling=sampling)  # .columns
            var_numbers = [list(vars_in_data[i].data.columns).index(vars[i.split(' ')[-1]]) for i in
                           vars_in_data.keys()]
        else:
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

        if learning_config['data_source'] == 'simulation':
            vars_in_data = self.load_data(1, sampling=sampling)  # .columns
            var_numbers = [list(vars_in_data[i].data.columns).index(vars[i.split(' ')[-1]]) for i in
                           vars_in_data.keys()]
        else:
            try:
                var_numbers = [variables[i][0].index(vars[i]) + 1 for i in
                               vars.keys()]  # +1 bc first column of data is useless and therefore not in variable list
            except ValueError:
                print(f"The variable  defined is not available")
                return fgs, axs

        vars = {'B1': (vars['B1'], var_numbers[0]), 'F1': (vars['F1'], var_numbers[1]),
                'F2': (vars['F2'], var_numbers[2])}

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
            if self.data_source == 'real_world':
                # to get data of entire scenario
                for measurement in measurements:
                    for test_bay in self.test_bays:
                        full_path = os.path.join(self.data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                        data = pd.read_csv(os.path.join(full_path, measurements[measurement][scenario - 1] + '.csv'),
                                           sep=',',
                                           decimal=',', low_memory=False)
                        data = data[
                               2 * 60 * 4:]  # cut off the first 2 minutes because this is where loads / PV where started up
                        data = data[
                               :6000]  # cut off after 25 minutes (25*60*4 bc 4 samples per second) because measurements were not turned off at same time
                        data['timestep'] = range(len(data))
                        data = data.set_index('timestep')

                        if sampling:
                            data = self.sample(data, sampling)

                        name = str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(test_bay)
                        relevant_measurements[
                            str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(
                                test_bay)] = Measurement(
                            data, name)
            if self.data_source == 'simulation':
                # to get data of entire scenario
                for measurement in measurements:
                    for test_bay in self.test_bays:
                        full_path = os.path.join(self.sim_data_path, self.pf_file, 'Test_Bay_' + test_bay)
                        data = pd.read_csv(os.path.join(full_path,
                                                        f'scenario_{scenario}_{measurement.split(" ")[1]}_control_Setup_{measurement.split(" ")[4]}.csv'),
                                           sep=',',
                                           decimal=',', low_memory=False)
                        data['timestep'] = range(len(data))
                        data = data.set_index('timestep')

                        if sampling:
                            data = self.sample(data, sampling)

                        name = str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(test_bay)
                        relevant_measurements[
                            str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(
                                test_bay)] = Measurement(
                            data, name)
        else:
            # get all data
            if self.data_source == 'real_world':
                for measurement in measurements:
                    for scenario in measurements[measurement]:
                        for test_bay in self.test_bays:
                            full_path = os.path.join(self.data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                            data = pd.read_csv(
                                os.path.join(full_path, measurements[measurement][
                                    measurements[measurement].index(scenario)] + '.csv'),
                                sep=',',
                                decimal=',', low_memory=False)
                            data = data[
                                   2 * 60 * 4:]  # cut off the first 2 minutes because this is where laods / PV where started up
                            data = data[
                                   :6000]  # cut off after 25 minutes (25*60*4 bc 4 samples per second) because measurements were not turned off at same time

                            if test_bay == 'C1':
                                script_name = f'ERI-Grid - Scenario {int(measurements[measurement].index(scenario))+1}.3_(use_in_all_Setups).txt'
                                script_path = os.path.join(self.data_path, 'Test_Bay_' + test_bay,
                                                         'Load_data', script_name)
                                with open(script_path) as f:
                                    contents = f.read()
                                    load_data = [load_data for load_data in contents.split('\n') if load_data[:4] == 'Load']
                                    p_values = [int(p_value.split(' ')[1][0])*1000 for p_value in load_data]
                                    powerfactors = [float(p_value.split(' ')[2][:4]) for p_value in load_data]
                                    s_values = [p_values[i]/powerfactors[i] for i in list(range(len(p_values)))]
                                    q_values = [s_values[i]*math.sin(math.acos(powerfactors[i])) for i in list(range(len(s_values)))]
                                    f.close()

                                #extrapolate P/Q/S values
                                data['Wirkleistung Total Avg'] =  pd.DataFrame(data=p_values, index=pd.date_range('1/1/2000', periods=len(s_values),
                                                                                freq='15T')).resample('3600ms').bfill().values[:-1]
                                data['Blindleistung Total Avg'] =  pd.DataFrame(data=q_values, index=pd.date_range('1/1/2000', periods=len(s_values),
                                                                                freq='15T')).resample('3600ms').bfill().values[:-1]
                                data['Scheinleistung Total Avg'] =  pd.DataFrame(data=s_values, index=pd.date_range('1/1/2000', periods=len(s_values),
                                                                                freq='15T')).resample('3600ms').bfill().values[:-1]


                            data['timestep'] = range(len(data))
                            data = data.set_index('timestep')

                            if sampling:
                                data = self.sample(data, sampling)

                            name = str(measurement)[13:] + ' Scenario ' + str(
                                measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(test_bay)
                            relevant_measurements[
                                str(measurement)[13:] + ' Scenario ' + str(
                                    measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(
                                    test_bay)] = Measurement(
                                data, name)
            if self.data_source == 'simulation':
                for measurement in measurements:
                    for scenario in list(range(len(measurements[measurement]))):
                        for test_bay in self.test_bays:
                            full_path = os.path.join(self.sim_data_path, self.pf_file, 'Test_Bay_' + test_bay)
                            data = pd.read_csv(os.path.join(full_path,
                                                            f'scenario_{scenario + 1}_{measurement.split(" ")[1]}_control_Setup_{measurement.split(" ")[4]}.csv'),
                                               sep=',',
                                               decimal=',', low_memory=False)
                            data['timestep'] = range(len(data))
                            data = data.set_index('timestep')

                            if sampling:
                                data = self.sample(data, sampling)

                            name = str(measurement)[13:] + ' Scenario ' + str(
                                scenario + 1) + ': Test Bay ' + str(test_bay)
                            relevant_measurements[
                                str(measurement)[13:] + ' Scenario ' + str(
                                    scenario + 1) + ': Test Bay ' + str(
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

    def disaggregation(self):

        '''
        disaggregate load profile at transformer into its contributions
        '''

        data = self.load_data(sampling=self.sampling_step_size_in_seconds)
        if config.use_case == 'DSM': trafo_point = 'B2'
        else: trafo_point = 'F2'
        data = create_dataset(type='complete', data=data, variables=self.variables, name=self.setup_chosen,
                              Setup=self.setup_chosen.split('_')[1], trafo_point=trafo_point)

        scores = self.cross_val(data, sampling=self.sampling_step_size_in_seconds, disagg_algo=self.disagg_algo)
        print(
            f"\n########## Metrics for disaggregation on {self.setup_chosen}  ##########")
        for score in scores:
            print("%s: %0.2f (+/- %0.2f)" % (
                    score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

    def remove_objects_in_list_from_list(self, list, object_list):
        for i in object_list:
            list.remove(i[0])
        return list

    def scoring(self, y_test, y_pred, timestep_wise=False):

        if timestep_wise:
            mapes = []
            wmapes = []
            for timestep in list(range(y_test.shape[1])):
                mapes.append(mean_absolute_percentage_error(y_test[:, timestep], y_pred[:, timestep]))
                wmapes = np.sum(np.abs(y_test[:, timestep] - y_pred[:, timestep])) / np.sum(np.abs(y_test[:, timestep]))
            mape_avg = statistics.mean(mapes)   #average error over all timesteps
            wmapes_avg = statistics.mean(wmapes)   #average error over all timesteps

            self.scores['MAPE'] = mape_avg
            self.scores['WMAPE'] = wmapes_avg
        else:
            self.scores['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
            self.scores['WMAPE'] = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test))

        return self.scores

    def disaggregation_algorithm(self, data, cross_val=False, disagg_algo='RF', scale=False, timestep_wise=False):

        if not cross_val:
            X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        scale = False

        if disagg_algo in ['SVR']:
            scale = True
        if disagg_algo in ['SVR']:
            timestep_wise = True

        if scale:  # do prediction for each timestep individually
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        if disagg_algo == 'SVR':  # do prediction for each timestep individually
            algo = SVR(C=1.0, epsilon=0.2)
        elif disagg_algo == 'RF':
            algo = RandomForestRegressor(n_estimators=100, n_jobs=4)

        # Train the model using the training sets
        wrapper = RegressorChain(algo)  #first model in the sequence uses the input and predicts one output; the second model uses the input and the output from the first model to make a prediction; the third model uses the input and output from the first two models to make a prediction, and so on

        if timestep_wise:
            for timestep in list(range(X_train.shape[1])):
                X_train_timestep = X_train[:, timestep]
                y_train_timestep = y_train[:, timestep]
                wrapper.fit(X_train_timestep, y_train_timestep)     #train model for each timestep

            #algo.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = np.empty(y_test.shape)
            for timestep in list(range(X_test.shape[1])):
                X_test_timestep = X_test[:, timestep]
                y_pred_timestep = wrapper.predict(X_test_timestep)     #predict outputs for each timestep
                y_pred[: , timestep] = y_pred_timestep

            #y_pred = algo.predict(X_test)
        else:
            wrapper.fit(X_train, y_train)
            y_pred = wrapper.predict(X_test)  # predict outputs for each timestep

        if not cross_val:
            self.scores = {}

            print(f"\n########## Metrics for {data.name} ##########")
            print("Mean absolute percentage error: {0}\n".format(self.scores['MAPE']))

        return y_pred, y_test

    def cross_val(self, data, sampling=None, disagg_algo='SVR', splits_defined=True):

        if not splits_defined:
            X = data.X
            y = data.y
            kf = KFold(n_splits=7, shuffle=True)
            #kf = StratifiedKFold(n_splits=7,
                                 #shuffle=True)  # ensures balanced classes in batches!! (as much as possible) > important

            cv_scores = []

            for train_index, test_index in kf.split(list(X), list(y)):
                # print('Split #%d' % (len(scores) + 1))

                train_keys = [key for key in X.keys() if list(X.keys()).index(key) in train_index]
                X_train = {your_key: X[your_key] for your_key in train_keys}
                y_train = {your_key: y[your_key] for your_key in train_keys}

                test_keys = [key for key in X.keys() if list(X.keys()).index(key) in test_index]
                X_test = {your_key: X[your_key] for your_key in test_keys}
                y_test = {your_key: y[your_key] for your_key in test_keys}

                #X_train, X_test = X[train_index], X[test_index]
                #y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

                X_train = np.array(list(X_train.values()))
                X_test = np.array(list(X_test.values()))
                y_train = np.array(list(y_train.values()))
                y_test = np.array(list(y_test.values()))

                y_pred, y_test = self.disaggregation_algorithm([X_train, X_test, y_train, y_test], cross_val=True, disagg_algo=disagg_algo)

                self.scores = {}
                cv_scores.append(self.scoring(y_test, y_pred), timestep_wise=True)
                #plot them against each other?

        else:
            cv_scores = []
            for split in data.splits:
                X_train = data.splits[split]['training_data'][list(data.complete_dataset.values())[0]['X'].keys()]
                y_train = data.splits[split]['training_data'][list(data.complete_dataset.values())[0]['y'].keys()]
                X_test = data.splits[split]['testing_data'][list(data.complete_dataset.values())[0]['X'].keys()]
                y_test = data.splits[split]['testing_data'][list(data.complete_dataset.values())[0]['y'].keys()]

                y_pred, y_test = self.disaggregation_algorithm([X_train, X_test, y_train, y_test], cross_val=True,
                                                               disagg_algo=disagg_algo)

                self.scores = {}
                cv_scores.append(self.scoring(y_test, y_pred))
                # plot them against each other?



        scores_dict = {'Mean absolute percentage error': [i['MAPE'] for i in cv_scores]}

        return scores_dict
