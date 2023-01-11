from plot_measurements import plot_scenario_test_bay, plot_scenario_case, plot_pca, plot_grid_search
from Measurement import Measurement
from Clustering import Clustering
from util import create_dataset
from detection_method_settings import Variables

v = Variables()
from detection_method_settings import Classifier_Combos

c = Classifier_Combos()

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
import collections
from sklearn import svm, neighbors, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


class Transformer_detection:

    def __init__(self, config, learning_config):

        self.sim_data_path = config.sim_data_path
        self.pf_file = config.pf_file
        self.data_path = config.data_path
        self.test_bays = config.test_bays
        self.scenario = config.scenario
        self.plot_all = config.plot_all
        self.plotting_variables = config.plotting_variables
        self.variables = config.variables
        self.sampling_step_size_in_seconds = config.sampling_step_size_in_seconds
        self.setups = config.setups
        self.plot_data = config.plot_data
        if learning_config['data_mode'] == 'combined_data':
            self.classifier_combos = c.classifier_combos[learning_config['classifier_combos'] + '_combined_dataset']
        else:
            self.classifier_combos = c.classifier_combos[learning_config['classifier_combos']]

        self.data_source = learning_config['data_source']
        self.setup_chosen = learning_config['setup_chosen']
        self.mode = learning_config['mode']
        self.data_mode = learning_config['data_mode']
        self.selection = learning_config['selection']
        self.clf = learning_config['clf']
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

    def load_data(self, scenario=None, sampling=None, data_source=None, phase_info=None, grid_setup=None, marker='-'):
        if data_source is None:
            data_source = self.data_source
        if phase_info is not None:
            measurements = {key: value for key, value in phase_info[1][1].items() if key.split(' ')[-1] == grid_setup}
            if marker.split('_')[-1].split(' ')[-1] == 'estimate': measurements = {key: value for key, value in measurements.items() if key.split(' ')[1] != 'correct'}
            data_path = config.data_path_dict[phase_info[0].split('_')[-1]]
            sim_data_path = config.sim_data_path_dict[phase_info[0].split('_')[-1]]
            test_bays = config.test_bays_dict[phase_info[0].split('_')[-1]]
        else:
            test_bays = self.test_bays
            data_path = self.data_path
            sim_data_path = self.sim_data_path

        print('Data loaded with sampling of ' + str(sampling))
        relevant_measurements = {}
        if scenario:
            if data_source == 'real_world':
                # to get data of entire scenario
                for measurement in measurements:
                    for test_bay in test_bays:
                        full_path = os.path.join(data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                        data = pd.read_csv(os.path.join(full_path, measurements[measurement][scenario - 1] + '.csv'),
                                           sep=',',
                                           decimal=',', low_memory=False)
                        data = data[
                               2 * 60 * 4:]  # cut off the first 2 minutes because this is where loads / PV where started up
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
            if data_source == 'simulation':
                # to get data of entire scenario
                for measurement in measurements:
                    for test_bay in test_bays:
                        full_path = os.path.join(sim_data_path, self.pf_file, 'Test_Bay_' + test_bay)
                        if marker == 'estimated' or marker.split('_')[-1].split(' ')[-1] == 'estimate':
                            data = pd.read_csv(os.path.join(full_path,
                                                            f'scenario_{scenario}_{measurement.split(" ")[1]}_control_Setup_{grid_setup}_{marker}.csv'),
                                               sep=',',
                                               decimal=',', low_memory=False)
                        else:
                            data = pd.read_csv(os.path.join(full_path,
                                                            f'scenario_{scenario}_{measurement.split(" ")[1]}_control_Setup_{measurement.split(" ")[4]}.csv'),
                                               sep=',',
                                               decimal=',', low_memory=False)
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
            if data_source == 'real_world':
                for measurement in measurements:
                    for scenario in measurements[measurement]:
                        for test_bay in test_bays:
                            full_path = os.path.join(data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                            data = pd.read_csv(
                                os.path.join(full_path, measurements[measurement][
                                    measurements[measurement].index(scenario)] + '.csv'),
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
            if data_source == 'simulation':
                for measurement in measurements:
                    for scenario in list(range(len(measurements[measurement]))):
                        for test_bay in test_bays:
                            full_path = os.path.join(sim_data_path, self.pf_file, 'Test_Bay_' + test_bay)
                            if marker == 'estimated' or marker.split('_')[-1].split(' ')[-1] == 'estimate':
                                data = pd.read_csv(os.path.join(full_path,
                                                                f'scenario_{scenario + 1}_{measurement.split(" ")[1]}_control_Setup_{grid_setup}_{marker}.csv'),
                                                   sep=',',
                                                   decimal=',', low_memory=False)
                            else:
                                data = pd.read_csv(os.path.join(full_path,
                                                                f'scenario_{scenario + 1}_{measurement.split(" ")[1]}_control_Setup_{measurement.split(" ")[4]}.csv'),
                                                   sep=',',
                                                   decimal=',', low_memory=False)
                            data['new_index'] = range(len(data))
                            data = data.set_index('new_index')

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

    def clustering(self):
        data = self.load_data(sampling=self.sampling_step_size_in_seconds)
        dataset = create_dataset(type='raw', data=data, name=self.setup_chosen, classes=self.setups[self.setup_chosen],
                                 bay=self.setup_chosen.split('_')[2],
                                 Setup=self.setup_chosen.split('_')[1], labelling=self.mode)

        Clustering_ward = Clustering(data=dataset, variables=self.variables[self.setup_chosen.split('_')[2]][1],
                                     metric='euclidean', method='ward',
                                     num_of_clusters=len(self.setups[self.setup_chosen]))
        cluster_assignments = Clustering_ward.cluster()
        score = Clustering_ward.rand_score()
        print(f'Score reached: {score}')

    def detection(self):

        if self.data_mode == 'measurement_wise':
            results_pca = self.pca(variables=self.variables, PCA_type='PCA', analyse=True,
                                   sampling=self.sampling_step_size_in_seconds)
            variable_selection = self.find_most_common_PCs(
                results_pca)  # , number_of_variables = 15)   #returns the most important variables for the PCA per measurement point > use to to do PCA and then SVM
            if self.plot_data:
                fgs_pca, axs_pca = self.pca_plotting(results_pca, type='PCA',
                                                     number_of_vars=len(self.variables['B1'][1]))

            if self.selection == 'most important':
                variable_selection = {'B1': [v.variables_B1, [i[0] for i in variable_selection[0]]],
                                      'F1': [v.variables_F1, [i[0] for i in variable_selection[1]]],
                                      'F2': [v.variables_F2, [i[0] for i in variable_selection[2]]]}
            if self.selection == 'least important':
                pca_variables_B1 = self.remove_objects_in_list_from_list(v.pca_variables_B1, variable_selection[0])
                pca_variables_F1 = self.remove_objects_in_list_from_list(v.pca_variables_F1, variable_selection[1])
                pca_variables_F2 = self.remove_objects_in_list_from_list(v.pca_variables_F2, variable_selection[2])
                variable_selection = {'B1': [v.variables_B1, pca_variables_B1],
                                      'F1': [v.variables_F1, pca_variables_F1],
                                      'F2': [v.variables_F2, pca_variables_F2]}

            if self.plot_data:
                results_pca = self.pca(variables=variable_selection, PCA_type='PCA',
                                       sampling=self.sampling_step_size_in_seconds)
                fgs_pca, axs_pca = self.pca_plotting(results_pca, type='PCA', number_of_vars=max(
                    [len(variable_selection['B1'][1]), len(variable_selection['F1'][1]),
                     len(variable_selection['F2'][1])]))

        # results_pca = pca(variables=variable_selection, type='PCA', n_components=len(variable_selection['B1'][1]))
        scores_by_n = {}

        # learning settings
        # clf = 'Assembly'  # SVM, NuSVM, kNN, Assembly
        # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        # kernels = ['poly']
        if self.kernels[0] == 'poly' and len(self.kernels) == 1:
            degrees = self.degrees
        else:
            degrees = [3]
        # gammas = ['scale']  # , 'auto']#[1/(i+1) for i in range(15)] #['scale', 'auto']
        # max_number_of_components = 2    #means 1
        # classifiers = {'SVM': {'poly': [8]}, 'NuSVM': {'linear': [9], 'poly': [11], 'rbf': [2]}, 'kNN': {3: [18,'uniform']}}
        # classifiers = {'NuSVM': {'linear': [11], 'poly': [3], 'rbf': [10]}}
        # classifiers = {'NuSVM': {'poly': [2]}}

        if self.clf != 'Assembly' and self.data_mode != 'combined_data':
            max_number_of_components = len(variable_selection['F2'][1])  # one less than variables due to range
            for n in range(1, max_number_of_components):

                print(f'\nPCA done with {n} components\n')
                results_pca = self.pca(variables=variable_selection, PCA_type='PCA', n_components=n,
                                       sampling=self.sampling_step_size_in_seconds)

                Setup_A_F2_Data = create_dataset(type='pca', data=results_pca, name='Setup_A_F2_data',
                                                 classes=self.setups['Setup_A_F2_data'],
                                                 bay='F2', Setup='A')

                Setup_B_F2_Data1_3c = create_dataset(type='pca', data=results_pca, name='Setup_B_F2_data1_3c',
                                                     classes=self.setups['Setup_B_F2_data1_3c'], bay='F2', Setup='B',
                                                     labelling=self.mode)  # ['correct', 'wrong', 'inversed']
                Setup_B_F2_Data2_2c = create_dataset(type='pca', data=results_pca, name='Setup_B_F2_data2_2c',
                                                     classes=self.setups['Setup_B_F2_data2_2c'],
                                                     bay='F2', Setup='B')
                Setup_B_F2_Data3_2c = create_dataset(type='pca', data=results_pca, name='Setup_B_F2_data3_2c',
                                                     classes=self.setups['Setup_B_F2_data3_2c'],
                                                     bay='F2', Setup='B')

                datasets = []
                datasets.append(Setup_A_F2_Data)
                datasets.append(Setup_B_F2_Data1_3c)
                datasets.append(Setup_B_F2_Data2_2c)
                datasets.append(Setup_B_F2_Data3_2c)

                scores_by_dataset = {}

                for dataset in datasets:

                    # SVM
                    results_svm = self.svm_algorithm(dataset)
                    # kNN
                    results_kNN = self.kNN_algorithm(dataset)

                    print("\n########## k-fold Cross-validation ##########")
                    if self.clf in ['SVM', 'NuSVM']:
                        scores_by_kernel = {}
                        for kernel in self.kernels:
                            scores_by_degree = {}
                            for degree in degrees:
                                scores = self.cross_val(dataset, clf=self.clf, kernel=kernel, degree=degree,
                                                        sampling=self.sampling_step_size_in_seconds)
                                if dataset.labelling == 'classification':
                                    print(
                                        f"\n########## Metrics for {self.clf} classifier applied on {dataset.name} using a {kernel} kernel of degree {degree} with classes {dataset.classes} ##########")
                                elif dataset.labelling == 'detection':
                                    print(
                                        f"\n########## Metrics for {self.clf} classifier applied on {dataset.name} using a {kernel} kernel of degree {degree} with classes normal and abnormal ##########")
                                for score in scores:
                                    print("%s: %0.2f (+/- %0.2f)" % (
                                        score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))
                                scores_by_degree[degree] = scores
                            scores_by_kernel[kernel] = scores_by_degree
                        scores_by_dataset[dataset.name] = scores_by_kernel
                    if self.clf in ['kNN']:
                        scores_by_neighbours = {}
                        for number in self.neighbours:
                            scores_by_weights = {}
                            for weight in self.weights:
                                scores = self.cross_val(dataset, neighbours=number, weights=weight,
                                                        sampling=self.sampling_step_size_in_seconds)
                                if dataset.labelling == 'classification':
                                    print(
                                        f"\n########## Metrics for {self.clf} classifier applied on {dataset.name} using {number} neigbours and {weight} weights with classes {dataset.classes} ##########")
                                elif dataset.labelling == 'detection':
                                    print(
                                        f"\n########## Metrics for {self.clf} classifier applied on {dataset.name} using {number} neigbours with and {weight} weights classes normal and abnormal ##########")
                                for score in scores:
                                    print("%s: %0.2f (+/- %0.2f)" % (
                                        score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))
                                    scores_by_weights[weight] = scores
                            scores_by_neighbours[number] = scores_by_weights
                        scores_by_dataset[dataset.name] = scores_by_neighbours
                scores_by_n[str(n)] = scores_by_dataset

            for dataset in self.setups.keys():
                if self.clf in ['SVM', 'NuSVM']:
                    for kernel in self.kernels:
                        for degree in degrees:
                            if kernel == 'poly':
                                poly_message = f'of degree {degree}'
                            else:
                                poly_message = ''
                            if self.mode == 'classification' or dataset.split('_')[-1] != '3c':
                                title = f'{self.clf} on {" ".join(dataset.split("_")[:4])} using a {kernel} kernel {poly_message} with classes {self.setups[dataset]}'
                            elif self.mode == 'detection' and dataset.split('_')[-1] == '3c':
                                title = f'{self.clf} on {" ".join(dataset.split("_")[:4])} using a {kernel} kernel {poly_message} with classes normal and abnormal'
                            fig_svm, ax_svm = plot_grid_search(list(range(1, max_number_of_components)),
                                                               [scores_by_n[i][dataset][kernel][degree] for i in
                                                                scores_by_n],
                                                               title=title)
                if self.clf in ['kNN']:
                    for number in self.neighbours:
                        for weight in self.weights:
                            if self.mode == 'classification' or dataset.split('_')[-1] != '3c':
                                title = f'{self.clf} on {" ".join(dataset.split("_")[:4])} using {number} neigbours and {weight} weights with classes {self.setups[dataset]}'
                            elif self.mode == 'detection' and dataset.split('_')[-1] == '3c':
                                title = f'{self.clf} on {" ".join(dataset.split("_")[:4])} using {number} neigbours and {weight} weights with classes normal and abnormal'
                            fig_svm, ax_svm = plot_grid_search(list(range(1, max_number_of_components)),
                                                               [scores_by_n[i][dataset][number][weight] for i in
                                                                scores_by_n],
                                                               title=title)

        if self.clf == 'Assembly' and self.data_mode != 'combined_data':
            for classifiers in self.classifier_combos:
                scores = self.cross_val(variable_selection, clf=self.clf, classifiers_and_parameters=classifiers,
                                        setup=self.setup_chosen,
                                        mode=self.mode, sampling=self.sampling_step_size_in_seconds)
                if self.mode == 'classification':
                    print(
                        f"\n########## Metrics for {self.clf} classifier on Setup_B_F2_data1_3c using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes {self.setups[self.setup_chosen]} ##########")
                elif self.mode == 'detection':
                    print(
                        f"\n########## Metrics for {self.clf} classifier on Setup_B_F2_data1_3c using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes normal and abnormal ##########")
                for score in scores:
                    print("%s: %0.2f (+/- %0.2f)" % (
                        score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

        if self.data_mode == 'combined_data':
            '''
            uses principal components instead of explained variances!
            '''
            data = self.load_data(sampling=self.sampling_step_size_in_seconds)
            if config.use_case == 'DSM': trafo_point = 'B2'
            else: trafo_point = 'F2'
            data = create_dataset(type='combined', data=data, variables=self.variables[trafo_point][1], name=self.setup_chosen,
                                  classes=self.setups[self.setup_chosen],
                                  bay=self.setup_chosen.split('_')[2], Setup=self.setup_chosen.split('_')[1],
                                  labelling=self.mode)
            '''scaled_data = data.scale()
            pca_data = data.PCA()
            labelled_data = data.label()'''

            for classifiers in self.classifier_combos:
                scores = self.cross_val(data, clf=self.clf, classifiers_and_parameters=classifiers,
                                        setup=self.setup_chosen,
                                        mode=self.mode, sampling=self.sampling_step_size_in_seconds,
                                        data_mode=self.data_mode)
                if self.mode == 'classification':
                    print(
                        f"\n########## Metrics for {self.clf} classifier on {self.setup_chosen} using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes {self.setups[self.setup_chosen]} ##########")
                elif self.mode == 'detection':
                    print(
                        f"\n########## Metrics for {self.clf} classifier on {self.setup_chosen} using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes normal and abnormal ##########")
                for score in scores:
                    print("%s: %0.2f (+/- %0.2f)" % (
                        score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

        ##########

        # results_kpca = pca(variables=variables, PCA_type='kPCA', sampling=sampling_step_size_in_seconds)
        # fgs_kpca, axs_kpca = pca_plotting(results_kpca, type='kPCA')

    # results_ssa = ssa()

    def pca(self, variables=None, PCA_type='PCA', analyse=False, n_components=2, data=None, sampling=None):
        if learning_config['data_source'] == 'simulation':
            if data is None:
                data = self.load_data(sampling=sampling)
                variables = {'B1': [v.variables_B1, list(data[list(data.keys())[0][:-2] + 'B1'].data.columns)[1:]],
                             'F1': [v.variables_F1, list(data[list(data.keys())[0][:-2] + 'F1'].data.columns)[1:]],
                             'F2': [v.variables_F2, list(data[list(data.keys())[0][:-2] + 'F2'].data.columns)[1:]]}
                results = {}
            else:
                results = []
        else:
            if variables is None:
                variables = {'B1': [v.variables_B1, ['Vrms ph-n AN Avg', 'Vrms ph-n BN Avg', 'Vrms ph-n CN Avg']],
                             'F1': [v.variables_F1, ['Vrms ph-n AN Avg', 'Vrms ph-n BN Avg', 'Vrms ph-n CN Avg']],
                             'F2': [v.variables_F2, ['Vrms ph-n L1N Avg', 'Vrms ph-n L2N Avg', 'Vrms ph-n L3N Avg']]}

            if data is None:
                data = self.load_data(sampling=sampling)
                results = {}
            else:
                results = []

        for measurement in data:
            if learning_config['data_source'] == 'simulation':
                var_numbers = [1, 2, 3]  # irrelevant in this case
            else:
                try:
                    if type(data) is dict:
                        var_numbers = [variables[data[measurement].name[-2:]][0].index(i) + 1 for i in
                                       variables[data[measurement].name[-2:]][1]]
                    else:
                        var_numbers = [variables[measurement.name[-2:]][0].index(i) + 1 for i in
                                       variables[measurement.name[-2:]][1]]
                except ValueError:
                    [print(f"Variable {i} not available") for i in variables[data[measurement].name[-2:]][1] if
                     i not in variables[data[measurement].name[-2:]][0]]

            if type(data) is dict:
                if PCA_type == 'PCA':
                    results[f"{data[measurement].name}"] = data[measurement].pca(
                        variables[data[measurement].name[-2:]][1],
                        var_numbers, analyse=analyse,
                        n_components=n_components)
                elif PCA_type == 'kPCA':
                    results[f"{data[measurement].name}"] = data[measurement].kpca(
                        variables[data[measurement].name[-2:]][1],
                        var_numbers)
                else:
                    print('Unknown type of PCA enterered (enter either PCA or kPCA)')
            else:
                if PCA_type == 'PCA':
                    results.append(measurement.pca(variables[measurement.name[-2:]][1], var_numbers, analyse=analyse,
                                                   n_components=n_components)[1])
                elif PCA_type == 'kPCA':
                    results.append(measurement.kpca(variables[measurement.name[-2:]][1],
                                                    var_numbers)[1])
                else:
                    print('Unknown type of PCA enterered (enter either PCA or kPCA)')

        if type(results) is list:
            results = np.array(results)

        return results

    def find_most_common_PCs(self, results_pca):  # , number_of_variables=15):
        """results_B1 = [print(
            key + ': #components: ' + str(results_pca[key][2]) + '; most important components: ' + str(results_pca[key][3]))
                      for key in results_pca if key[-2:] == 'B1']"""

        min_number_of_dimensions_B1 = min([results_pca[i][2] for i in results_pca if i[
                                                                                     -2:] == 'B1'])  # get lowest number of dimensions needed to capture 99% of variance
        number_of_variables_B1 = min_number_of_dimensions_B1

        min_number_of_dimensions_F1 = min([results_pca[i][2] for i in results_pca if i[
                                                                                     -2:] == 'F1'])  # get lowest number of dimensions needed to capture 99% of variance
        number_of_variables_F1 = min_number_of_dimensions_F1

        min_number_of_dimensions_F2 = min([results_pca[i][2] for i in results_pca if i[
                                                                                     -2:] == 'F2'])  # get lowest number of dimensions needed to capture 99% of variance
        number_of_variables_F2 = min_number_of_dimensions_F2

        most_common_B1 = []
        least_common_B1 = []
        for most_important in [results_pca[key][3] for key in results_pca if key[-2:] == 'B1']:
            most_common_B1 = most_common_B1 + most_important[:number_of_variables_B1]
            # least_common_B1 = most_common_B1 + most_important[:number_of_variables_B1]
        a_counter = collections.Counter(most_common_B1)
        most_common_B1 = a_counter.most_common(number_of_variables_B1)
        print('most common most important variables for PCA for B1: ' + str(most_common_B1))

        """results_F1 = [print(
            key + ': #components: ' + str(results_pca[key][2]) + '; most important components: ' + str(results_pca[key][3]))
                      for key in results_pca if key[-2:] == 'F1']"""
        most_common_F1 = []
        for most_important in [results_pca[key][3] for key in results_pca if key[-2:] == 'F1']:
            most_common_F1 = most_common_F1 + most_important[:number_of_variables_F1]
        a_counter = collections.Counter(most_common_F1)
        most_common_F1 = a_counter.most_common(number_of_variables_F1)
        print('most common most important variables for PCA for F1: ' + str(most_common_F1))

        """results_F2 = [print(
            key + ': #components: ' + str(results_pca[key][2]) + '; most important components: ' + str(results_pca[key][3]))
                      for key in results_pca if key[-2:] == 'F2']"""
        most_common_F2 = []
        for most_important in [results_pca[key][3] for key in results_pca if key[-2:] == 'F2']:
            most_common_F2 = most_common_F2 + most_important[:number_of_variables_F2]
        a_counter = collections.Counter(most_common_F2)
        most_common_F2 = a_counter.most_common(number_of_variables_F2)
        print('most common most important variables for PCA for F2: ' + str(
            most_common_F2))  # each occurence means that the variable is the most important for a component of the PCA

        return most_common_B1, most_common_F1, most_common_F2

    def pca_plotting(self, results, type='PCA', number_of_vars=len(v.pca_variables_B1)):
        explained_variances = {}

        for test_bay in self.test_bays:
            for measurement in measurements:
                explained_variances[str(measurement)[13:] + ': Test Bay ' + str(test_bay)] = []
                for scenario in measurements[measurement]:
                    explained_variances[str(measurement)[13:] + ': Test Bay ' + str(test_bay)].append(
                        results[str(measurement)[13:] + ' Scenario ' + str(
                            measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(test_bay)][1])

        fgs, axs = plot_pca(explained_variances, type=type, number_of_vars=number_of_vars)

        return fgs, axs

    def remove_objects_in_list_from_list(self, list, object_list):
        for i in object_list:
            list.remove(i[0])
        return list

    def svm_algorithm(self, data, SVM_type='SVM', cross_val=False, kernel='linear', gamma='scale', degree=3):
        if not cross_val:
            X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        # Create a svm Classifier
        if SVM_type == 'SVM':
            clf = svm.SVC(kernel=kernel, gamma=gamma, degree=degree)  # default Linear Kernel
        elif SVM_type == 'NuSVM':
            clf = svm.NuSVC(kernel=kernel,
                            gamma=gamma,
                            degree=degree)  # Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors.

        # Train the model using the training sets
        clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        if not cross_val:
            scores = self.scoring(y_test, y_pred)

            print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
            print(f"\n########## Metrics for {data.name} ##########")
            print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1],
                                                                                     scores[1][2],
                                                                                     scores[1][3]))

        return y_pred, y_test

    def kNN_algorithm(self, data, cross_val=False, neighbours=2, weights='uniform'):
        if not cross_val:
            X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        # Create a kNN Classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors=neighbours, weights=weights)

        # Train the model using the training sets
        clf.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        if not cross_val:
            scores = self.scoring(y_test, y_pred)

            print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
            print(f"\n########## Metrics for {data.name} ##########")
            print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1],
                                                                                     scores[1][2],
                                                                                     scores[1][3]))

        return y_pred, y_test

    def scoring(self, y_test, y_pred):
        metrics = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        return [accuracy, metrics]

    def cross_val(self, data, clf='SVM', kernel='linear', neighbours=2, weights='uniform', degree=3,
                  classifiers_and_parameters=None,
                  setup='Setup_B_F2_data1_3c', mode='classification', sampling=None, data_mode='measurement_wise'):
        if classifiers_and_parameters is None:
            classifiers_and_parameters = {'SVM': {'poly': [8]}, 'NuSVM': {'linear': [9], 'poly': [11], 'rbf': [2]},
                                          'kNN': {3: [18, 'uniform']}}
        if clf != 'Assembly' or data_mode == 'combined_data':
            X = data.X
            y = data.y
            # kf = KFold(n_splits=7, shuffle=True)
            kf = StratifiedKFold(n_splits=7,
                                 shuffle=True)  # ensures balanced classes in batches!! (as much as possible) > important
        if clf == 'Assembly' and data_mode == 'measurement_wise':
            variables = data
            data = self.load_data(sampling=sampling)
            raw_dataset = create_dataset(type='raw', data=data, name=setup, classes=self.setups[setup],
                                         bay=setup.split('_')[2],
                                         Setup=setup.split('_')[1], labelling=mode)
            X = raw_dataset.X
            y = raw_dataset.y
            kf = StratifiedKFold(n_splits=7,
                                 shuffle=True)  # ensures balanced classes in batches!! (as much as possible) > important
        scores = []

        for train_index, test_index in kf.split(X, y):
            # print('Split #%d' % (len(scores) + 1))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

            if clf == 'SVM' or clf == 'NuSVM':
                y_pred, y_test = self.svm_algorithm([X_train, X_test, y_train, y_test], SVM_type=clf, cross_val=True,
                                                    kernel=kernel, degree=degree)
                scores.append(self.scoring(y_test, y_pred))
            elif clf == 'kNN':
                y_pred, y_test = self.kNN_algorithm([X_train, X_test, y_train, y_test], cross_val=True,
                                                    neighbours=neighbours,
                                                    weights=weights)
                scores.append(self.scoring(y_test, y_pred))
            elif clf == 'Assembly':
                if data_mode == 'measurement_wise':
                    y_pred, y_test = self.assembly_learner_single_dataset([X_train, X_test, y_train, y_test],
                                                                          classifiers_and_parameters, cross_val=True,
                                                                          variables=variables)
                elif data_mode == 'combined_data':
                    y_pred, y_test = self.assembly_learner_combined_dataset([X_train, X_test, y_train, y_test],
                                                                            classifiers_and_parameters, cross_val=True)
                scores.append(self.scoring(y_test, y_pred))
            else:
                print('undefined classifier entered')

        scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores],
                       'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores]}

        return scores_dict

    def assembly_learner_single_dataset(self, data, clf_types_and_paras, cross_val=False, variables=None):
        if not cross_val:
            X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        clfs = {}
        y_preds = {}
        for clf_type in clf_types_and_paras.keys():

            # Create a classifier & train the model using the training sets
            if clf_type == 'SVM':
                for kernel in clf_types_and_paras[clf_type]:
                    X_train_pca = self.pca(variables=variables, PCA_type='PCA',
                                           n_components=clf_types_and_paras[clf_type][kernel][0], data=X_train,
                                           sampling=self.sampling_step_size_in_seconds)
                    X_test_pca = self.pca(variables=variables, PCA_type='PCA',
                                          n_components=clf_types_and_paras[clf_type][kernel][0], data=X_test,
                                          sampling=self.sampling_step_size_in_seconds)
                    if kernel == 'poly':
                        clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel,
                                                                degree=clf_types_and_paras[clf_type][kernel][
                                                                    1])  # default Linear Kernel
                    else:
                        clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel)
                    clfs[clf_type + '_' + kernel].fit(X_train_pca, y_train)
                    y_preds[clf_type + '_' + kernel] = clfs[clf_type + '_' + kernel].predict(
                        X_test_pca)  # Predict the response for test dataset
            elif clf_type == 'NuSVM':
                for kernel in clf_types_and_paras[clf_type]:
                    X_train_pca = self.pca(variables=variables, PCA_type='PCA',
                                           n_components=clf_types_and_paras[clf_type][kernel][0], data=X_train,
                                           sampling=self.sampling_step_size_in_seconds)
                    X_test_pca = self.pca(variables=variables, PCA_type='PCA',
                                          n_components=clf_types_and_paras[clf_type][kernel][0], data=X_test,
                                          sampling=self.sampling_step_size_in_seconds)
                    if kernel == 'poly':
                        clfs[clf_type + '_' + kernel] = svm.NuSVC(
                            kernel=kernel, degree=clf_types_and_paras[clf_type][kernel][
                                1])  # Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors.
                    else:
                        clfs[clf_type + '_' + kernel] = svm.NuSVC(
                            kernel=kernel)  # Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors.
                    clfs[clf_type + '_' + kernel].fit(X_train_pca, y_train)
                    y_preds[clf_type + '_' + kernel] = clfs[clf_type + '_' + kernel].predict(
                        X_test_pca)  # Predict the response for test dataset
            elif clf_type == 'kNN':
                for neighbours in clf_types_and_paras[clf_type]:
                    X_train_pca = self.pca(variables=variables, PCA_type='PCA',
                                           n_components=clf_types_and_paras[clf_type][neighbours][0],
                                           data=X_train, sampling=self.sampling_step_size_in_seconds)
                    X_test_pca = self.pca(variables=variables, PCA_type='PCA',
                                          n_components=clf_types_and_paras[clf_type][neighbours][0],
                                          data=X_test, sampling=self.sampling_step_size_in_seconds)
                    clfs[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        1] + '_weights'] = neighbors.KNeighborsClassifier(n_neighbors=neighbours,
                                                                          weights=
                                                                          clf_types_and_paras[clf_type][neighbours][
                                                                              1])
                    clfs[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        1] + '_weights'].fit(X_train_pca, y_train)
                    y_preds[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        1] + '_weights'] = clfs[
                        clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                            1] + '_weights'].predict(
                        X_test_pca)  # Predict the response for test dataset

        y_pred = []
        for index in list(range(len(y_test))):
            y_pred.append(max(set([i[index] for i in y_preds.values()]),
                              key=[i[index] for i in
                                   y_preds.values()].count))  # pick class that's most commonly predicted
        y_pred = np.array(y_pred)

        if not cross_val:
            scores = self.scoring(y_test, y_pred)

            print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
            print(f"\n########## Metrics for {data.name} ##########")
            print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1],
                                                                                     scores[1][2],
                                                                                     scores[1][3]))

        return y_pred, y_test

    def assembly_learner_combined_dataset(self, data, clf_types_and_paras, cross_val=False):
        if not cross_val:
            X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        clfs = {}
        y_preds = {}
        for clf_type in clf_types_and_paras.keys():

            # Create a classifier & train the model using the training sets
            if clf_type == 'SVM':
                for kernel in clf_types_and_paras[clf_type]:
                    if kernel == 'poly':
                        clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel,
                                                                degree=clf_types_and_paras[clf_type][kernel][
                                                                    0])  # default Linear Kernel
                    else:
                        clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel)
                    clfs[clf_type + '_' + kernel].fit(X_train, y_train)
                    y_preds[clf_type + '_' + kernel] = clfs[clf_type + '_' + kernel].predict(
                        X_test)  # Predict the response for test dataset
            elif clf_type == 'NuSVM':
                for kernel in clf_types_and_paras[clf_type]:
                    if kernel == 'poly':
                        clfs[clf_type + '_' + kernel] = svm.NuSVC(
                            kernel=kernel, degree=clf_types_and_paras[clf_type][kernel][
                                0])  # Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors.
                    else:
                        clfs[clf_type + '_' + kernel] = svm.NuSVC(
                            kernel=kernel)  # Nu-Support Vector Classification. Similar to SVC but uses a parameter to control the number of support vectors.
                    clfs[clf_type + '_' + kernel].fit(X_train, y_train)
                    y_preds[clf_type + '_' + kernel] = clfs[clf_type + '_' + kernel].predict(
                        X_test)  # Predict the response for test dataset
            elif clf_type == 'kNN':
                for neighbours in clf_types_and_paras[clf_type]:
                    clfs[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        0] + '_weights'] = neighbors.KNeighborsClassifier(n_neighbors=neighbours,
                                                                          weights=
                                                                          clf_types_and_paras[clf_type][neighbours][
                                                                              0])
                    clfs[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        0] + '_weights'].fit(X_train, y_train)
                    y_preds[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        0] + '_weights'] = clfs[
                        clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                            0] + '_weights'].predict(
                        X_test)  # Predict the response for test dataset
            elif clf_type == 'DT':
                for criterion in clf_types_and_paras[clf_type]:
                    clfs[clf_type + '_' + str(criterion)] = tree.DecisionTreeClassifier(criterion=criterion)
                    clfs[clf_type + '_' + str(criterion)].fit(X_train, y_train)
                    y_preds[clf_type + '_' + str(criterion)] = clfs[clf_type + '_' + str(criterion)].predict(
                        X_test)  # Predict the response for test dataset

        y_pred = []
        for index in list(range(len(y_test))):
            y_pred.append(max(set([i[index] for i in y_preds.values()]),
                              key=[i[index] for i in
                                   y_preds.values()].count))  # pick class that's most commonly predicted
        y_pred = np.array(y_pred)

        if not cross_val:
            scores = self.scoring(y_test, y_pred)

            print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
            print(f"\n########## Metrics for {data.name} ##########")
            print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1],
                                                                                     scores[1][2],
                                                                                     scores[1][3]))

        return y_pred, y_test
