"""
Author:
    David Fellner (no Software engineer by training, so please don't get enraged)
Description:
    Set settings for QDS (quasi dynamic load flow simulation in a power grid) and elements and save results to file to
    create a data set executing a QDS.
    At first the grid is prepared and scenario settings are set. Then samples are created from raw data.
    These samples are time series of voltages at points of connection of households and photovoltaics (PVs) of a low
    voltage distribution network. Finally a deep learning approach is compared to a linear classifier to either
    determine if a sample is from a term with PV (class 1) or no PV (class 0) or from a term with a regularly
    behaving PV (class 0) or a PV with a malfunctioning reactive power control curve (class 1).
    Additionally a dummy dataset can be created that only consists of samples that are constant over the entire
    timeseries (class 0) and samples that are not (class 1). Randomly chosen samples of either classes are plotted
    along with execution at default.
    See framework diagrams for a better overview.
    Test files are in the project folder.

    Choose experiment (dataset and learning settings) in experiment_config.py
    Predefined experiments vary the dataset type (dummy, PV vs no PV, regular PV vs malfunctioning PV) as well as the
    timeseries length of samples (1 day vs 1 week) and the number of samples (too little, 'just enough', sufficient to
    produce a meaningful output after training with the basic network design used, i.e. no Fscore ill defined because only
    always one class predicted in any run of cross validation; note that 1 day vs 7 days also means increasing the amount
    of data points, therefore redundant experiments (i.e. increasing the sample number even more for 1 day timeseries
    experiments was neglected to allow for a better orientation between experiments)
    The experiment also defines the network architecture (in the predefined experiments this is a simple 2 layer Elman
    RNN with 6 hidden nodes in each layer). Multiple options are available such as changing the mini batch size, early
    stopping, warm up, controlling the learning rate...

    Metrics: Deep learning approach should perform better than linear classifier (which just guesses between 0 and 1 class)
             meaning that a higher Fscore should be achieved
             Experiment configs state if this goal can be fulfilled with the experiment settings

                    Task      Dataset collection  ANN design  ANN tuning  Results     Report      Presentation
      Time planned: (Hours)   15                  7.5         15          7.5         10          4
      Time spent:   (Hours)   ~20                 25          ~15         5             to be seen
      Conclusion:   It took much longer than planned to actually get the RNN running and producing meaningful outputs
"""
import importlib

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config
import plotting

if not config.raw_data_available:
    from start_powerfactory import start_powerfactory
    from raw_data_generation.grid_preparation import prepare_grid
    from raw_data_generation.data_creation import create_data

from dataset_creation.create_instances import create_samples
from malfunctions_in_LV_grid_dataset import MlfctinLVdataset
from PV_noPV_dataset import PVnoPVdataset
from dummy_dataset import Dummydataset
from util import load_model, export_model, save_model, plot_samples, choose_best, save_result, create_dataset
from deeplearning import Deeplearning

import pandas as pd
import os

#muss noch überarbeitet werden
from extract_measurements import extract_data
from plot_measurements import plot_scenario_test_bay, plot_scenario_case, plot_pca, plot_grid_search
from Measurement import Measurement
from Clustering import Clustering
from detection_method_settings import Variables
v = Variables()
from detection_method_settings import Classifier_Combos
c = Classifier_Combos()
from detection_method_settings import measurements

import numpy as np
import os
import matplotlib.pyplot as plt
import collections
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

def generate_deeplearning_raw_data():

    for file in os.listdir(config.grid_data_folder):
        if os.path.isdir(os.path.join(config.grid_data_folder, file)):
            print('Creating data using the grid %s' % file)
            app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
            grid_data = prepare_grid(app, file, o_ElmNet)

            create_data(app, o_ElmNet, grid_data, study_case_obj, file)
            print('Done with grid %s' % file)

    print('Done with all grids')

    return

def generate_detectionmethods_raw_data():
    '''
    USE PNDC GRID MODEL HERE
    :return:
    '''

    print('Done with all grids')

    return


############################
def sample(data, sampling):
    datetimeindex = pd.DataFrame(columns=['Datetime'], data=pd.to_datetime(data['Datum'] + ' ' + data['Zeit']))
    data = pd.concat((data, datetimeindex), axis=1)
    data = data.set_index('Datetime')
    # data = data.drop(['Datum', 'Zeit'], axis=1)
    data.resample(str(sampling) + 'S')
    index = pd.DataFrame(index=datetimeindex['Datetime'], columns=['new_index'], data=range(len(data)))
    data = pd.concat((data, index), axis=1)
    sampled_data = data.set_index('new_index')

    return sampled_data


def load_data(scenario=None, sampling=None):
    print('Data loaded with sampling of ' + str(sampling))
    relevant_measurements = {}
    if scenario:
        # to get data of entire scenario
        for measurement in measurements:
            for test_bay in test_bays:
                full_path = os.path.join(data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
                data = pd.read_csv(os.path.join(full_path, measurements[measurement][scenario - 1] + '.csv'), sep=',',
                                   decimal=',', low_memory=False)
                data = data[
                       2 * 60 * 4:]  # cut off the first 2 minutes because this is where laods / PV where started up
                data = data[
                       :6000]  # cut off after 25 minutes (25*60*4 bc 4 samples per second) because measurements were not turned off at same time
                data['new_index'] = range(len(data))
                data = data.set_index('new_index')

                if sampling:
                    data = sample(data, sampling)

                name = str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(test_bay)
                relevant_measurements[
                    str(measurement)[13:] + ' Scenario ' + str(scenario) + ': Test Bay ' + str(test_bay)] = Measurement(
                    data, name)
    else:
        # get all data
        for measurement in measurements:
            for scenario in measurements[measurement]:
                for test_bay in test_bays:
                    full_path = os.path.join(data_path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
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
                        data = sample(data, sampling)

                    name = str(measurement)[13:] + ' Scenario ' + str(
                        measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(test_bay)
                    relevant_measurements[
                        str(measurement)[13:] + ' Scenario ' + str(
                            measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(
                            test_bay)] = Measurement(
                        data, name)

    return relevant_measurements


def scenario_plotting_test_bay(variables, plot_all=True, scenario=1, vars=None, sampling=None):
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

    vars = {'B1': (vars['B1'], var_numbers[0]), 'F1': (vars['F1'], var_numbers[1]), 'F2': (vars['F2'], var_numbers[2])}
    if plot_all:
        for scenario in range(1, 16):
            relevant_measurements = load_data(scenario, sampling=sampling)
            fgs, axs = plot_scenario_test_bay(relevant_measurements, fgs, axs, vars)
    else:
        relevant_measurements = load_data(scenario, sampling=sampling)
        fgs, axs = plot_scenario_test_bay(relevant_measurements, fgs, axs, vars)

    return fgs, axs


def scenario_plotting_case(variables, plot_all=True, scenario=1, vars=None, sampling=None):
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

    vars = {'B1': (vars['B1'], var_numbers[0]), 'F1': (vars['F1'], var_numbers[1]), 'F2': (vars['F2'], var_numbers[2])}
    if plot_all:
        for scenario in range(1, 16):
            relevant_measurements = load_data(scenario, sampling=sampling)
            fgs, axs = plot_scenario_case(relevant_measurements, fgs, axs, vars)
    else:
        relevant_measurements = load_data(scenario, sampling=sampling)
        fgs, axs = plot_scenario_case(relevant_measurements, fgs, axs, vars)

    return fgs, axs


def pca(variables=None, PCA_type='PCA', analysis=False, n_components=2, data=None, sampling=None):
    if variables is None:
        variables = {'B1': [v.variables_B1, ['Vrms ph-n AN Avg', 'Vrms ph-n BN Avg', 'Vrms ph-n CN Avg']],
                     'F1': [v.variables_F1, ['Vrms ph-n AN Avg', 'Vrms ph-n BN Avg', 'Vrms ph-n CN Avg']],
                     'F2': [v.variables_F2, ['Vrms ph-n L1N Avg', 'Vrms ph-n L2N Avg', 'Vrms ph-n L3N Avg']]}

    if data is None:
        data = load_data(sampling=sampling)
        results = {}
    else:
        results = []

    for measurement in data:
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
                results[f"{data[measurement].name}"] = data[measurement].pca(variables[data[measurement].name[-2:]][1],
                                                                             var_numbers, analysis=analysis,
                                                                             n_components=n_components)
            elif PCA_type == 'kPCA':
                results[f"{data[measurement].name}"] = data[measurement].kpca(variables[data[measurement].name[-2:]][1],
                                                                              var_numbers)
            else:
                print('Unknown type of PCA enterered (enter either PCA or kPCA)')
        else:
            if PCA_type == 'PCA':
                results.append(measurement.pca(variables[measurement.name[-2:]][1], var_numbers, analysis=analysis,
                                               n_components=n_components)[1])
            elif PCA_type == 'kPCA':
                results.append(measurement.kpca(variables[measurement.name[-2:]][1],
                                                var_numbers)[1])
            else:
                print('Unknown type of PCA enterered (enter either PCA or kPCA)')

    if type(results) is list:
        results = np.array(results)

    return results


def pca_plotting(results, type='PCA', number_of_vars=len(v.pca_variables_B1)):
    explained_variances = {}

    for test_bay in test_bays:
        for measurement in measurements:
            explained_variances[str(measurement)[13:] + ': Test Bay ' + str(test_bay)] = []
            for scenario in measurements[measurement]:
                explained_variances[str(measurement)[13:] + ': Test Bay ' + str(test_bay)].append(
                    results[str(measurement)[13:] + ' Scenario ' + str(
                        measurements[measurement].index(scenario) + 1) + ': Test Bay ' + str(test_bay)][1])

    fgs, axs = plot_pca(explained_variances, type=type, number_of_vars=number_of_vars)

    return fgs, axs


def ssa(variables=None, sampling=None):
    # TO DO?
    if variables is None:
        variables = {'B1': [v.variables_B1, ['Vrms ph-n AN Avg', 'Vrms ph-n BN Avg', 'Vrms ph-n CN Avg']],
                     'F1': [v.variables_F1, ['Vrms ph-n AN Avg', 'Vrms ph-n BN Avg', 'Vrms ph-n CN Avg']],
                     'F2': [v.variables_F2, ['Vrms ph-n L1N Avg', 'Vrms ph-n L2N Avg', 'Vrms ph-n L3N Avg']]}
    results = {}

    data = load_data(sampling=sampling)
    for measurement in data:
        var_numbers = [variables[data[measurement].name[-2:]][0].index(i) + 1 for i in
                       variables[data[measurement].name[-2:]][1]]
        results[f"{data[measurement].name}"] = data[measurement].ssa(variables[data[measurement].name[-2:]][1],
                                                                     var_numbers)

    return results


def find_most_common_PCs(results_pca):  # , number_of_variables=15):
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


def scoring(y_test, y_pred):
    metrics = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    return [accuracy, metrics]


def svm_algorithm(data, SVM_type='SVM', cross_val=False, kernel='linear', gamma='scale', degree=3):
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
        scores = scoring(y_test, y_pred)

        print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
        print(f"\n########## Metrics for {data.name} ##########")
        print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1], scores[1][2],
                                                                                 scores[1][3]))

    return y_pred, y_test


def kNN_algorithm(data, cross_val=False, neighbours=2, weights='uniform'):
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
        scores = scoring(y_test, y_pred)

        print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
        print(f"\n########## Metrics for {data.name} ##########")
        print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1], scores[1][2],
                                                                                 scores[1][3]))

    return y_pred, y_test


def assembly_learner_single_dataset(data, clf_types_and_paras, cross_val=False, variables=None):
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
                X_train_pca = pca(variables=variables, PCA_type='PCA',
                                  n_components=clf_types_and_paras[clf_type][kernel][0], data=X_train,
                                  sampling=sampling_step_size_in_seconds)
                X_test_pca = pca(variables=variables, PCA_type='PCA',
                                 n_components=clf_types_and_paras[clf_type][kernel][0], data=X_test,
                                 sampling=sampling_step_size_in_seconds)
                if kernel == 'poly':
                    clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel, degree=clf_types_and_paras[clf_type][kernel][
                        1])  # default Linear Kernel
                else:
                    clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel)
                clfs[clf_type + '_' + kernel].fit(X_train_pca, y_train)
                y_preds[clf_type + '_' + kernel] = clfs[clf_type + '_' + kernel].predict(
                    X_test_pca)  # Predict the response for test dataset
        elif clf_type == 'NuSVM':
            for kernel in clf_types_and_paras[clf_type]:
                X_train_pca = pca(variables=variables, PCA_type='PCA',
                                  n_components=clf_types_and_paras[clf_type][kernel][0], data=X_train,
                                  sampling=sampling_step_size_in_seconds)
                X_test_pca = pca(variables=variables, PCA_type='PCA',
                                 n_components=clf_types_and_paras[clf_type][kernel][0], data=X_test,
                                 sampling=sampling_step_size_in_seconds)
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
                X_train_pca = pca(variables=variables, PCA_type='PCA',
                                  n_components=clf_types_and_paras[clf_type][neighbours][0],
                                  data=X_train, sampling=sampling_step_size_in_seconds)
                X_test_pca = pca(variables=variables, PCA_type='PCA',
                                 n_components=clf_types_and_paras[clf_type][neighbours][0],
                                 data=X_test, sampling=sampling_step_size_in_seconds)
                clfs[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                    1] + '_weights'] = neighbors.KNeighborsClassifier(n_neighbors=neighbours,
                                                                      weights=clf_types_and_paras[clf_type][neighbours][
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
                          key=[i[index] for i in y_preds.values()].count))  # pick class that's most commonly predicted
    y_pred = np.array(y_pred)

    if not cross_val:
        scores = scoring(y_test, y_pred)

        print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
        print(f"\n########## Metrics for {data.name} ##########")
        print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1], scores[1][2],
                                                                                 scores[1][3]))

    return y_pred, y_test


def assembly_learner_combined_dataset(data, clf_types_and_paras, cross_val=False):
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
                    clfs[clf_type + '_' + kernel] = svm.SVC(kernel=kernel, degree=clf_types_and_paras[clf_type][kernel][
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
                                                                      weights=clf_types_and_paras[clf_type][neighbours][
                                                                          0])
                clfs[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                    0] + '_weights'].fit(X_train, y_train)
                y_preds[clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                    0] + '_weights'] = clfs[
                    clf_type + '_' + str(neighbours) + 'NN' + '_' + clf_types_and_paras[clf_type][neighbours][
                        0] + '_weights'].predict(
                    X_test)  # Predict the response for test dataset

    y_pred = []
    for index in list(range(len(y_test))):
        y_pred.append(max(set([i[index] for i in y_preds.values()]),
                          key=[i[index] for i in y_preds.values()].count))  # pick class that's most commonly predicted
    y_pred = np.array(y_pred)

    if not cross_val:
        scores = scoring(y_test, y_pred)

        print(f'Predicted labels: {y_pred}; correct labels: {y_test}')
        print(f"\n########## Metrics for {data.name} ##########")
        print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}\n".format(scores[0], scores[1][1], scores[1][2],
                                                                                 scores[1][3]))

    return y_pred, y_test


def cross_val(data, clf='SVM', kernel='linear', neighbours=2, weights='uniform', degree=3,
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
        data = load_data(sampling=sampling)
        raw_dataset = Raw_Dataset(data, name=setup, classes=setups[setup], bay=setup.split('_')[2],
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
            y_pred, y_test = svm_algorithm([X_train, X_test, y_train, y_test], SVM_type=clf, cross_val=True,
                                           kernel=kernel, degree=degree)
            scores.append(scoring(y_test, y_pred))
        elif clf == 'kNN':
            y_pred, y_test = kNN_algorithm([X_train, X_test, y_train, y_test], cross_val=True, neighbours=neighbours,
                                           weights=weights)
            scores.append(scoring(y_test, y_pred))
        elif clf == 'Assembly':
            if data_mode == 'measurement_wise':
                y_pred, y_test = assembly_learner_single_dataset([X_train, X_test, y_train, y_test],
                                                                 classifiers_and_parameters, cross_val=True,
                                                                 variables=variables)
            elif data_mode == 'combined_data':
                y_pred, y_test = assembly_learner_combined_dataset([X_train, X_test, y_train, y_test],
                                                                   classifiers_and_parameters, cross_val=True)
            scores.append(scoring(y_test, y_pred))
        else:
            print('undefined classifier entered')

    scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores],
                   'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores]}

    return scores_dict


def remove_objects_in_list_from_list(list, object_list):
    for i in object_list:
        list.remove(i[0])
    return list


plot_data = True

classifier_combos = c.classifier_combos[config.classifier_combos]   #detection, c_vs_w ...


if __name__ == '__main__':  # see config file for settings

    if config.raw_data_available == False:
        if config.deeplearning:
            generate_deeplearning_raw_data()
        if config.detection_methods:
            generate_detectionmethods_raw_data()

    if config.dataset_available == False or config.detection_methods:
        dataset = create_dataset()

    #in config file do as: if deeplearning: learning_config = ... elif ...: learning_config = ....
    print("\n########## Configuration ##########")
    for key, value in learning_config.items():
        print(key, ' : ', value)
    if config.deeplearning: print("number of samples : %d" % config.number_of_samples)

    if config.deeplearning:
        deep_learning = Deeplearning(config, learning_config)
        deep_learning.training_or_testing()

    elif config.detection_methods:
        '''
        TODO: pack into classes, also functions defined above, pack modules in folders... 
        '''
        data_path = os.path.join(os.getcwd(), config.raw_data_folder, 'ERIGrid-Test-Results-26-11-2021-phase1_final')
        test_bays = ['B1', 'F1', 'F2']
        scenario = 1  # 1 to 15 as there is 15 scenarios (profiles)
        plotting_variables = {'B1': 'Vrms ph-n AN Avg', 'F1': 'Vrms ph-n AN Avg',
                              'F2': 'Vrms ph-n L1N Avg'}  # see dictionary above
        variables = {'B1': [v.variables_B1, v.pca_variables_B1], 'F1': [v.variables_F1, v.pca_variables_F1],
                     'F2': [v.variables_F2, v.pca_variables_F2]}
        sampling_step_size_in_seconds = None  # None or 0 to use all data, 1, 20 to sample once every n seconds ....

        setups = {'Setup_A_F2_data': ['correct', 'wrong'], 'Setup_B_F2_data1_3c': ['correct', 'wrong', 'inversed'],
                  'Setup_B_F2_data2_2c': ['correct', 'wrong'], 'Setup_B_F2_data3_2c': ['correct', 'inversed']}
        setup_chosen = 'Setup_A_F2_data'  # for assembly or clustering
        mode = 'detection'  # classification means wrong as wrong and inversed as inversed, detection means wrong and inversed as wrong
        data_mode = 'combined_data'  # 'measurement_wise', 'combined_data'

        approach = 'PCA+clf'  # 'PCA+clf', 'clustering'

        if plot_data:
            fgs_test_bay, axs_test_bay = scenario_plotting_test_bay(variables, plot_all=False, vars=plotting_variables,
                                                                    sampling=sampling_step_size_in_seconds)
            fgs_case, axs_case = scenario_plotting_case(variables, plot_all=False, vars=plotting_variables,
                                                        sampling=sampling_step_size_in_seconds)

        if approach == 'clustering':
            data = load_data(sampling=sampling_step_size_in_seconds)
            dataset = Raw_Dataset(data, name=setup_chosen, classes=setups[setup_chosen], bay=setup_chosen.split('_')[2],
                                  Setup=setup_chosen.split('_')[1], labelling=mode)

            Clustering_ward = Clustering(data=dataset, variables=variables[setup_chosen.split('_')[2]][1],
                                         metric='euclidean', method='ward', num_of_clusters=len(setups[setup_chosen]))
            cluster_assignments = Clustering_ward.cluster()
            score = Clustering_ward.rand_score()
            print(f'Score reached: {score}')

        if approach == 'PCA+clf':

            if data_mode == 'measurement_wise':
                results_pca = pca(variables=variables, PCA_type='PCA', analysis=True,
                                  sampling=sampling_step_size_in_seconds)
                variable_selection = find_most_common_PCs(
                    results_pca)  # , number_of_variables = 15)   #returns the most important variables for the PCA per measurement point > use to to do PCA and then SVM
                if plot_data:
                    fgs_pca, axs_pca = pca_plotting(results_pca, type='PCA', number_of_vars=len(variables['B1'][1]))

                selection = 'most important'  # most impotant, least important variables picked after assessment by PCA

                if selection == 'most important':
                    variable_selection = {'B1': [v.variables_B1, [i[0] for i in variable_selection[0]]],
                                          'F1': [v.variables_F1, [i[0] for i in variable_selection[1]]],
                                          'F2': [v.variables_F2, [i[0] for i in variable_selection[2]]]}
                if selection == 'least important':
                    pca_variables_B1 = remove_objects_in_list_from_list(v.pca_variables_B1, variable_selection[0])
                    pca_variables_F1 = remove_objects_in_list_from_list(v.pca_variables_F1, variable_selection[1])
                    pca_variables_F2 = remove_objects_in_list_from_list(v.pca_variables_F2, variable_selection[2])
                    variable_selection = {'B1': [v.variables_B1, pca_variables_B1],
                                          'F1': [v.variables_F1, pca_variables_F1],
                                          'F2': [v.variables_F2, pca_variables_F2]}

                if plot_data:
                    results_pca = pca(variables=variable_selection, PCA_type='PCA',
                                      sampling=sampling_step_size_in_seconds)
                    fgs_pca, axs_pca = pca_plotting(results_pca, type='PCA', number_of_vars=max(
                        [len(variable_selection['B1'][1]), len(variable_selection['F1'][1]),
                         len(variable_selection['F2'][1])]))

            # results_pca = pca(variables=variable_selection, type='PCA', n_components=len(variable_selection['B1'][1]))
            scores_by_n = {}

            # learning settings
            clf = 'Assembly'  # SVM, NuSVM, kNN, Assembly
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            # kernels = ['poly']
            if kernels[0] == 'poly' and len(kernels) == 1:
                degrees = list(range(1, 7))
            else:
                degrees = [3]
            gammas = ['scale']  # , 'auto']#[1/(i+1) for i in range(15)] #['scale', 'auto']
            neighbours = [i + 1 for i in range(5)]
            weights = ['uniform', 'distance']
            # max_number_of_components = 2    #means 1
            # classifiers = {'SVM': {'poly': [8]}, 'NuSVM': {'linear': [9], 'poly': [11], 'rbf': [2]}, 'kNN': {3: [18,'uniform']}}
            # classifiers = {'NuSVM': {'linear': [11], 'poly': [3], 'rbf': [10]}}
            # classifiers = {'NuSVM': {'poly': [2]}}

            if clf != 'Assembly' and data_mode != 'combined_data':
                max_number_of_components = len(variable_selection['F2'][1])  # one less than variables due to range
                for n in range(1, max_number_of_components):

                    print(f'\nPCA done with {n} components\n')
                    results_pca = pca(variables=variable_selection, PCA_type='PCA', n_components=n,
                                      sampling=sampling_step_size_in_seconds)

                    Setup_A_F2_Data = PCA_Dataset(results_pca, name='Setup_A_F2_data',
                                                  classes=setups['Setup_A_F2_data'],
                                                  bay='F2', Setup='A')
                    Setup_B_F2_Data1_3c = PCA_Dataset(results_pca, name='Setup_B_F2_data1_3c',
                                                      classes=setups['Setup_B_F2_data1_3c'], bay='F2', Setup='B',
                                                      labelling=mode)  # ['correct', 'wrong', 'inversed']
                    Setup_B_F2_Data2_2c = PCA_Dataset(results_pca, name='Setup_B_F2_data2_2c',
                                                      classes=setups['Setup_B_F2_data2_2c'],
                                                      bay='F2', Setup='B')
                    Setup_B_F2_Data3_2c = PCA_Dataset(results_pca, name='Setup_B_F2_data3_2c',
                                                      classes=setups['Setup_B_F2_data3_2c'],
                                                      bay='F2', Setup='B')

                    datasets = []
                    datasets.append(Setup_A_F2_Data)
                    datasets.append(Setup_B_F2_Data1_3c)
                    datasets.append(Setup_B_F2_Data2_2c)
                    datasets.append(Setup_B_F2_Data3_2c)

                    scores_by_dataset = {}

                    for dataset in datasets:

                        # SVM
                        results_svm = svm_algorithm(dataset)
                        # kNN
                        results_kNN = kNN_algorithm(dataset)

                        print("\n########## k-fold Cross-validation ##########")
                        if clf in ['SVM', 'NuSVM']:
                            scores_by_kernel = {}
                            for kernel in kernels:
                                scores_by_degree = {}
                                for degree in degrees:
                                    scores = cross_val(dataset, clf=clf, kernel=kernel, degree=degree,
                                                       sampling=sampling_step_size_in_seconds)
                                    if dataset.labelling == 'classification':
                                        print(
                                            f"\n########## Metrics for {clf} classifier applied on {dataset.name} using a {kernel} kernel of degree {degree} with classes {dataset.classes} ##########")
                                    elif dataset.labelling == 'detection':
                                        print(
                                            f"\n########## Metrics for {clf} classifier applied on {dataset.name} using a {kernel} kernel of degree {degree} with classes normal and abnormal ##########")
                                    for score in scores:
                                        print("%s: %0.2f (+/- %0.2f)" % (
                                            score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))
                                    scores_by_degree[degree] = scores
                                scores_by_kernel[kernel] = scores_by_degree
                            scores_by_dataset[dataset.name] = scores_by_kernel
                        if clf in ['kNN']:
                            scores_by_neighbours = {}
                            for number in neighbours:
                                scores_by_weights = {}
                                for weight in weights:
                                    scores = cross_val(dataset, neighbours=number, weights=weight,
                                                       sampling=sampling_step_size_in_seconds)
                                    if dataset.labelling == 'classification':
                                        print(
                                            f"\n########## Metrics for {clf} classifier applied on {dataset.name} using {number} neigbours and {weight} weights with classes {dataset.classes} ##########")
                                    elif dataset.labelling == 'detection':
                                        print(
                                            f"\n########## Metrics for {clf} classifier applied on {dataset.name} using {number} neigbours with and {weight} weights classes normal and abnormal ##########")
                                    for score in scores:
                                        print("%s: %0.2f (+/- %0.2f)" % (
                                            score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))
                                        scores_by_weights[weight] = scores
                                scores_by_neighbours[number] = scores_by_weights
                            scores_by_dataset[dataset.name] = scores_by_neighbours
                    scores_by_n[str(n)] = scores_by_dataset

                for dataset in setups.keys():
                    if clf in ['SVM', 'NuSVM']:
                        for kernel in kernels:
                            for degree in degrees:
                                if kernel == 'poly':
                                    poly_message = f'of degree {degree}'
                                else:
                                    poly_message = ''
                                if mode == 'classification' or dataset.split('_')[-1] != '3c':
                                    title = f'{clf} on {" ".join(dataset.split("_")[:4])} using a {kernel} kernel {poly_message} with classes {setups[dataset]}'
                                elif mode == 'detection' and dataset.split('_')[-1] == '3c':
                                    title = f'{clf} on {" ".join(dataset.split("_")[:4])} using a {kernel} kernel {poly_message} with classes normal and abnormal'
                                fig_svm, ax_svm = plot_grid_search(list(range(1, max_number_of_components)),
                                                                   [scores_by_n[i][dataset][kernel][degree] for i in
                                                                    scores_by_n],
                                                                   title=title)
                    if clf in ['kNN']:
                        for number in neighbours:
                            for weight in weights:
                                if mode == 'classification' or dataset.split('_')[-1] != '3c':
                                    title = f'{clf} on {" ".join(dataset.split("_")[:4])} using {number} neigbours and {weight} weights with classes {setups[dataset]}'
                                elif mode == 'detection' and dataset.split('_')[-1] == '3c':
                                    title = f'{clf} on {" ".join(dataset.split("_")[:4])} using {number} neigbours and {weight} weights with classes normal and abnormal'
                                fig_svm, ax_svm = plot_grid_search(list(range(1, max_number_of_components)),
                                                                   [scores_by_n[i][dataset][number][weight] for i in
                                                                    scores_by_n],
                                                                   title=title)

            if clf == 'Assembly' and data_mode != 'combined_data':
                for classifiers in classifier_combos:
                    scores = cross_val(variable_selection, clf=clf, classifiers_and_parameters=classifiers,
                                       setup=setup_chosen,
                                       mode=mode, sampling=sampling_step_size_in_seconds)
                    if mode == 'classification':
                        print(
                            f"\n########## Metrics for {clf} classifier on Setup_B_F2_data1_3c using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes {setups[setup_chosen]} ##########")
                    elif mode == 'detection':
                        print(
                            f"\n########## Metrics for {clf} classifier on Setup_B_F2_data1_3c using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes normal and abnormal ##########")
                    for score in scores:
                        print("%s: %0.2f (+/- %0.2f)" % (
                            score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

            if data_mode == 'combined_data':
                '''
                uses principal components instead of explained variances!!!
                '''
                data = load_data(sampling=sampling_step_size_in_seconds)
                data = Combined_Dataset(data, pca_variables_F2, name=setup_chosen,
                                                 classes=setups[setup_chosen],
                                                 bay=setup_chosen.split('_')[2], setup=setup_chosen.split('_')[1],
                                                 labelling=mode)
                '''scaled_data = data.scale()
                pca_data = data.PCA()
                labelled_data = data.label()'''

                for classifiers in classifier_combos:
                    scores = cross_val(data, clf=clf, classifiers_and_parameters=classifiers, setup=setup_chosen,
                                       mode=mode, sampling=sampling_step_size_in_seconds, data_mode=data_mode)
                    if mode == 'classification':
                        print(
                            f"\n########## Metrics for {clf} classifier on {setup_chosen} using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes {setups[setup_chosen]} ##########")
                    elif mode == 'detection':
                        print(
                            f"\n########## Metrics for {clf} classifier on {setup_chosen} using {[(i, classifiers[i]) for i in classifiers.keys()]} classifiers with classes normal and abnormal ##########")
                    for score in scores:
                        print("%s: %0.2f (+/- %0.2f)" % (
                            score, np.array(scores[score]).mean(), np.array(scores[score]).std() * 2))

            ##########

            # results_kpca = pca(variables=variables, PCA_type='kPCA', sampling=sampling_step_size_in_seconds)
            # fgs_kpca, axs_kpca = pca_plotting(results_kpca, type='kPCA')

        # results_ssa = ssa()

        plt.show()
        a = 1
