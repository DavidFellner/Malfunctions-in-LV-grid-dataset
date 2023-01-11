import importlib
import os
import matplotlib.pyplot as plt
import util
from experiment_config import experiment_path, chosen_experiment
import pandas as pd
import numpy as np
from math import sin, acos

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

if not config.raw_data_available:
    from start_powerfactory import start_powerfactory
    from raw_data_generation.grid_preparation import prepare_grid
    from raw_data_generation.data_creation import create_deeplearning_data
    from raw_data_generation.data_creation import create_detectionmethods_data

from util import create_dataset
from transformer_detection import Transformer_detection
from disaggregation_module import Disaggregation

from detection_method_settings import measurements_DSM as measurements_phase2
from detection_method_settings import measurements as measurements_phase1

from raw_data_generation.grid_preparation import prepare_grid


class Detection_application:
    # import lab data > only pick 'correct' samples > guess PQ loads for rest of network (using U and I values there?) > simulate 'wrong' samples
    # > assemble dataset of 'correct' lab samples and 'wrong'simulated samples except for latest sample, where both correct and incorrect should
    # be classified; rotate through samples so as to have 14 different last samples

    def __init__(self):

        self.sensor_data = None
        self.transformer_data = None  # pick out correct samples; faulty ones are to be simulated
        self.load_data = None  # get V and sigma values per sample and estimate PQ for simulations
        self.profiles = None # PV and  load profiles used in the lab phase

        self.load_data_estimated = None  # PQ values estimated for simulation of faulty cases
        self.simulation_data = None  # do loadflow of faulty cases
        self.sim_transformer_data = None
        self.complete_transformer_data = None  # combine sim (hisoric faulty ones) and sensor data (correct ones and one faulty one) here

        self.detection_module = Transformer_detection(config, learning_config)

        self.pv_input = None
        self.pad_factor = 1

        self.trafo_point_phase1 = 'F2'  # pv use case
        self.trafo_point_phase2 = 'B2'  # dsm use case
        self.phases = {'trafo_point_phase1': [self.trafo_point_phase1, measurements_phase1],
                       'trafo_point_phase2': [self.trafo_point_phase2, measurements_phase2]}
        self.setups = ['A', 'B']

        self.score_dict = {}

    def load_sensor_data(self, phase, setup):

        data = self.detection_module.load_data(self.detection_module.sampling_step_size_in_seconds, data_source='real_world',
                                        phase_info=[phase, self.phases[phase]], grid_setup=setup)

        self.sensor_data = create_dataset(type='sensor', data=data, phase_info=[phase, self.phases[phase]],
                                          variables=config.variables_dict[phase.split('_')[-1]],
                                          name=phase.split('_')[-1] + '_setup_' + setup,
                                          Setup=setup)
        return self.sensor_data

    def load_estimation(self):

        load_data_estimate_dict = {}
        num_loads = int(len(self.sensor_data.load_data_correct) / len(self.sensor_data.trafo_data_correct))

        for sample_no in list(range(len(self.sensor_data.trafo_data_correct.index))):

            trafo_data_estimation = self.sensor_data.trafo_data_correct.iloc[sample_no]
            load_data_estimation = self.sensor_data.load_data_correct.iloc[sample_no:sample_no+num_loads]

            load_data_estimate_dict[sample_no] = 1 #(trafo_data_estimation, load_data_estimation) # TO DO > ASK SARAH ABOUT CODE

        self.load_data_estimated = pd.DataFrame.from_dict(load_data_estimate_dict)  #TO DO: assemble dataframe of samples

        return self.load_data_estimated

    def create_application_dataset(self, phase, setup, estimation):

        """self.simulation_data = self.detection_module.load_data(self.detection_module.sampling_step_size_in_seconds,
                                                               data_source='simulation',
                                                               phase_info=[phase, self.phases[phase]], grid_setup=setup,
                                                               marker='estimated')"""

        trafo_point = self.phases[phase][0]

        self.simulation_data = self.detection_module.load_data(self.detection_module.sampling_step_size_in_seconds,
                                                               data_source='simulation',
                                                               phase_info=[phase, self.phases[phase]], grid_setup=setup)

        self.simulation_data_estimation = self.detection_module.load_data(self.detection_module.sampling_step_size_in_seconds,
                                                               data_source='simulation',
                                                               phase_info=[phase, self.phases[phase]], grid_setup=setup, marker=estimation)

        classes = config.setups[learning_config['setup_chosen']]
        classes.remove('correct')
        self.simulation_trafo_data_wrong = {applicable_measurements.name: self.simulation_data_estimation[applicable_measurements.name] for
                                 applicable_measurements in
                                 [self.simulation_data_estimation[measurement] for measurement in self.simulation_data_estimation if
                                  measurement[-2:] == trafo_point and
                                  measurement.split(' ')[3] == setup and
                                  measurement.split(' ')[0] in classes]}

        self.simulation_trafo_data_correct = {
            applicable_measurements.name: self.simulation_data[applicable_measurements.name] for
            applicable_measurements in
            [self.simulation_data[measurement] for measurement in self.simulation_data if
             measurement[-2:] == trafo_point and
             measurement.split(' ')[3] == setup and
             measurement.split(' ')[0] == 'correct']}

        self.simulation_trafo_data_wrong = {
            applicable_measurements.name: self.simulation_data[applicable_measurements.name] for
            applicable_measurements in
            [self.simulation_data[measurement] for measurement in self.simulation_data if
             measurement[-2:] == trafo_point and
             measurement.split(' ')[3] == setup and
             measurement.split(' ')[0] in classes]}


        self.complete_transformer_datasets = create_dataset(type='detection_application', data=[self.simulation_trafo_data_correct, self.simulation_trafo_data_wrong, self.simulation_trafo_data_wrong], phase_info=[phase, self.phases[phase]],
                                          variables=config.sim_variables_dict[phase.split('_')[-1]],
                                          name=phase.split('_')[-1] + '_setup_' + setup,
                                          Setup=setup, labelling=learning_config['mode'],
                                          classes=classes)

        return self.complete_transformer_datasets

    def cross_val(self, classifiers_and_parameters=None):
        # ITERATE OVER COMBOS DATA TO TEST ALL SCENARIOS AS IN CV

        if classifiers_and_parameters is None:
            classifiers_and_parameters = {'SVM': {'poly': [8]}, 'NuSVM': {'linear': [9], 'poly': [11], 'rbf': [2]},
                                          'kNN': {3: [18, 'uniform']}}

        scores = []
        for combination in self.complete_transformer_data.scenario_combos_data:
        # application dataset!! split into training (28 samples) and test (1 correct 1 wrong) to do training and test on differenct combinations
            data = self.complete_transformer_data.scenario_combos_data[combination]

            X_train, X_test = data['training']['X'], data['testing']['X']
            y_train, y_test = np.array(data['training']['y']), np.array(data['testing']['y'])

            y_pred, y_test = self.detection_module.assembly_learner_combined_dataset([X_train, X_test, y_train, y_test],
                                                                            classifiers_and_parameters, cross_val=True)
            score = self.detection_module.scoring(y_test, y_pred)       #do differently?
            score.append((list(y_pred), list(y_test)))
            scores.append(score)
            #print(f'Pred vs Test{(list(y_pred), list(y_test))}')

        scores_dict = {'Accuracy': [i[0] for i in scores], 'Precision': [i[1][0] for i in scores],
                       'Recall': [i[1][1] for i in scores], 'FScore': [i[1][2] for i in scores], 'Pred_vs_Test': [i[2] for i in scores]}

        return scores_dict



    def detection(self, phase, setup):

        scores_by_clfs = {}
        for classifier in self.detection_module.classifier_combos:
            key = f'{classifier}'
            if key == "{'DT': {'gini': []}}":
                a = 1
            scores_by_clfs[key] = self.cross_val(classifiers_and_parameters=classifier)

            print(
                f"\n########## Metrics for {classifier} classifier on data of {phase} in setup {setup} ##########")
            for score in scores_by_clfs[key]:
                if score == 'Pred_vs_Test':
                    print((f"%s: {scores_by_clfs[key][score]}" % (score)))
                else:
                    print("%s: %0.2f (+/- %0.2f)" % (
                        score, np.array(scores_by_clfs[key][score]).mean(), np.array(scores_by_clfs[key][score]).std() * 2))


        return scores_by_clfs

    def pick_estimation_input_data(self, just_pv=False):

        data = self.sensor_data
        test_bays = data.test_bays
        trafo_point = data.trafo_point
        setup = data.setup
        columns = []
        num_columns_smart_meter_data = int(len(data.load_data_correct_unflattened.columns) / (len(test_bays)-1))
        if config.power_unit_sensor_data == 'MW':
            pv_and_load_factor = 1000
            flows_factor = 1/1000
        elif config.power_unit_sensor_data == 'KW':
            pv_and_load_factor = 1
            flows_factor = 1/1000
        else:
            pv_and_load_factor = 0.001
            flows_factor = 1/1000

        q_table = list(np.zeros(50)) + [sin(acos((x))) for x in (np.zeros(50) + 1) - np.linspace(0, 1,
                                                                                                 50) / 10]  # from cosphi = 1 to cosphi = 0.9
        cosphi_of_P_curve = {'p': list(np.linspace(0, 1, 100)), 'q': q_table}
        p_rated = 6  # in kW

        # PV input > assumed to be known bc PVs known + solar radation known
        PV_data = self.profiles[list(self.profiles.keys())[-1]]
        p_s = [PV_data[day_data].vector for day_data in self.profiles[list(self.profiles.keys())[-1]] if
               day_data[0] == 'p']
        pv_p = []
        pv_q = []

        self.pad_factor = int(len(data.load_data_correct_unflattened.index) / (len(p_s) * len(p_s[0])))

        minus = 1
        for day in p_s:
            p_values = []
            q_values = []
            for p_value in day:
                ind = min(range(len(cosphi_of_P_curve['p'])),
                          key=lambda i: abs(cosphi_of_P_curve['p'][i] - p_value / p_rated))
                q_value = cosphi_of_P_curve['q'][ind] * p_value
                p_value = [p_value] * self.pad_factor
                p_values += p_value
                q_value = [q_value] * self.pad_factor
                q_values += q_value
            pv_p += [value / (pv_and_load_factor) * minus for value in p_values]
            pv_q += [value / (pv_and_load_factor) * minus *-1 for value in q_values]

        pv_p = pv_p
        self.pv_input = pv_p
        columns.append(f'PV A_P')

        if just_pv:
            return [0,0]

        pv_q = pv_q
        columns.append(f'PV A_Q')

        pv_inactive = list(np.zeros(len(pv_p)))
        columns.append(f'PV B_P')

        pv_inactive = list(np.zeros(len(pv_q)))
        columns.append(f'PV B_Q')

        # Load input > to be estimated, therefore, used as test labels
        LB_data = pd.DataFrame(index=data.load_data_correct_unflattened.index)
        for load in list(self.profiles.keys())[:-1]:
            lb_ps = [self.profiles[load][day_data].vector for day_data in self.profiles[load] if
               day_data[0] == 'p']
            lb_qs  = [self.profiles[load][day_data].vector for day_data in self.profiles[load] if
                                             day_data[0] == 'q']

            p_merged = []
            q_merged = []
            for day in lb_ps:
                p_values = []
                for p_value in day:
                    p_value = [p_value] * self.pad_factor
                    p_values += p_value
                p_merged += [value / (pv_and_load_factor * 1) for value in p_values]
            for day in lb_qs:
                q_values = []
                for q_value in day:
                    q_value = [q_value] * self.pad_factor
                    q_values += q_value
                q_merged += [value / (pv_and_load_factor * minus)  for value in q_values]

            LB_data[load.loc_name + '_P'] = p_merged
            LB_data[load.loc_name + '_Q'] = q_merged

        #voltages
        trafo_voltage = data.trafo_data_correct_unflattened.iloc[:, :3].mean(axis=1) / 230
        columns.append(f'Test Bay {trafo_point}_V')
        smart_meter_voltages = {}
        n = 0
        for test_bay  in test_bays:
            if test_bay is not trafo_point:
                smart_meter_voltages[f'Test Bay {test_bay}_V'] = data.load_data_correct_unflattened.iloc[:, n*num_columns_smart_meter_data:3+n*num_columns_smart_meter_data].mean(axis=1) / 230
                columns.append(f'Test Bay {test_bay}_V')
                n +=1

        #active power(s)
        trafo_p = abs(data.trafo_data_correct_unflattened.iloc[:, 44:47].mean(axis=1)/flows_factor)
        columns.append(f'Test Bay {trafo_point}_p')

        smart_meter_ps = {}
        n = 0
        for test_bay in test_bays:
            if test_bay is not trafo_point:
                smart_meter_ps[f'Test Bay {test_bay}_p'] = abs(data.load_data_correct_unflattened.iloc[:,
                                           52+(n * num_columns_smart_meter_data):55+(n * num_columns_smart_meter_data)].mean(
                    axis=1)/flows_factor)
                columns.append(f'Test Bay {test_bay}_p')
                n += 1

        #reactive power(s)
        trafo_q = abs(data.trafo_data_correct_unflattened.iloc[:, 56:59].mean(axis=1)/(flows_factor))
        columns.append(f'Test Bay {trafo_point}_q')

        smart_meter_qs = {}
        n = 0
        for test_bay in test_bays:
            if test_bay is not trafo_point:
                smart_meter_qs[f'Test Bay {test_bay}_q'] = abs(data.load_data_correct_unflattened.iloc[:,
                                           64+(n * num_columns_smart_meter_data):67+(n * num_columns_smart_meter_data)].mean(
                    axis=1)/(flows_factor))
                columns.append(f'Test Bay {test_bay}_q')
                n += 1



        X = pd.DataFrame(index=data.load_data_correct_unflattened.index)

        if setup == 'A':
            X[columns[0]] = pv_p
            X[columns[1]] = pv_q
            X[columns[2]] = pv_inactive
            X[columns[3]] = pv_inactive
        else:
            X[columns[0]] = pv_inactive
            X[columns[1]] = pv_inactive
            X[columns[2]] = pv_p
            X[columns[3]] = pv_q

        test_bays.remove(trafo_point)
        X[columns[4]] = trafo_voltage
        for column in test_bays:
            X[f'Test Bay {column}_V'] = smart_meter_voltages[f'Test Bay {column}_V']


        X[f'Test Bay {trafo_point}_p'] = trafo_p
        for column in test_bays:
            X[f'Test Bay {column}_p'] = smart_meter_ps[f'Test Bay {column}_p']

        X[f'Test Bay {trafo_point}_q'] = trafo_q
        for column in test_bays:
            X[f'Test Bay {column}_q'] = smart_meter_qs[f'Test Bay {column}_q']

        return [X, LB_data]

