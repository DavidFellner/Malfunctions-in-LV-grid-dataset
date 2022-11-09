import importlib
import os
import matplotlib.pyplot as plt
import util
from experiment_config import experiment_path, chosen_experiment
import pandas as pd
import numpy as np

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


class Detection_application:
    # import lab data > only pick 'correct' samples > guess PQ loads for rest of network (using U and I values there?) > simulate 'wrong' samples
    # > assemble dataset of 'correct' lab samples and 'wrong'simulated samples except for latest sample, where both correct and incorrect should
    # be classified; rotate through samples so as to have 14 different last samples

    def __init__(self):

        self.sensor_data = None
        self.transformer_data = None  # pick out correct samples; faulty ones are to be simulated
        self.load_data = None  # get V and sigma values per sample and estimate PQ for simulations

        self.load_data_estimated = None  # PQ values estimated for simulation of faulty cases
        self.simulation_data = None  # do loadflow of faulty cases
        self.sim_transformer_data = None
        self.complete_transformer_data = None  # combine sim (hisoric faulty ones) and sensor data (correct ones and one faulty one) here

        self.detection_module = Transformer_detection(config, learning_config)

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

    def create_application_dataset(self, phase, setup):

        """self.simulation_data = self.detection_module.load_data(self.detection_module.sampling_step_size_in_seconds,
                                                               data_source='simulation',
                                                               phase_info=[phase, self.phases[phase]], grid_setup=setup,
                                                               marker='estimated')"""

        trafo_point = self.phases[phase][0]

        self.simulation_data = self.detection_module.load_data(self.detection_module.sampling_step_size_in_seconds,
                                                               data_source='simulation',
                                                               phase_info=[phase, self.phases[phase]], grid_setup=setup)

        self.simulation_trafo_data_wrong = {applicable_measurements.name: self.simulation_data[applicable_measurements.name] for
                                 applicable_measurements in
                                 [self.simulation_data[measurement] for measurement in self.simulation_data if
                                  measurement[-2:] == trafo_point and
                                  measurement.split(' ')[3] == setup and
                                  measurement.split(' ')[0] == 'wrong']}

        self.complete_transformer_datasets = create_dataset(type='detection_application', data=[self.sensor_data, self.simulation_trafo_data_wrong], phase_info=[phase, self.phases[phase]],
                                          variables=config.sim_variables_dict[phase.split('_')[-1]],
                                          name=phase.split('_')[-1] + '_setup_' + setup,
                                          Setup=setup, labelling=learning_config['mode'])

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
