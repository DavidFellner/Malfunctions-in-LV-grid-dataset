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
import os
import matplotlib.pyplot as plt

import util
from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

if not config.raw_data_available or config.detection_application:
    from start_powerfactory import start_powerfactory
    from raw_data_generation.grid_preparation import prepare_grid
    from raw_data_generation.data_creation import create_deeplearning_data
    from raw_data_generation.data_creation import create_detectionmethods_data
    from raw_data_generation.data_creation import create_load_estimation_training_data
    from raw_data_generation.data_creation import create_load_estimation_input_data
    from load_estimation.neurallelf.runner import load_estimation as training
    from load_estimation.neurallelf.evaluator import eval as estimate

from util import create_dataset, pick_classes
from deeplearning import Deeplearning
from transformer_detection import Transformer_detection
import plotting
#from disaggregation_module import Disaggregation
from detection import Detection_application


def generate_deeplearning_raw_data():

    for file in os.listdir(config.grid_data_folder):
        if os.path.isdir(os.path.join(config.grid_data_folder, file)):
            print('Creating data using the grid %s' % file)
            app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
            grid_data = prepare_grid(app, file, o_ElmNet)

            create_deeplearning_data(app, o_ElmNet, grid_data, study_case_obj, file)
            print('Done with grid %s' % file)

    print('Done with all grids')

    return

def generate_detectionmethods_raw_data(data=None, phase=None, setup=None, extract_profiles=False, estimation=None, pv_input=None):
    '''
    USE PNDC GRID MODELS HERE
    :return:
    '''
    if config.detection_application:
        file = config.pf_file_dict[phase.split('_')[-1]]
    else:
        file = config.pf_file

    if extract_profiles:
        print('Load data for the grid %s' % file)
    else:
        print('Creating data using the grid %s' % file)
    app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
    grid_data = prepare_grid(app, study_case_obj, o_ElmNet, data=data, setup=setup, extract_profiles=extract_profiles, pv_input=pv_input)
    if not extract_profiles:
        create_detectionmethods_data(app, o_ElmNet, grid_data, study_case_obj, file, setup, estimation=estimation, phase=phase)

        print('Done with all simulations')

    return grid_data

def load_profiles():
    grid_data = prepare_grid()
    return grid_data

def generate_load_estimation_training_data(phase=None, setup=None):
    '''
    USE PNDC GRID MODELS HERE
    :return:
    '''

    file = config.pf_file_dict[phase.split('_')[-1]]
    file = file + '_training'

    print('Creating training data for load estimation NN using the grid %s' % file)
    app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
    grid_data = prepare_grid(app, study_case_obj, o_ElmNet, data='sampled', phase=phase, setup=setup)
    create_load_estimation_training_data(app, o_ElmNet, grid_data, study_case_obj, file, phase, setup)

    print('Done with all simulations')

    return

def compile_load_estimation_input_data(data, phase=None, setup=None):
    '''
    Used to merge sensor data with (PV) profiles which are assumed to be known
    USE PNDC GRID MODELS HERE
    :return:
    '''

    if type(data) is not list:
        print('Please pass the sensor data and the (PV) profiles as two elements of a list')
        return

    file = config.pf_file_dict[phase.split('_')[-1]]
    file = file + '_training'

    app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
    grid_data = prepare_grid(app, study_case_obj, o_ElmNet, data=data, phase=phase, setup=setup)
    create_load_estimation_input_data(app, o_ElmNet, grid_data, study_case_obj, file, phase, setup)

    print('Done with all simulations')

    return


############################

"""def ssa(variables=None, sampling=None):
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

    return results"""

if __name__ == '__main__':  # see config file for settings

    #if learning_config["do hyperparameter sensitivity analysis"]: plotting.plot_hyp_para_tuning()
    #if learning_config["do grid search"]: plotting.plot_grid_search()

    if config.raw_data_available is False:
        if config.deeplearning:
            generate_deeplearning_raw_data()
        if config.detection_methods:
            generate_detectionmethods_raw_data()

    print("\n########## Configuration ##########")
    for key, value in learning_config.items():
        print(key, ' : ', value)
    if config.deeplearning: print("number of samples : %d" % config.number_of_samples)

    if config.deeplearning:
        if config.dataset_available is False:
            dataset = create_dataset()
        if config.detection_methods:
            X, y = util.load_dataset()
            util.detection_method_dl(Transformer_detection, X, y)

        else:
            if learning_config["do hyperparameter sensitivity analysis"]:
                runs = len(learning_config["hyperparameter tuning"][1])
            else:
                runs = 1
            for i in range(runs):
                deep_learning = Deeplearning(config, learning_config, i)
                deep_learning.training_or_testing(i)
            if learning_config["do hyperparameter sensitivity analysis"]: plotting.plot_hyp_para_tuning()

    elif config.detection_methods:
        detection = Transformer_detection(config, learning_config)
        #if detection.plot_data and config.use_case != 'DSM': detection.plotting_data()
        if detection.plot_data: detection.plotting_data()
        if detection.approach == 'clustering': detection.clustering()
        if detection.approach == 'PCA+clf': detection.detection()

        """elif config.disaggregation:
            disaggregation = Disaggregation(config, learning_config)
            disaggregation.disaggregation()"""

    elif config.detection_application:
        application = Detection_application()
        for phase in application.phases:

            for setup in application.setups:
                if learning_config['setup_chosen'] != 'all' and setup in learning_config['setup_chosen'][phase.split('_')[-1]]:
                    if type(learning_config['setup_chosen'][phase.split('_')[-1]]) is set or learning_config['setup_chosen'][phase.split('_')[-1]][setup] == config.setups[phase.split('_')[-1]][setup][1:]:
                        continue

                if config.raw_data_available is False:
                    if not config.load_estimation_training_data_available:
                        generate_load_estimation_training_data(phase=phase, setup=setup)
                    else:
                        print('\nTraining data found')
                    if not config.pretrained:
                        training(phase, setup=setup)  # train NN with generated data
                    else:
                        print('Pretrained Network used')

                    if not config.load_estimation_input_data_available or config.use_saved_load_estimation_input_data:
                        application.profiles = \
                            generate_detectionmethods_raw_data(phase=phase, setup=setup, extract_profiles=True)[
                                0]  # extract load and pv data used in lab setting"""
                        application.sensor_data = application.load_sensor_data(phase, setup)
                        if not config.use_saved_load_estimation_input_data:
                            inputs = application.pick_estimation_input_data()
                            compile_load_estimation_input_data(data=[application.sensor_data, inputs], phase=phase, setup=setup)
                            inputs = 'from_file'
                        else:
                            application.pick_estimation_input_data(just_pv=True)
                            inputs = 'from_file'
                    else:
                        application.profiles = \
                            generate_detectionmethods_raw_data(phase=phase, setup=setup, extract_profiles=True)[
                                0]  # extract load and pv data used in lab setting
                        application.sensor_data = application.load_sensor_data(phase, setup)
                        inputs = application.pick_estimation_input_data()

                    application.load_data_estimated, load_data = estimate(phase, setup, X=inputs, pad_factor=application.pad_factor, config=config) #predict the loads using selected sensor data; returns both the estimation using a NN as well as LR
                else:
                    application.load_data_estimated = {'NN estimate': 'in file', 'LR estimate': 'in file'}

                for estimation in application.load_data_estimated:
                    if estimation.split(" ")[0] not in learning_config['disaggregation algorithm']:
                        continue
                    if config.raw_data_available is False:
                        #generate (wrong) samples using simulation
                        generate_detectionmethods_raw_data(data=application.load_data_estimated[estimation].iloc[::application.pad_factor, :], phase=phase, setup=setup, estimation=estimation, pv_input=application.pv_input[::240])

                    print('\n-----------------------------------------------------------')
                    print(f'\nData estimated with {estimation.split(" ")[0]} used to generate malicious samples\n')
                    #application.sensor_data = application.load_sensor_data(phase, setup)
                    classes, modes = pick_classes(phase.split('_')[-1], setup)
                    for mode in modes:
                        print(f'mode: {mode}')
                        #if mode == 'correct_vs_wrong' or mode == 'correct_vs_inversed': continue
                        application.complete_transformer_data = application.create_application_dataset(phase, setup, estimation, classes, modes[mode])
                        application.score_dict[phase + '_setup_' + setup] = application.detection(phase, setup, mode)    #try to classify one correct and one incorrect real sample from the same scenario with the classifier built on the rest of the 14 real correct and 14 simulated incorrect samples

    plt.show()