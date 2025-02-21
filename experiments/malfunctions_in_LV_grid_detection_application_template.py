import os

from detection_method_settings import Variables, Mapping_Fluke_to_PowerFactory
v = Variables()
m = Mapping_Fluke_to_PowerFactory()

'''
insert conclusio
'''

# Sytem settings
dev_mode = False
grid_data_folder = os.path.join(os.getcwd(), 'raw_data_generation', 'input')
raw_data_folder = os.path.join(os.getcwd(), 'raw_data')
data_path = os.path.join(os.getcwd(), raw_data_folder, 'ERIGrid-Test-Results-26-11-2021-phase1_final')
sim_data_path = os.path.join(os.getcwd(), raw_data_folder, 'ERIGrid_phase_1_sim_data')
data_path_DSM = os.path.join(os.getcwd(), raw_data_folder, 'ERIGrid-Test-Results-28-04-2022-phase2_final')
sim_data_path_DSM = os.path.join(os.getcwd(), raw_data_folder, 'ERIGrid_phase_2_sim_data')
datasets_folder = os.path.join(os.getcwd(), 'datasets')
test_data_folder = os.path.join(os.getcwd(), 'test')
models_folder = os.path.join(os.getcwd(), 'models')
load_estimation_folder = os.path.join(os.getcwd(), 'load_estimation')
local_machine_tz = 'Europe/Berlin'  # timezone; it's important for Powerfactory

# Deep learning settings
learning_config = {
    'data_source': 'real_world', #real_world, simulation
    'setup_chosen' : {"phase1": {"A", "B"}, "phase2": {}},  # either 'all' for phase 1 setup A+B and phase 2 setup A+B or enter setups to be ommited such as: if phase 1, setup A should be omitted enter {"phase1": {"A"}, "phase2": {}} to omit the entire phase 1 enter {"phase1": {"A", "B"}, "phase2": {}}, if only class 'inversed' should be omitted for setup A enter {"phase1": {"A": ["inversed"]}, "phase2": {}}
    'mode' : 'classification',  # classification means wrong as wrong and inversed as inversed, detection means wrong and inversed as wrong
    'data_mode' : 'combined_data',  # 'measurement_wise', 'combined_data'
    'selection' : 'most important', # 'most important', 'least important' variables picked after assessment by PCA > only applicable when in measurement_wise data mode
    'approach' : 'PCA+clf',  # 'PCA+clf', 'clustering'
    'clf' : 'Assembly', # SVM, NuSVM, kNN, Assembly
    'kernels' : ['linear', 'poly', 'rbf', 'sigmoid'],   # ['linear', 'poly', 'rbf', 'sigmoid'] SVM kernels
    'gammas' : ['scale'],  # , 'auto']#[1/(i+1) for i in range(15)] #['scale', 'auto'] ; regularization for rbf kernels
    'degrees' : list(range(1, 7)),              #degrees for poly kernels
    'neighbours' : [i + 1 for i in range(5)],   #for kNN
    'weights' : ['uniform', 'distance'],        #for kNN
    'classifier_combos' : 'general', # detection, c_vs_w, c_vs_inv, A, c_vs_w_combined_dataset not all work for all!
    'components' : 0.99, #for combined dataset: percentage of variance that is to be retained by primary components
    'disaggregation algorithm' : '[NN, LR]',  #[NN, LR], [NN], [LR]
    'real_vs': 'sim' #est, sim; form negative samples either using the original sim data or the estimated one
}

#########################################################################
###       only change if new dataset or raw data should be created    ###
#########################################################################

# Dataset settings
raw_data_available = True  # set to False to generate raw data using the simulation; leave True if DIGSILENT POWERFACTORY is not available
add_data = True  # raw_data_available = False has to be set for this! set add_data = True to add more data to raw data or fill gaps i scenarios that are not done yet;
#dataset_available = True  # set to False to recreate instances from raw data
load_estimation_training_data_available = True #create input for training of NN for load estimation
pretrained = True #to skip training if training set hasn't changed
load_estimation_input_data_available = True #create input for estimation using NN/LR for load estimation
use_saved_load_estimation_input_data = True
load_estimation_training_data_distributions_dict = {'phase1': {'LB 7 8' : (10, 10/3, 2, -0.095, 0.25), 'LB 2' : (4, 3.5/3, 2, -0.14, 0.25),'PV' : (4, 6/3), 'p_low':1},
                                                    'phase2': {'LB 1' : (3, 21/3, 1.25, -0.375, -4), 'LB 2' : (11, 74/3, 1.75, -0.115, -4), 'LB 3' : (4, 21/3, 1.75, -0.4375, -3.5),'PV' : (4, 6/3), 'p_low': -2}, 'PV' : {'mu':6, 'sigma': 6 / 3}}       #mean and std for gaussian normal distributions to be sampled from for load values; for uniform dist: min: p_low or for q max*4th para / 3rd para, max mu+3*sigma
"""load_estimation_training_data_distributions_dict = {'phase1': {'LB 7 8' : (10, 10/3, 2, -0.095), 'LB 2' : (4, 3.5/3, 2, -0.14),'PV' : (4, 6/3), 'p_low':1},
                                                    'phase2': {'LB 1' : (3, 21/3, 1.25, -0.325, -4), 'LB 2' : (11, 74/3, 2, -0.125, -3.5), 'LB 3' : (4, 21/3, 1.75, -0.45, -2.75),'PV' : (4, 6/3), 'p_low': -2}, 'PV' : {'mu':6, 'sigma': 6 / 3}}"""
#p_low = 1   #absolute value in kW for uniform dist.                #               2.5  -0.11                         -0.2                        5  -0.42                            0.5
#q_low = -0.035    #percent of p value; e.g. 0.1 means 10% of maximum value for uniform dist.
print_loss_and_y_vs_pred = False
plot_real_vs_estimate = False
training_data_dist = 'uniform' #uniform, standard
num_training_samples = 10000 + 1 #+1 in order for PowerFactory to yield the desired number of samples
power_unit_sensor_data = 'KW'
average_estimation_results = True #if false only use results of best model found
detection_methods = False
deeplearning = False
#disaggregation = False
detection_application = True
plot_data = True
use_case = 'DSM' # 'DSM', 'q_control'
test_bays_dict = {'phase1' : ['B1', 'F1', 'F2'], 'phase2' : ['A1', 'B1', 'B2', 'C1']}
load_test_bays_map_dict = {'phase1' : {'LB 2': 'B1', 'LB 7 8': 'F1'}, 'phase2' : {'A1', 'B1', 'B2', 'C1'}}
PV_test_bays_map_dict = {'phase1' : {'A': 'F1', 'B': 'B1'}, 'phase2' : {'A1', 'B1', 'B2', 'C1'}}
data_path_dict = {'phase1' : data_path, 'phase2': data_path_DSM}
sim_data_path_dict = {'phase1' : sim_data_path, 'phase2': sim_data_path_DSM}
if use_case == 'DSM':
    test_bays = ['A1', 'B1', 'B2', 'C1']
    data_path = data_path_DSM
    sim_data_path = sim_data_path_DSM
else:
    test_bays = ['B1', 'F1', 'F2']
extended = True #also add inversed curve to Setup A simulations
save_figures = True # save figures of data
scenario = 14  # 1 to 15 as there is 15 scenarios (profiles)
plot_all = True # whether to plot all scenarios
plot_only_trafo_and_pv = True # whether only the data of the trafo and PV connection test bay should be plotted in 'plot_scenario_test_bay' in 'plot_measurements'
note_avg_and_std = False # whether average and standard deviation should be annotated in 'plot_scenario_test_bay' and 'plot_scenario_case' in 'plot_measurements'
plotting_variables = {'B1': 'Vrms ph-n AN Avg', 'F1': 'Vrms ph-n AN Avg',
                      'F2': 'Vrms ph-n L1N Avg'}  # see dictionary above
if learning_config['data_source'] == 'real_world':
    variables_dict = {'phase1' : {'B1': [v.variables_B1, v.pca_variables_B1], 'F1': [v.variables_F1, v.pca_variables_F1],
                     'F2': [v.variables_F2, v.pca_variables_F2]}, 'phase2': {'B2': [v.variables_B2, v.pca_variables_B2_phase2], 'A1': [v.disaggregation_variables_A1, v.pca_variables_A1_phase2],
                 'B1': [v.variables_B1, v.pca_variables_B1_phase2],
                 'C1': [v.disaggregation_variables_C1, v.pca_variables_B2_phase2]}}
    disaggregation_variables_dict = {'phase1' : {'B1': v.disaggregation_variables_B1,
                                   'F1': v.disaggregation_variables_F1,
                                   'F2': v.disaggregation_variables_F2}, 'phase2': {'A1': v.disaggregation_variables_A1,
                                   'B1': v.disaggregation_variables_B1,
                                   'B2': v.disaggregation_variables_B2,
                                   'C1': v.disaggregation_variables_C1,
                                    'inputs': v.disaggregation_input_variables}}

    pf_to_fluke_map = m.map

    #for phase2
    B2_B1_C1_vars = [[k[0] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]
    B2_B1_C1_vars = B2_B1_C1_vars[0] + B2_B1_C1_vars[1]
    A1_vars = [[k[1] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]
    A1_vars = A1_vars[0] + A1_vars[1]

    #for phase1
    B1_F1_vars = [[k[0] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]
    B1_F1_vars = B1_F1_vars[0] + B1_F1_vars[1]
    F2_vars = [[k[1] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]
    F2_vars = F2_vars[0] + F2_vars[1]
    B2_vars = [[k[1] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]    #same for all trafo points?
    B2_vars = B2_vars[0] + B2_vars[1]

    sim_variables_dict = {'phase1' : {'B1': [v.variables_B1, B1_F1_vars], 'F1': [v.variables_F1, B1_F1_vars],
                     'F2': [v.variables_F2, F2_vars]}, 'phase2': {'B2': [v.variables_B2, B2_B1_C1_vars],
                 'A1': [v.disaggregation_variables_A1, A1_vars],
                 'B1': [v.variables_B1, B2_B1_C1_vars],
                 'C1': [v.disaggregation_variables_C1, B2_B1_C1_vars]}}

    if use_case == 'DSM':
        variables = variables_dict['phase2']
        disaggregation_variables = disaggregation_variables_dict['phase2']

    else:
        variables = variables_dict['phase1']
        disaggregation_variables = disaggregation_variables_dict['phase1']

elif learning_config['data_source'] == 'simulation':
    pf_to_fluke_map = m.map
    B1_F1_vars = [[k[0] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]
    B1_F1_vars = B1_F1_vars[0] + B1_F1_vars[1]
    F2_vars = [[k[1] for k in pf_to_fluke_map[i].values()] for i in pf_to_fluke_map]
    F2_vars = F2_vars[0] + F2_vars[1]
    variables = {'B1': [v.variables_B1, B1_F1_vars], 'F1': [v.variables_F1, B1_F1_vars],
                 'F2': [v.variables_F2, F2_vars]}
else:
    variables = plotting_variables
sampling_step_size_in_seconds = None  # None or 0 to use all data, 1, 20 to sample once every n seconds ....

if detection_application:
    setups = {'phase1': {'A': ['correct', 'wrong', 'inversed'], 'B': ['correct', 'wrong', 'inversed']}, 'phase2': {'A': ['noDSM', 'DSM'], 'B': ['noDSM', 'DSM']}}
elif extended:
    setups = {'Setup_A_F2_data1_3c': ['correct', 'wrong', 'inversed'],
              'Setup_A_F2_data2_2c': ['correct', 'wrong'],
              'Setup_A_F2_data3_2c': ['correct', 'inversed'],
              'Setup_B_F2_data1_3c': ['correct', 'wrong', 'inversed'],
              'Setup_B_F2_data2_2c': ['correct', 'wrong'],
              'Setup_B_F2_data3_2c': ['correct', 'inversed']}
else:
    setups = {'Setup_A_F2_data': ['correct', 'wrong'], 'Setup_B_F2_data1_3c': ['correct', 'wrong', 'inversed'],
              'Setup_B_F2_data2_2c': ['correct', 'wrong'], 'Setup_B_F2_data3_2c': ['correct', 'inversed'],
              'Setup_A_B2_DSM': ['noDSM', 'DSM'], 'Setup_B_B2_DSM': ['noDSM', 'DSM']}


# additional settings : necessary here after?
sample_length = 7 * 96  # 96 datapoints per day
number_of_samples = 200000
share_of_positive_samples = 0.5  # should be 0.5! only chose values that yield real numbers as invers i.e. 0.2, 0.25, 0.5 > otherwise number of samples corrupted
number_of_grids = len([i for i in os.listdir(grid_data_folder) if os.path.isdir(os.path.join(grid_data_folder, i))])
float_decimal = 5  # decimals in dataset

# Powerfactory settings
user = 'FellnerD'
system_language = 1  # chose 0 for english, 1 for german according to the lagnuage of powerfactory installed on the system
parallel_computing = True
QDSL_models_available = True
cores = 4  # cores to be used for parallel computing (when 64 available use 12 - 24)
reduce_result_file_size = True  # save results as integers to save memory in csv
just_voltages = False  # if variables defined in file are used

# Simulation settings
sim_setting = 'ERIGrid_phase_1'
if sim_setting == 'ERIGrid_phase_1': pf_file = 'PNDC_ERIGrid_phase1'
pf_file_dict = {'phase1': 'PNDC_ERIGrid_phase1', 'phase2': 'PNDC_ERIGrid_phase2'}
resolution = '15T' #resolution of load/generation profiles to be used
t_start = None  # default(None): times inferred from profiles in data
t_end = None
step_size = 3
step_unit = 0 #0...seconds, 1... minutes
balanced = 1 # 0... AC Load Flow, balanced, positive sequence, 1... imbalanced 3 phase

sim_length = 365  # simulation length in days (has to be equal or bigger than sample length
if sim_length < sample_length / 96: print(
    'Choose different simulation length or sample length (sim_length >= sample_length')

"""if raw_data_set_name == 'PV_noPV':
    positive_samples_per_simrun = 5  # data from how many terminals are there in the grid minimally > determines how many yearly simulations have to be run and used for dataset creation
    simruns = math.ceil(
        (number_of_samples * share_of_positive_samples) / (positive_samples_per_simrun * number_of_grids) / (
                    sim_length * 96 / sample_length))
elif raw_data_set_name == 'malfunctions_in_LV_grid_dataset':
    simruns = math.ceil((number_of_samples * share_of_positive_samples) / (number_of_grids) / int(
        (sim_length * 96 / sample_length) - 1))
elif raw_data_set_name == 'dummy':
    terms_per_simrun = 5  # data from how many terminals are there in the grid minimally > determines how many yearly simulations have to be run and used for dataset creation
    simruns = math.ceil((number_of_samples * share_of_positive_samples) / (terms_per_simrun * number_of_grids) / (
                sim_length * 96 / sample_length))
else:
    simruns = 10  # number of datasets produced and also used per grid (location of malfunction/PVs... is varied)
step_size = 15  # simulation step size in minutes
percentage = {'PV': 0,
              'EV': 25, 'BESS': 0, 'HP': 0}  # percentage of busses with active PVs (PV proliferation), EV charging stations etc...
control_curve_choice = 0  # for all PVs: choose control curve for 0 = cos(phi)(P), 1 = Q(P), 2 = brokenQ(P) (flat curve)
broken_control_curve_choice = 3  # for broken PV: choose control curve for 0 = cos(phi)(P), 1 = Q(P), 2 = broken Q(P) (flat curve), 3 = wrong Q(P) (inversed curve)
number_of_broken_devices_and_type = (1, 'EV')  # define number of devices to experience malfunctions and type (PV, EV, BESS, HP) during simulation
load_scaling = 100  # general load scaling for all loads in simulation (does not apply to setup)
generation_scaling = 100  # general generation scaling for all generation units in simulation (does not apply to setup)
whole_year = True  # if True malfunction is present from start of simulation on; if False malfunction is at random point
"""
# t_start = pd.Timestamp('2017-01-01 00:00:00', tz='utc')                                 # example for custom sim time
# t_end = pd.Timestamp('2018-01-01 00:00:00', tz='utc') - pd.Timedelta(step_size + 'T')
