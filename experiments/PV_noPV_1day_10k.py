import os
import math

'''
Metric goal is NOT reached
'''

#Sytem settings
data_folder = os.path.join(os.getcwd(), 'input')
results_folder = os.path.join(os.getcwd(), 'output')
test_data_folder = os.path.join(os.getcwd(), 'test')
models_folder = os.path.join(os.getcwd(), 'models')
local_machine_tz = 'Europe/Berlin'                          #timezone; it's important for Powerfactory

#Deep learning settings
learning_config = {
    "dataset": "PV_noPV_1day_10k",
    "RNN model settings": [1, 2, 6, 2],     # number of input features, number of output features, number of features in hidden state, number of of layers
    "number of epochs": 100,
    "learning rate": 1*10**-6,
    "activation function": 'tanh',          # relu, tanh
    "mini batch size": 60,
    "optimizer": 'Adam',                    # Adam, SGD
    "k folds": 5,                           #choose 1 to not do crossval
    "cross_validation": True,
    "early stopping": True,
    "LR adjustment": 'LR controlled',               #None, 'warm up' , 'LR controlled'
    "percentage of epochs for warm up": 10,         #warm up not performed if percentage of epochs for warm up * epochs > epochs
    "train test split": 0.2,                        #if int, used as number up testing examples; if float, used as share of data
    "baseline": True,
    "metrics": ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    "cross_val_metrics": ['fit_time', 'test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro'],
    "plot samples": True,
    "classifier": "RNN"  # RNN

}

#########################################################################
###       only change if new dataset or raw data should be created    ###
#########################################################################

# Dataset settings
raw_data_set_name = 'PV_noPV'                   #'malfunctions_in_LV_grid_dataset', 'PV_noPV', dummy
dataset_available = True                       #set to False to recreate instances from raw data
raw_data_available = True                      #set to False to generate raw data using the simulation; leave True if DIGSILENT POWRFACTORY is not available
add_data = True                                #raw_data_available = False has to be set for this! set add_data = True to add more data to raw data;
add_noise = False
accuracy = 0.01                                 #accuracy according to the Genauigkeitsklasse of SmartMeter (1 = 1% i.e.)
sample_length = 1 * 96                          #96 datapoints per day
smartmeter_ratedvoltage_range = [400, 415]
smartmeter_voltage_range = [363, 457]
number_of_samples = 10000
share_of_positive_samples = 0.5        #should be 0.5! only chose values that yield real numbers as invers i.e. 0.2, 0.25, 0.5 > otherwise number of samples corrupted
number_of_grids = len([i for i in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, i))])
float_decimal = 5                       #decimals in dataset

#Powerfactory settings
user = 'FellnerD'
system_language = 0                     #chose 0 for english, 1 for german according to the lagnuage of powerfactory installed on the system
parallel_computing = True
cores = 12                              #cores to be used for parallel computing (when 64 available use 12 - 24)
reduce_result_file_size = True          #save results as integers to save memory in csv
just_voltages = True                    #if False also P and Q results given

# Simulation settings
sim_length = 365                         #simulation length in days (has to be equal or bigger than sample length
if sim_length < sample_length/96: print('Choose different simulation length or sample length (sim_length >= sample_length')

if raw_data_set_name == 'PV_noPV':
    positive_samples_per_simrun = 5     #data from how many terminals are there in the grid minimally > determines how many yearly simulations have to be run and used for dataset creation
    simruns = math.ceil((number_of_samples * share_of_positive_samples) / (positive_samples_per_simrun * number_of_grids) / (sim_length * 96/sample_length))
elif raw_data_set_name == 'malfunctions_in_LV_grid_dataset':
    simruns = math.ceil((number_of_samples * share_of_positive_samples) / (number_of_grids) / int(
        (sim_length * 96 / sample_length) - 1))
elif raw_data_set_name == 'dummy':
    terms_per_simrun = 5                #data from how many terminals are there in the grid minimally > determines how many yearly simulations have to be run and used for dataset creation
    simruns = math.ceil((number_of_samples * share_of_positive_samples) / (terms_per_simrun * number_of_grids) / (sim_length * 96/sample_length))
else:
    simruns = 10                        #number of datasets produced and also used per grid (location of malfunction/PVs... is varied)
step_size = 15                          #simulation step size in minutes
percentage = 25                         #percentage of busses with active PVs (PV proliferation)
control_curve_choice = 0                #for all PVs: choose control curve for 0 = cos(phi)(P), 1 = Q(P), 2 = brokenQ(P) (flat curve)
broken_control_curve_choice = 3         #for broken PV: choose control curve for 0 = cos(phi)(P), 1 = Q(P), 2 = broken Q(P) (flat curve), 3 = wrong Q(P) (inversed curve)
number_of_broken_devices = 1            #define number of devices to expereince malfunctions during simulation
load_scaling = 100                      #general load scaling for all loads in simulation (does not apply to setup)
generation_scaling = 100                #general generation scaling for all generation units in simulation (does not apply to setup)
whole_year = True                       #if True malfunction is present from start of simulation on; if False malfunction is at random point
t_start = None                          #default(None): times inferred from profiles in data
t_end = None
#t_start = pd.Timestamp('2017-01-01 00:00:00', tz='utc')                                 # example for custom sim time
#t_end = pd.Timestamp('2018-01-01 00:00:00', tz='utc') - pd.Timedelta(step_size + 'T')