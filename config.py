import os

#Sytem settings
data_folder = os.getcwd() + '\\input\\'
results_folder = os.getcwd() + '\\output\\'
local_machine_tz = 'Europe/Berlin'                          #timezone; it's important for Powerfactory

#Deep learning settings
learning_config = {
    "dataset": "malfunctions_in_LV_grid",
    "malfunction_in_LV_grid_data": os.getcwd() + '\\output\\malfunctions_in_LV_grid_dataset.csv',
    "test_data_set": os.getcwd() + '\\test\\malfunctions_in_LV_grid_dataset.csv',
    "RNN model settings" : [1, 1, 6, 2],   # number of input features, number of output features, number of features in hidden state, number of of layers
    "plot_confusion_matrix": False,
    "plot_learning_curve": False,
    "plot_validation_curve": False,
    "cross_validation": True,
    "metrics": ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    "cross_val_metrics": ['fit_time', 'test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro'],
    "grid_search": False,
    "classifier": "RNN"  # RNN
}

# Dataset settings
data_set_name = 'malfunctions_in_LV_grid_dataset.csv'
raw_data_available = True                       #set to False to generate raw data using the simulation
dataset_available = True                       #set to False to recreate instances from raw data
accuracy = 0.01                                 #accuracy according to the Genauigkeitsklasse of SmartMeter (1 = 1% i.e.)
smartmeter_ratedvoltage_range = [400, 415]
smartmeter_voltage_range = [363, 457]
number_of_samples = 40
share_of_malfunction_samples = 0.25        #only chose values that yield real numbers as invers i.e. 0.2, 0.25, 0.5 > otherwise number of samples corrupted
number_of_grids = len([i for i in os.listdir(data_folder) if os.path.isdir(data_folder + i)])

#Powerfactory settings
user = 'FellnerD'
system_language = 0                     #chose 0 for english, 1 for german according to the lagnuage of powerfactory installed on the system
parallel_computing = True
cores = 12                              #cores to be used for parallel computing (when 64 available use 12 - 24)
reduce_result_file_size = True          #save results as integers to save memory in csv
just_voltages = True                    # if False also P and Q results given

# Simulation settings
start = 0                               #start = 5 yields result_run#5
simruns = (number_of_samples * share_of_malfunction_samples) / number_of_grids  #number of datasets produced and also used per grid (location of malfunction is varied)
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