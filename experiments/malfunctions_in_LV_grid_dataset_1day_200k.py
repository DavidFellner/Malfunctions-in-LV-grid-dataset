import os
import math

'''
Metric goal is reached
'''

# Sytem settings
dev_mode = False
grid_data_folder = os.path.join(os.getcwd(), 'raw_data_generation', 'input')
raw_data_folder = os.path.join(os.getcwd(), 'raw_data')
datasets_folder = os.path.join(os.getcwd(), 'datasets')
test_data_folder = os.path.join(os.getcwd(), 'test')
models_folder = os.path.join(os.getcwd(), 'models')
local_machine_tz = 'Europe/Berlin'  # timezone; it's important for Powerfactory

# Deep learning settings
learning_config = {
    "mode": "train",  # train, eval
    "dataset": "malfunctions_in_LV_grid_dataset_1day_200k",
    "type": "PV",
    # PV, EV, (PV, EV) > if more than one class of device use tuple: number of tuple entries defines number of classes to be detected
    "RNN model settings": [1, 2, 20, 5],
    # number of input features, number of output features, number of features in hidden state, number of of layers
    "LSTM model settings": [1, 2, 3, 5],
    # number of input features, number of output features, number of features in hidden state, number of of layers
    "GRU model settings": [1, 2, 20, 5],
    # number of input features, number of output features, number of features in hidden state, number of of layers
    "Transformer model settings": [2, 1, 1, 3, 4, 0.1],
    # ntoken > 2 outputs, ninp > word/input embedding, nhead, nhid, nlayers, dropout=0.5
    "R-Transformer model settings": [1, 3, 2, 1, 'GRU', 7, 4, 1, 0.1, 0.1],
    # input size, dimension of model,output size, h (heads?), rnn_type ('GRU', 'LSTM', 'RNN'), ksize (key size?), n (# local RNN layers), n_level (how many RNN-multihead-attention-fc blocks), dropout, emb_dropout
    "number of epochs": 20,
    "learning rate": 1 * 10 ** -3,
    "decision criteria": 'majority vote',
    # most informed, majority vote; either the most informed (last output) or the majority of outputs is used for classification
    "calibration rate": 0.8,
    # share of (first) outputs not used for majority vote of each sequence in order to let the network calibrate; between 0 and 1
    "activation function": 'relu',  # relu, tanh
    "mini batch size": 60,
    "optimizer": 'SGD',  # Adam, SGD
    "k folds": 5,  # choose 1 to not do crossval
    "cross_validation": False,
    "early stopping": True,
    "LR adjustment": 'LR controlled',  # None, 'warm up' , 'LR controlled'
    "percentage of epochs for warm up": 10,
    # warm up not performed if percentage of epochs for warm up * epochs > epochs
    "gradient clipping": 0.25,
    "train test split": 1000,  # if int, used as number of testing examples; if float, used as share of data
    "baseline": False,
    "metrics": ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    "cross_val_metrics": ['fit_time', 'test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro'],
    "plot samples": True,
    "classifier": "RNN",  # RNN, LSTM, GRU, Transformer, RTransformer
    "save_model": True,  # saves state dict and optimizer for later use/further training
    "save_result": True,  # saves evaluation result in text file
    "export_model": False,  # for an application
    "do grid search": False,  # grid search for learning parameters optimization
    "grid search": ("calibration rate", [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), # hyperparameter and values to be tried out as tuple  (("calibration rate", [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
    "do hyperparameter sensitivity analysis": False,  # grid search for hyperparameter optimization
    "hyperparameter tuning" :  ("n_heads", [2,4,10]),#("key_size", [1,2,3,4,5,6,7,8,9,10]), #("n_layers", [4, 8, 12, 16])
    "training time sweep" : True, #use to evaluate influence of training time (records score for after each epoch)
}

#########################################################################
###       only change if new dataset or raw data should be created    ###
#########################################################################

# Dataset settings
raw_data_set_name = 'malfunctions_in_LV_grid_dataset'  # 'malfunctions_in_LV_grid_dataset', 'PV_noPV', dummy
detection_methods = True       #to apply classical ML methods
deeplearning = True
type = 'PV'  # PV, EV, (PV, EV)
dataset_available = True  # set to False to recreate instances from raw data
train_test_split = 0.2  # if int, used as number of testing examples; if float, used as share of data
dataset_format = 'HDF'  # HDF, everything else yields CSV
raw_data_available = True  # set to False to generate raw data using the simulation; leave True if DIGSILENT POWRFACTORY is not available
add_data = True  # raw_data_available = False has to be set for this! set add_data = True to add more data to raw data;
add_noise = False
accuracy = 0.01  # accuracy according to the Genauigkeitsklasse of SmartMeter (1 = 1% i.e.)
sample_length = 1 * 96  # 96 datapoints per day
smartmeter_ratedvoltage_range = [400, 415]
smartmeter_voltage_range = [363, 457]
number_of_samples = 200000
share_of_positive_samples = 0.5  # should be 0.5! only chose values that yield real numbers as invers i.e. 0.2, 0.25, 0.5 > otherwise number of samples corrupted
number_of_grids = len([i for i in os.listdir(grid_data_folder) if os.path.isdir(os.path.join(grid_data_folder, i))])
float_decimal = 5  # decimals in dataset

# Powerfactory settings
user = 'FellnerD'
system_language = 1  # chose 0 for english, 1 for german according to the lagnuage of powerfactory installed on the system
parallel_computing = True
QDSL_models_available = True
cores = 12  # cores to be used for parallel computing (when 64 available use 12 - 24)
reduce_result_file_size = True  # save results as integers to save memory in csv
just_voltages = True  # if False also P and Q results given

# Simulation settings
sim_length = 365  # simulation length in days (has to be equal or bigger than sample length
if sim_length < sample_length / 96: print(
    'Choose different simulation length or sample length (sim_length >= sample_length')

if raw_data_set_name == 'PV_noPV':
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
t_start = None  # default(None): times inferred from profiles in data
t_end = None
# t_start = pd.Timestamp('2017-01-01 00:00:00', tz='utc')                                 # example for custom sim time
# t_end = pd.Timestamp('2018-01-01 00:00:00', tz='utc') - pd.Timedelta(step_size + 'T')
