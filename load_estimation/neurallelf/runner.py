''' 
Data loading, parameter selection, training and evaluating artificial neural networks for determination of unknown values from 
known values at the buses of a grid.
'''
import os,json
from pathlib import Path
import argparse
import json
import time
from datetime import date

from sklearn.model_selection import ParameterGrid

from neurallelf.data.dataset_generators import DatasetTestGrids, DatasetGasen
from neurallelf.features.feature import *
from neurallelf.models.load_flow import *
from neurallelf.models.load_estimation import *
from neurallelf.visualization.load_estimation_viz import *

# set working directory 
wdir = Path(os.getcwd()) / Path(__file__)
wdir = wdir.parents[1]
#os.chdir(wdir)
print(os.getcwd())

### Alternative 1: Parameters specifications from arguments
################################################################

parser = argparse.ArgumentParser(description='Load estimation')

parser.add_argument('--dir_results', dest='dir_results', type=str, help='Directroy name for results')
parser.add_argument('--le_model_dir', dest='le_model_dir', type=str, help='Directroy with trained models for mode LFfromLE')
parser.add_argument('--mode', dest='mode', type=str, help='Mode')
parser.add_argument('--dir_name', dest='dir_name', type=str, help='General directory name pattern of the data')
parser.add_argument('--name_list', dest='name_list', type=str, nargs='+', default=[], help='Individual directory name pattern of the data')
parser.add_argument('--name_single', dest='name_single', type=str, default='', help='Individual name of directory')
parser.add_argument('--known_buses', dest='known_buses', type=str, nargs='+', default=[], help='Names of known buses')
parser.add_argument('--file_pattern_loads', dest='file_pattern_loads', type=str, help='Data file name pattern for loads')
parser.add_argument('--file_pattern_voltages', dest='file_pattern_voltages', type=str, help='Data file name pattern for voltages')
parser.add_argument('--file_pattern_pflow', dest='file_pattern_pflow', type=str, help='Data file name pattern for pflow')
parser.add_argument('--file_pattern_qflow', dest='file_pattern_qflow', type=str, help='Data file name pattern for qflow')
parser.add_argument('--graph_pattern', dest='graph_pattern', type=str, nargs='+', default=None, help='Graph data file name pattern parts')
parser.add_argument('--graph_name', dest='graph_name', type=str, default='', help='Graph data file name')
parser.add_argument('--attempts', dest='attempts', type=int, help='Number of scenarios per fraction')
parser.add_argument('--n_known', dest='n_known', type=int, help='Number of known loads (buses) per scenario')
parser.add_argument('--cv', dest='cv', type=int, help='Cross validation runs')
parser.add_argument('--nodes_from_X', '--nodes_from_X', action='store_false') 
parser.add_argument('--inodes', dest='inodes', type=int, help='Number of nodes in the hidden layers')
parser.add_argument('--hidden_layers', dest='hidden_layers', type=int, nargs='+', default=[], help='Hidden layer configuration')
parser.add_argument('--activation', dest='activation', type=str, nargs='+', default=[], help='Activation function')
parser.add_argument('--optimizer', dest='optimizer', type=str, nargs='+', default=[], help='Optimizer')
parser.add_argument('--loss', dest='loss', type=str, nargs='+', default=[], help='Loss function')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, nargs='+', default=[], help='Learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int, nargs='+', default=[], help='Batch size')

args = parser.parse_args()
print(args)

dir_results = args.dir_results
print(dir_results)
le_model_dir = args.le_model_dir
print(le_model_dir)
mode=args.mode 
print(mode)
dir_name=args.dir_name
print(dir_name)
name_list=args.name_list
print(name_list)
name_single=args.name_single
print(name_single)
known_buses=args.known_buses
print(known_buses)
file_pattern = {'loads':args.file_pattern_loads,
                'voltages':args.file_pattern_voltages,
                'pflow':args.file_pattern_pflow,
                 'qflow':args.file_pattern_qflow}
print(file_pattern)
graph_pattern=args.graph_pattern
print(graph_pattern)
graph_name = args.graph_name
print(graph_name)
attempts=args.attempts
print(attempts)
n_known=args.n_known
print(n_known)
cv=args.cv
print(cv)
nodes_from_X=args.nodes_from_X
print(nodes_from_X)
inodes=args.inodes
print(inodes)
hidden_layers=args.hidden_layers
print(hidden_layers)
activation    = tuple(args.activation)
print(activation)
optimizer     = args.optimizer
optimizer     = tuple(optimizer)
print(optimizer)
loss          = tuple(args.loss) 
print(loss)  
learning_rate = tuple(args.learning_rate)
print(learning_rate)
batch_size    = tuple(args.batch_size)
print(batch_size)


def load_estimation(dir_results=None, setup='A'):

    ### Alternative 2: Parameters specifications within the script
    ################################################################
    from datetime import date

    dir_results = f"ERIGrid_{dir_results.split('_')[-1]}"
    dir_results_models = f"ERIGrid_{dir_results.split('_')[-1]}_Setup_{setup}"
    if dir_results is None:
        # specify the directory for the run results:
        dir_results = "ERIGrid_phase1"
        dir_results_models = "ERIGrid_phase1_Setup_A"

    mode = 'LE'           # either of 'LE','DLEF','DLF','BusLF','LF' or 'LFfromLE'
    date = date.today()
    le_model_dir = f"{date.year}_{date.month}_{date.day}_LE_base"    # required only for mode 'LFfromLE'

    # specify datasets, directory is dir_name+name from name_list
    dir_name = f'raw_{dir_results}_' #'raw_grid' # 'raw_Gasen_'
    #dir_name = dir_name  # 'raw_grid' # 'raw_Gasen_'
    #???? name_list = ['v4_13June']#,'02','03','04','05','06'] # ['v4']
    name_list = ['training_data']  # ,'02','03','04','05','06'] # ['v4']
    #name_single = os.path.join('\\'.join(os.getcwd().split('\\')[:-1]), f'raw_data\\{dir_results}_training_data\\PNDC_{dir_results}\\Load_estimation_training_data\\load_estimation_training_data.csv')  # use this in case of a single datafile, else put None
    name_single = os.path.join(os.getcwd(),
                              f'raw_data\\{dir_results}_training_data\\PNDC_{dir_results}_training\\Load_estimation_training_data\\load_estimation_training_data_setup_{setup}.csv')  # use this in case of a single datafile, else put None

    os.path.isfile(name_single)

    # specify always 'known' buses by original dataset name
    # (known includes: voltage, pflow and qflow)
    if dir_results.split('_')[1] == 'phase1':
        trafo_point = 'F2'  # pv use case
        known_buses = ['Test Bay ' + trafo_point, "Test Bay B1", "Test Bay F1"]  # [trafo_point,'node_57']
    else:
        trafo_point = 'B2'  # dsm use case
        known_buses =  ['Test Bay ' + trafo_point, "Test Bay B1", "Test Bay A1", "Test Bay C1"] #[trafo_point,'node_57']                    #e.g. 'node01' these are not included in mode 'LF'
    #scen_string = "[ [\"Test Bay F2_q\", \"Test Bay F2_p\", \"Test Bay F2_V\", \"Test Bay B1_V\", \"Test Bay F1_V\"], []]"
    scen_string = "[[\"PV A_P\", \"PV A_Q\", \"PV B_P\", \"PV B_Q\"]]"#"[ [\"LB 2_P\", \"LB 2_Q\", \"LB 7 8_P\", \"LB 7 8_Q\"]]"
    jsonString = json.dumps(scen_string)
    if not os.path.isdir(os.path.join(os.getcwd(), 'model summaries', dir_results_models)):
        os.mkdir(os.path.join(os.getcwd(), 'model summaries', dir_results_models))
    jsonFile = open(f"{os.path.join(os.getcwd(), 'model summaries', dir_results_models, 'scenarios gridtraining_data')}.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    # optional: depending on the Dataset to use
    #file_pattern = {'loads':'full_df_load.csv','voltages':'full_df_voltage.csv','pflow':'full_df_Pflow.csv', 'qflow':'full_df_Qflow.csv'}
    #file_pattern = {'loads':'narrow_df_load.csv','voltages':'narrow_df_voltage.csv','pflow':'narrow_df_Pflow.csv', 'qflow':'narrow_df_Qflow.csv'}
    #file_pattern = {'loads':'df_load.csv','voltages':'df_voltage.csv','pflow':'df_Pflow.csv', 'qflow':'df_Qflow.csv'}

    # depending on the Dataset to use specify the graph data:
    #graph_pattern = ['CLUE_Test_','_Load_Bus_Mapping.txt']
    graph_name = os.path.join(os.getcwd(),
                              f'raw_data\\{dir_results}_training_data\\PNDC_{dir_results}\\Load_estimation_training_data\\{dir_results}_Load_Bus_Mapping.txt')


    # specify parameters for run (attempts needs to fit a given scenarios file)
    attempts = 1  # number of random scenarios per fraction of unknown loads, for injected scenarios enter the total number of scenarios
    n_known = 2  # the number of known loads to consider in scenario creation, if n_known=-1 all factions of known loads will be created
    cv = 3        # number of cross validation runs

    # specify hyperparameter grid
    nodes_from_X = True    # if True then input layer of network has nodes equal to features in training data; overwrites inodes value below
    inodes = 0
    hidden_layers = [1]         # the amount of hidden layers in different runs, idx is the run
    activation    = ('relu',)
    #import tensorflow as tf
    #activation    = (tf.keras.layers.LeakyReLU(),)
    optimizer     = ('adam',)
    #optimizer     = ('rmsprop',)
    loss = ('mse',)
    #loss          = ('mse',)
    learning_rate = (0.001,)
    batch_size    = (32,)





    # specify important paths
    data_path = Path("data")
    results_path = Path("model summaries")
    results_path.mkdir(exist_ok=True)

    run_path = results_path / dir_results_models
    run_path.mkdir(exist_ok=True)



    ### Raw data loading
    ###################################

    # load the data into dataset objects stored in a dataset_dict
    dataset_dict = {}

    # option 1: single or multiple input files in directories defined by name_list and a predefined Dataset generator
    for name_str in name_list:
        directory = dir_name + name_str
        dataset = DatasetGasen()
        dataset.create_dataset(data_path,directory,name_single,graph_name)
        #dataset = DatasetTestGrids()
        #dataset.create_dataset(data_path,directory,name_str,file_pattern,graph_pattern)
        dataset.known_buses = [known_bus+'_V' for known_bus in known_buses]
        dataset_dict[name_str] = dataset

    # option 2: put your own loading method here (needs to use the Dataset class)



    ### Scenarios and modes
    ############################################
    #modes: 'LE','DLEF','DLF','BusLF','LF' or 'LFfromLE'
    # scenarios can be created randomly or they can be read from a scenarios-file in the run_path
    ledto_dict = {}
    results_dict_LE_run = {}

    for name in name_list:

        if mode in ['LE','DLEF','DLF','BusLF']:
            ledto = load_or_create_scenarios(run_path,name,dataset_dict[name],attempts,n_known)
            if ledto is not None: ledto_dict[name] = ledto
        elif mode=='LFfromLE':
            ledto_dict[name] = load_scenarios_LFfromLE(run_path,results_path,le_model_dir,name,dataset_dict[name],attempts,n_known)
            results_dict_LE_run[name]=load_ledto(results_path / le_model_dir,name,dataset_dict[name],load_models_too=True)
        elif mode=='LF':
            ledto = LEDTO([[]])
            ledto_dict[name]=ledto
        else:
            print('Mode undefined!')
            raise(ValueError)


    ### Model training
    ###################################

    # train neural networks
    results_dict = {}
    for name,ledto in ledto_dict.items():
        dataset = dataset_dict[name]
        known_buses_ind = [dataset.v_df.columns.get_loc(known_element) for known_element in dataset.known_buses]
        save_ledto_scenarios(ledto,dataset,path=run_path,name=f"grid{name}")

        # train models for each scenario:
        if (ledto.last_index==0) or (ledto.last_index < len(ledto.pq_ind_known)-1):
            for idx,scenario in enumerate(ledto.pq_ind_known[ledto.last_index:],start=ledto.last_index):
                pq_known = scenario
                if mode=='LFfromLE':
                    v_known = ledto.v_ind_known[idx]
                    X_pr, y_pr, _, _ = select_feature_label_scaled_scenario(pq_known,v_known,dataset,'LE',known_buses_ind)
                    nndto = results_dict_LE_run[name].nndto_list[idx]
                    nndto.df_avg, nndto.best_model_path = get_best_model(nndto.results_df,nndto.results_models_paths,True)
                    nndto.best_model = load_specific_models(results_path / le_model_dir,nndto.best_model_path)
                    # for now only the first model is used for the prediction
                    prediction_LE = nndto.best_model[0].predict(X_pr.values)
                    dataset.pq_df = replace_columns_with_prediction(dataset,prediction_LE,y_pr.columns)
                    pq_known = range(len(dataset.pq_df.columns))   # all loads are now known
                    X_le, y_le, _, _ = select_feature_label_scaled_scenario(pq_known,v_known,dataset,'DLF',known_buses_ind)
                elif mode in ['LE','DLEF','DLF','BusLF']:
                    v_known = ledto.v_ind_known[idx]
                    X_le, y_le, scaler_y, scaler_X = select_feature_label_scaled_scenario(pq_known,v_known,dataset,mode,known_buses_ind)
                elif (mode=='LF'):
                    X_le, y_le, _ = select_feature_label_scaled_LF(dataset)   #known_buses are not included here

                if True:
                    report_column_names(X_le,y_le,run_path,scenario_name=idx)
                    save_scaler(scaler_y, scaler_X,run_path,scenario_name=idx)

                X_le = X_le.values
                y_le = y_le.values

                """import matplotlib.pyplot as plt
                plt.figure()
                ax = plt.gca()
                ax.boxplot(X_le)
                plt.figure()
                ax = plt.gca()
                ax.boxplot(y_le)"""

                # define grid parameters for NN:
                if nodes_from_X:
                    inodes = X_le.shape[1]
                layers_nodes = create_layers(hidden_layers,inodes)
                parameters ={
                    'layers_nodes':layers_nodes,
                    'activation':activation,
                    'opt':optimizer,
                    'loss':loss,
                    'learning_rate':learning_rate,
                    'batch_size':batch_size,
                }
                grid = ParameterGrid(parameters)
                nndto = NNDTO()
                nndto.gridparams = parameters
                nndto.results_df, nndto.results_models, nndto.Xtest, nndto.ytest  = grid_cv_neural_parallel(grid,X_le,y_le,cv=cv)
                nndto.fraction_known = int(len(scenario)/2)/(dataset_dict[name].pq_df.shape[1]*0.5)
                ledto.nndto_list.append(nndto)
                save_nndto(nndto,path=run_path,name=f"grid{name}_{idx}")
        results_dict[name]= ledto


    ### Write log-file
    ###################################
    logfile = run_path / 'A_log.txt'
    with open(logfile,'w') as log:
        if name_single:
            filename = name_single
            graph = graph_name
        else:
            filename = file_pattern
            graph = graph_pattern
        log.write(
            f"Working directory: {os.getcwd()}\n"\
            f"Directory: {dir_results}\n"\
            f"Mode: {mode}\n"\
            f"Name_list: {name_list}\n"\
            f"Dir_name: {dir_name}\n"\
            f"File_names: {filename}\n"\
            f"Graph path: {graph}\n"\
            f"Attempts: {attempts}\n"\
            f"N_known: {n_known}\n"\
            f"Inodes: {inodes}\n"\
            f"CV: {cv}\n"    )
