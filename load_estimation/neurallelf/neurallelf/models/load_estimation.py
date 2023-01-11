''' 
A module for creating and running load estimation scenarios.
'''
import joblib
import numpy as np
import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

from ..models.load_flow import *


class LEDTO:
    ''' 
    Data transfer object carries information about scenarios in state estimation.
    '''
    def __init__(self,pq_ind_known):
        self.pq_ind_known = pq_ind_known                        # list of list of column indices of known loads for scenarios
        self.v_ind_known = None                                 # list of list of column indices of known buses for scenarios
        self.nndto_list = []
        self.last_index = 0
        
        self.metrics_summary_df = None


def save_ledto_scenarios(ledto,dataset, path='',name=''):
    ''' 
    Stores fields from a LEDTO object at path. Stored is pq_ind_known.
    '''
    path.mkdir(exist_ok=True)
    #scen_decoded = scenarios_encoding(dataset,ledto.pq_ind_known,encode=False)
    with open(path / f"scenarios {name}.json",'w') as outfile:
        output = scenarios_encoding(dataset=dataset,scenarios=ledto.pq_ind_known,encode=False)
        write_str = json.dumps(output)
        json.dump(write_str,outfile)


def save_ledto_metrics_summary(ledto,path='',name=''):
    ''' 
    Stores fields from a LEDTO object at path. Stored are the pq_ind_known and pq_ind_unknown.
    '''
    path.mkdir(exist_ok=True)
    try:
        ledto.metrics_summary_df.to_csv(path / f"metrics_summary_df {name}.csv")
    except:
        print("No metrics_summary_df found to store.")
    for idx,nndto in enumerate(ledto.nndto_list):
        save_nndto_metric_df(nndto,path,name,idx)


def load_scenarios(model_path,name,dataset):
    '''
    Load stored data on the scenarios of the LE run from json-files.
    '''
    # load pq_ind_known
    _ , scenario_file = search_scenario_file(model_path,name)
    with open(model_path / scenario_file) as jsonfile:
        scen_string = json.load(jsonfile)
        #print(type(json.loads(scen_string.splitlines()[0])))
        pq_ind_known = []
        scen_list = json.loads(scen_string)
        a = 1
        for sublist in scen_list:
            #pq_ind_known.append([int(item) for item in sublist]) #befor version 3.0.0
            pq_ind_known.append([item for item in sublist])
        pq_ind_known = scenarios_encoding(dataset,pq_ind_known,encode=True)
    return pq_ind_known


def scenarios_encoding(dataset,scenarios,encode):
    '''
    Encode / decode the scenarios file information. 
    Args:
        encode (boolean): if True encode original columns names to column indices, if False do it reversed
        scenarios (list): a list of lists of the scenarios to encode or decode 
    '''
    if encode is True:
        # from original name of column to indices
        #scenarios_encoded = [[dataset.load_encode_dict[element] for element in sublist] for sublist in scenarios]
        scenarios_transformed = [sorted([dataset.pq_df.columns.get_loc(element) for element in sublist]) for sublist in scenarios]
    elif encode is False:
        # from indices to original column names
        scenarios_transformed = [[dataset.pq_df.columns[element] for element in sublist] for sublist in scenarios]
        #scenarios_transformed = [[dataset.load_decode_dict[element] for element in sublist] for sublist in scenarios_encoded]
    return scenarios_transformed


def load_ledto(model_path,name,dataset,load_models_too=False):
    '''
    Load stored data in model_path to an LEDTO and return in.
    '''
    ledto = LEDTO(None)

    # load metrics_summary_df
    try:
        metrics_summary_df_path = [path for path in os.listdir(model_path) if ("metrics_summary_df" in path) and (name in path)]
        ledto.metrics_summary_df = pd.read_csv(model_path / metrics_summary_df_path[0] ,index_col=[0])
    except:
        pass
    
    # load scenario information from json files
    ledto.pq_ind_known = load_scenarios(model_path,name,dataset)

    # load the nndtos into nndto_list
    nndto_list = []
    for id in range(len(ledto.pq_ind_known)):
        nndto_list.append(load_nndto(model_path,name,id,load_models_too))
    ledto.nndto_list = nndto_list

    return ledto


def search_scenario_file(run_path,name):
    '''
    Look for a scenarios-file in run_path with 'name.json' ending and return it if present once.
    '''
    if name=='':
        return (False,'')
    pattern = r'scenarios\s\w*'+name+r'\.json'
    scenario_file = [file for file in os.listdir(run_path) if len(re.findall(pattern,file))>0]
    if len(scenario_file)==1:
        return (True,scenario_file[0])
    else: 
        return (False,'')


def scenario_validation(pq_ind_known,attempts,n_known,pq_df):
    '''
    Assert that the scenario file contents (pq_ind_known) fits the run parameters
    defined through attempts and n_known. It does not check the contents of the elements though.
    '''
    if n_known==-1:
        n_loads = int(len(pq_df.columns)/2)
        length_expected = attempts*(n_loads-1)
        if length_expected == len(pq_ind_known):
            return True
        else:
            return False
    elif n_known!=-1:
        if (len(pq_ind_known)==attempts):
            return True
        else: 
            return False


def confirm_models_results_params(run_path,name):
    '''
    Confirm that a path contains files including the name and of type 'model...h5', 'parameters...json' and 
    'results_df...csv'. The correct number of files is not investigated.
    '''
    if name=='':
        raise ValueError('No name received.')
    pattern_model = r'model\s\w*'+name+r'_\w+\.h5'
    pattern_parameters = r'parameters\s\w*'+name+r'_\w+\.json'
    pattern_resultsdf = r'results_df\s\w*'+name+r'_\w+\.csv'
    model_file = [file for file in os.listdir(run_path) if len(re.findall(pattern_model,file))>0]
    parameters_file = [file for file in os.listdir(run_path) if len(re.findall(pattern_parameters,file))>0]
    results_file = [file for file in os.listdir(run_path) if len(re.findall(pattern_resultsdf,file))>0]
    if (len(model_file)>0) and (len(parameters_file)>0) and (len(results_file)>0):
        return True
    else:
        return False



##  Scenario creation functions 
#####################################################

def look_for_valid_scenarios(run_path,name,attempts,n_known,dataset):
    '''
    Checks if the run_path contains scenarios files of type 'scenarios... .json'. 
    Found scenarios in a file are checked to fit the number of attempts and n_known. 
    If the scenarios are valid a check is done if training results are present too.
    
    Returns:
        scenarios_found (bool): True if a scenarios json file was found
        scenarios_valid (bool): True if scenario file content is validated
        models_results_parameters_present (bool): True if training results files are present
    '''
    scenario_found = scenarios_valid = models_results_parameters_present = False
    scenario_found, _ = search_scenario_file(run_path,name)
    if scenario_found:
        pq_ind_known = load_scenarios(run_path,name,dataset)
        scenarios_valid = scenario_validation(pq_ind_known,attempts,n_known,dataset.pq_df)    
        if scenarios_valid:
            # find out if there are already models and results for a name (the number is not checked)
            models_results_parameters_present = confirm_models_results_params(run_path,name) 
    return scenario_found, scenarios_valid, models_results_parameters_present


def load_or_create_scenarios(run_path,name,dataset,attempts,n_known):
    '''
    Create random scenarios or load scenarios from a 'scenarios... .json' file at run_path.
    The run_path is also checked for results from a previous run - if found the scenarios won't be
    rerun.
    '''
    scenario_found, scenarios_valid, _ = look_for_valid_scenarios(run_path,name,attempts,n_known,dataset)
    model_index_list = [element[1][0] for element in get_all_model_ids(run_path,name)]
    last_index = max(model_index_list) if len(model_index_list)>0 else 0

    if (scenario_found==False):
        print(f'The directory for the run does not contain valid results for {name} - scenarios will be created randomly.')
        ledto = create_random_scenarios(dataset.pq_df,dataset.v_df,attempts,n_known,dataset.graph_df)
        return ledto
    elif (scenario_found==True) and (scenarios_valid==False):
        raise ValueError(f'The scenario file for {name} does not fit the run parameters.')
    elif (scenario_found==True) and (scenarios_valid==True):
        print(f'Scenario information found for {name} - scenarios from file will be used.')
        pq_ind_known = load_scenarios(run_path,name,dataset)
        ledto = create_ledto(pq_ind_known,dataset.graph_df,dataset.pq_df,dataset.v_df)
        ledto.last_index = last_index
        return ledto
    else: 
        print(f'Run appears to be already done for {name}')
        return None


def load_scenarios_LFfromLE(run_path,results_path,le_model_dir,name,dataset,attempts,n_known):
    '''
    Load scenarios from a 'scenarios... .json' file at results_path/le_model_dir as path.
    The path is also checked for parameter compatibility.
    '''
    path = results_path / le_model_dir
    scenario_found, scenarios_valid, models_results_parameters_present = look_for_valid_scenarios(path,name,attempts,n_known,dataset)
    model_index_list = [element[1][0] for element in get_all_model_ids(run_path,name)]
    last_index = max(model_index_list) if len(model_index_list)>0 else 0

    if (scenario_found==False):
        raise ValueError(f'The scenario file for {name} was not found.')
    elif (scenario_found==True) and (scenarios_valid==False):
        raise ValueError(f'The scenario file for {name} does not fit the run parameters.')
    elif (scenario_found==True) and (scenarios_valid==True) and (models_results_parameters_present==False):
        raise ValueError(f'Scenario information found for {name} but result files seem to be missing.')
    elif (scenario_found==True) and (scenarios_valid==True) and (models_results_parameters_present==True):
        pq_ind_known = load_scenarios(path,name,dataset)
        ledto = create_ledto(pq_ind_known,dataset.graph_df,dataset.pq_df,dataset.v_df)
        ledto.last_index = last_index
        return ledto
        

def random_scenarios(attempts,n_loads,n_known=-1,seed=8):
    ''' 
    Generates scenarios with the amount of known loads varying from 1 to the number of loads minus one (n_known=-1) or
    with a defined number of known loads (n_known).
    '''
    known_loads_scenarios = []

    if (n_loads==n_known) or (attempts==0) or (n_loads==0):
        raise ValueError("A function parameter is out of bounds.")

    if n_known == -1:
        load_range = range(1,n_loads)
    else:
        load_range = [n_known]
    
    np.random.seed(seed)
    for loads_known in load_range:
        attempt_list = []
        ii = count = 0
        while (ii < attempts) & (count < 500):
            count +=1
            rand_attempt = sorted(np.random.choice(np.arange(n_loads),size=loads_known,replace=False).tolist())
            if rand_attempt not in attempt_list:
                attempt_list.append(rand_attempt)
            ii = len(attempt_list)
            if count==500:
                print("Random scenario generation is at limit, check parameters!")
        known_loads_scenarios.append(attempt_list)
    known_loads_scenarios = [item for sublist in known_loads_scenarios for item in sublist]
    return known_loads_scenarios


def create_random_scenarios(pq_df,v_df,attempts,n_known,graph): 
    n_loads = int(len(pq_df.columns)/2)
    known_loads_scenarios = random_scenarios(attempts,n_loads,n_known)
    pq_ind_known = known_loads_to_indices(known_loads_scenarios)
    ledto = create_ledto(pq_ind_known,graph,pq_df,v_df)
    return ledto


def create_ledto(pq_ind_known,graph,pq_df,v_df,ledto_prefilled=None):
    v_ind_known = []
    load_to_bus = match_loads_buses(graph)
    for scenario in pq_ind_known:
        v_ind = select_loads_buses_scenario(scenario,pq_df,v_df,load_to_bus)
        v_ind_known.append(v_ind)
    ledto = LEDTO(pq_ind_known) if ledto_prefilled is None else ledto_prefilled
    ledto.pq_ind_known = pq_ind_known
    ledto.v_ind_known = v_ind_known
    return ledto


def known_loads_to_indices(known_loads_scenarios):
    ''' 
    Takes a list of lists of scenarios containing the known loads in the scenarios
    and returns a list of scenarios with indices of loads as found in the dataframe columns.
    Args:
        known_loads_scenario (list): e.g. [[1],[2],[1,4],...] 
    Returns:
        pq_ind_known (list): e.g. [[2,3],[4,5],[2,3,8,9],...]
    '''
    pq_ind_known = []
    for items in known_loads_scenarios:
        sublist = []
        for element in items:
            sublist.append(element*2)
            sublist.append(element*2+1)
        pq_ind_known.append(sublist)
    return pq_ind_known


def match_loads_buses(graph_df):
    ''' 
    Match loads and buses with the graph data of the grid.
    Args:
        graph_df (DataFrame): dataframe containing graph data
    Returns:
        load_to_bus (dict): load to bus relation dictionary e.g. {'UL01':'Bus01',...}
    '''
    load_to_bus = dict(list(zip(graph_df['load'],graph_df['bus'])))
    return load_to_bus



def select_loads_buses_scenario(pq_ind_known,pq_df,v_df,load_to_bus):
    '''
    Generates and returns for a single scenario v_ind_known containing the indices of the 'known' 
    voltage columns for load estimation functions. 
    Args:
        pq_ind_known (list): list of load indices e.g. [0,1,8,9]
        load_to_bus (dict): load to bus mapping dictionary e.g. {'UL01':'Bus01',...}
    Returns:
        v_ind_known (list)
    '''
    v_ind_known = []
    for known_load in pq_ind_known:
        if known_load%2==0:
            load_name = pq_df.columns[known_load]
            try:
                bus_name = load_to_bus[load_name[:-2]]
            except: print(f"Load column {load_name} not found in the buses of the graph data.")
            v_ind_known.append(v_df.columns.get_loc(bus_name+'_V'))
    return v_ind_known

# def find_indices(df_columns,known_element):
#     indices = []
#     for idx,colname in enumerate(df_columns):
#         if str(colname).find(known_element) != -1:
#             indices.append(idx)   
#     if len(indices)==0: raise ValueError(f"Error: {known_element} not found in the columns.")
#     return indices




##  Feature selection functions 
#####################################################

# def report_column_names(X_le,y_le,dataset,path):
#     feat_col_list = X_le.columns.values.tolist()
#     feat_col_list = [dataset.load_decode_dict[col] for col in feat_col_list if col[-2:]=='_p']
#     feat_col_list = [dataset.bus_decode_dict[col] for col in feat_col_list if col[:]=='_p']
#     with open(path/"feature_column_names.json",'w') as file:
#         json.dumps(feat_col_list,file)
#     label_col_list = y_le.columns.values.tolist()
#     with open(path/"label_column_names.json",'w') as file:
#         json.dumps(label_col_list,file)

def select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode,known_buses_ind,path=None, eval=False, training_scalerX=None, training_scalery=None, columns_in_order=None):
    '''
    Select data as either features or labels.
    Scales features and labels by data type (loads and flows - MinMax, voltages - Standard)
    Args: 
        pq_ind_known (list): the column indices of the 'known' loads; it is assumed that even numbers represent P-parts, uneven number Q-parts of the loads.
        v_ind_known (list): the column indices of the 'known' buses
        mode (string): defines the splitting of features and labels; either 'LE','DLEF','DLF','BusLF'
        known_buses_ind: indices of buses that are always known (also used for power flows)
    '''
    pq_df = dataset.pq_df
    """for col in pq_ind_known:
        column = pq_df.columns[col]
        if column[-1] == 'P':
            pq_df[column] = pq_df[column] * -1"""
    v_df = dataset.v_df
    pflow_df = dataset.pflow_df
    qflow_df = dataset.qflow_df

    # add the known buses to the voltage indices
    v_ind_known = list(set(v_ind_known + known_buses_ind))

    # raise errors if all columns are known or none of them are known
    if (len(pq_ind_known)==len(pq_df.columns)) and (len(v_ind_known)==len(v_df.columns)):
        raise ValueError()
    elif (len(pq_ind_known)==0) and (len(v_ind_known)==0):
        raise ValueError()

    X = pq_df.values
    y = v_df.values

    scalerY = StandardScaler()
    y_sc = scalerY.fit_transform(y)
    scalerX = MinMaxScaler()
    X_sc = scalerX.fit_transform(X)

    # put all scaled data in a dataframe, with column names
    data_le = pq_df.merge(v_df,left_index=True,right_index=True)
    data_sc = data_le.copy()
    data_le[:] = np.concatenate([X_sc,y_sc],axis=1)

    if (pflow_df is not None) and (len(known_buses_ind) > 0):
        known_buses_ind_just_trafo_flow = [known_buses_ind[0]]
        known_buses_ind = known_buses_ind_just_trafo_flow
        X_le_p = pflow_df.iloc[:,known_buses_ind]
        X_le_q = qflow_df.iloc[:,known_buses_ind]
        X_le_flow = X_le_p.merge(X_le_q,left_index=True,right_index=True)
        scalerXflow = MinMaxScaler()
        X_le_flow_sc = scalerXflow.fit_transform(X_le_flow.values)
        X_le_flow_df_sc = X_le_flow.copy()
        X_le_flow_df_sc[:] = X_le_flow_sc

    # for each mode, select the feature and label columns
    if mode in ['LE','DLEF','DLF']:

        # features are known loads, known voltages and known flows
        if eval:
            if training_scalerX is not None and training_scalery is not None:
                X = pd.DataFrame(index=pq_df.index)
                X_sorted = pd.DataFrame(index=pq_df.index)
                for var in training_scalerX.columns:
                    if var in pq_df.columns:
                        X[var] = training_scalerX[var][0].transform(pq_df[var].values.reshape(-1,1))
                    elif var in v_df.columns:
                        X[var] = training_scalerX[var][0].transform(v_df[var].values.reshape(-1,1))
                    elif var in pflow_df.columns:
                        X[var] = training_scalerX[var][0].transform(pflow_df[var].values.reshape(-1,1))
                    elif var in qflow_df.columns:
                        X[var] = training_scalerX[var][0].transform(qflow_df[var].values.reshape(-1,1))
                for column in columns_in_order[0]:
                    X_sorted[column] = X[column]

                y = pd.DataFrame(index=pq_df.index)
                y_sorted = pd.DataFrame(index=pq_df.index)
                for var in training_scalery:
                    y[var] = training_scalery[var][0].transform(pq_df[var].values.reshape(-1,1))
                for column in columns_in_order[1]:
                    y_sorted[column] = y[column]

                return X_sorted, y_sorted, training_scalery, training_scalerX

            X_le_v = v_df.loc[:, list(v_df.columns[v_ind_known])]
            scaler_X = create_scalers(v_df.loc[:, X_le_v.columns], scaler='standard')
            X_le_pq = pq_df.loc[:,list(pq_df.columns[pq_ind_known])]
            if len(pq_ind_known) > 0:
                scaler_X = scaler_X.merge(create_scalers(pq_df.loc[:, X_le_pq.columns], scaler='minmax'),left_index=True,right_index=True)
            X_le = data_le.loc[:, list(pq_df.columns[pq_ind_known]) + list(v_df.columns[v_ind_known])]
        else:
            X_le = data_le.loc[:,list(pq_df.columns[pq_ind_known])+list(v_df.columns[v_ind_known])]
            scaler_X = create_scalers(data_sc.loc[:,X_le.columns],scaler='minmax')
        if (pflow_df is not None) and (len(known_buses_ind) > 0):
            X_le = X_le.merge(X_le_flow_df_sc,left_index=True,right_index=True)
            if eval:
                scaler_X = scaler_X.merge(create_scalers(
                    X_le_flow, scaler='minmax'), left_index=True,right_index=True)
            else:
                scaler_X = create_scalers(data_sc.merge(X_le_flow,left_index=True,right_index=True).loc[:,X_le.columns],scaler='minmax')

        # select the label columns depending on the mode
        if mode=='LE':
            if len(pq_df.columns)==len(pq_ind_known):
                raise ValueError()
            y_le = data_le.drop(list(pq_df.columns[pq_ind_known])+list(v_df.columns),axis=1)
            scaler_y = create_scalers(data_sc.drop(list(pq_df.columns[pq_ind_known])+list(v_df.columns),axis=1),scaler='minmax')
        elif mode=='DLEF':
            y_le = data_le.drop(list(pq_df.columns[pq_ind_known])+list(v_df.columns[v_ind_known]),axis=1)
            scaler_y_part1 = create_scalers(data_sc.drop(list(pq_df.columns[pq_ind_known])+list(v_df.columns),axis=1),scaler='minmax')
            scaler_y_part2 = create_scalers(data_sc.drop(list(v_df.columns[v_ind_known])+list(pq_df.columns),axis=1),scaler='std')
            scaler_y = scaler_y_part1.merge(scaler_y_part2,left_index=True,right_index=True)
        elif mode=='DLF':
            if len(v_df.columns)==len(v_ind_known):
                raise ValueError()
            elif len(v_df.columns)==len(known_buses_ind):
                raise ValueError()
            y_le = data_le.drop(list(v_df.columns[v_ind_known])+list(pq_df.columns),axis=1)
            scaler_y = create_scalers(data_sc.drop(list(v_df.columns[v_ind_known])+list(pq_df.columns),axis=1),scaler='std')

    elif mode=='BusLF':
        if len(v_df.columns)==len(known_buses_ind):
                raise ValueError()
        elif len(v_df.columns)==len(v_ind_known):
                raise ValueError()
        elif (len(v_ind_known)==0) and (len(known_buses_ind)==0):
                raise ValueError()
        X_le = data_le.loc[:,v_df.columns[v_ind_known]]
        scaler_X = create_scalers(data_sc.loc[:,X_le.columns],scaler='std')
        if (pflow_df is not None) and (len(known_buses_ind) > 0):   
            X_le = X_le.merge(X_le_flow_df_sc,left_index=True,right_index=True)
            scaler_X = create_scalers(data_sc.merge(X_le_flow,left_index=True,right_index=True).loc[:,X_le.columns],scaler='std')                               
        y_le = data_le.drop(list(v_df.columns[v_ind_known])+list(pq_df.columns),axis=1)
        scaler_y = create_scalers(data_sc.drop(list(v_df.columns[v_ind_known])+list(pq_df.columns),axis=1),scaler='std')
    
    if (len(y_le.columns)==0) or (len(X_le.columns)==0):
        raise ValueError('No columns left in either X or y.')

    return X_le,y_le,scaler_y, scaler_X


def select_feature_label_scaled_LF(dataset):
    '''
    Select loads as features and buses as labels. 
    Scales features and labels by data type (load - MinMax, voltage - Standard)
    '''
    pq_df = dataset.pq_df
    v_df = dataset.v_df

    X = pq_df.values
    y = v_df.values

    scalerY = StandardScaler()
    y_sc = scalerY.fit_transform(y)
    scalerX = MinMaxScaler()
    X_sc = scalerX.fit_transform(X)

    # dataframes of scaled data
    y_lf = v_df.copy()
    y_lf[:] = y_sc

    X_lf = pq_df.copy()
    X_lf[:] = X_sc

    scaler_y_df = create_scalers(v_df,'std')

    return X_lf,y_lf,scaler_y_df



def replace_columns_with_prediction(dataset,prediction_LE,y_pr_columns):
    '''
    Replace the columns of 'unknown' loads in the dataset pq_df with the predicted values from the load estimation.

    Args:
        prediction_LE (array): predicted load values
        y_pr_columns (label list): colum labels of the predicted load values
    '''
    pq_df_modified = dataset.pq_df.copy()
    pq_df_modified.loc[:,y_pr_columns] = prediction_LE
    return pq_df_modified


def report_column_names(X_le,y_le,path,scenario_name):
    with open(path / f"feature_columns_{scenario_name}.json",'w') as file:
        json.dump(X_le.columns.values.tolist(),file)
    with open(path / f"label_columns_{scenario_name}.json",'w') as file:
        json.dump(y_le.columns.values.tolist(),file)

def save_scaler(scaler_y, scaler_X,path,scenario_name):
    joblib.dump(scaler_y, path / f"scaler_y_{scenario_name}.pkl")
    joblib.dump(scaler_X, path / f"scaler_X_{scenario_name}.pkl")