'''
A module for data loading and preparation.
'''
import pandas as pd
import numpy as np
import re, os
from pathlib import Path


# class Dataset():
    
#     def __init__(self):
#         self.pq_df = None
#         self.v_df = None
#         self.pflow_df = None
#         self.qflow_df = None
#         self.flow_present = False           #if flow data is loaded

#         self.graph_df = None
#         self.known_buses = None

#         self.load_encode_dict = None
#         self.bus_encode_dict = None
#         self.load_decode_dict = None
#         self.bus_decode_dict = None


# def create_dataset(data_path,directory,name,file_pattern_dict,graph_pattern):
#     '''
#     Look in data_path / directory for files with name and pattern and load them into dataframes.
#     Do it for every pattern in the file_pattern_dict. Create a dataset object to store the dataframes in.
#     Returns:
#         dataset (Dataset)
#     '''
#     dataset = Dataset()

#     for key,file_pattern in file_pattern_dict.items():
#         fname = get_file(data_path/directory,name,file_pattern)
#         if len(fname)!=1:
#             raise KeyError()
#         filepath = Path(directory) / fname[0]
#         if key=='loads':
#             pq_df = produce_load_frame([filepath], data_path)
#             dataset.pq_df = pq_df
#         else:
#             df = produce_frame([filepath], data_path)
#             if key=='voltages':
#                 dataset.v_df = df
#             elif key=='pflow':
#                 dataset.pflow_df = df
#                 dataset.flow_present = True
#             elif key=='qflow':
#                 dataset.qflow_df = df

#     graph_path_full = graph_pattern[0]+name+graph_pattern[1]
#     graph_df = retrieve_loadBus_frame(data_path/directory/graph_path_full)
#     dataset.graph_df = graph_df

#     return dataset


def get_file(path,name,file_pattern):
    if (path=='') or (name=='') or (file_pattern==''):
        raise ValueError()
    pattern = r'\w*'+name+r'_'+file_pattern
    file_list = [file for file in os.listdir(path) if len(re.findall(pattern,file))>0]
    return file_list


def produce_frame(file_paths, data_path):
    '''
    Create and return a singular dataframe from the files in file_paths.
    '''
    df = pd.read_csv(data_path / file_paths[0],index_col=[0],dtype=np.float64)
    for idx,path in enumerate(file_paths[1:]):
        print(idx)
        df = df.append(pd.read_csv(data_path / path,index_col=[0],dtype=np.float64),ignore_index=True)
    return df 


def produce_load_frame(file_paths, data_path):
    '''
    Create and return a singular dataframe from the load files in file_paths with a column name derived from the first two lines in the load file.
    '''
    df = pd.read_csv(data_path / file_paths[0])#,index_col=[0])
    cols = df.columns
    cols_sub = df.iloc[0,:].values
    columns_new = [val+'_'+cols_sub[idx][0] for idx,val in enumerate(cols)]
    
    df = pd.read_csv(data_path / file_paths[0],skiprows=1,dtype=np.float64)#,index_col=[0])
    df.index = range(len(df.index))
    df.columns = columns_new
    if len(file_paths)>1:
        for idx,path in enumerate(file_paths[1:]):
            print(idx)
            df_raw = pd.read_csv(data_path / path,skiprows=1,dtype=np.float64)#,index_col=[0]
            df_raw.columns = columns_new
            df = df.append(df_raw,ignore_index=True)
    return df 


def retrieve_loadBus_frame(data_path):
    '''
    Load a cleaned dataframe with load and bus graph data from file.
    '''
    df = pd.read_csv(data_path, delimiter='\t')
    df = df.loc[1:,['Name','Terminal.1']]
    df.columns = ['load','bus']
    df.loc[:,'load'] = df.loc[:,'load'].str.strip('\t\n ')
    df.loc[:,'bus'] = df.loc[:,'bus'].str.strip('\t\n ')
    return df