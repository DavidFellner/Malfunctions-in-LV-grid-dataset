'''
A module for data loading and preparation.
'''
from pathlib import Path

import pandas as pd
import numpy as np

from .load import *
from .dataset import Dataset


class DatasetTestGrids(Dataset):
    """
    Creates a dataset for testgrids 1 to 6 with many individual datafiles.
    TODO: currently not functinal, update to the new suffix encoding of columns
    """
    def __init__(self):
        super().__init__()
        
    def create_dataset(self, data_path,directory,name,file_pattern_dict,graph_pattern):
        '''
        Look in data_path / directory for files with name and pattern and load them into dataframes.
        Do it for every pattern in the file_pattern_dict. Create a dataset object to store the dataframes in.
        '''
        for key,file_pattern in file_pattern_dict.items():
            fname = get_file(data_path/directory,name,file_pattern)
            if len(fname)!=1:
                raise KeyError()
            filepath = Path(directory) / fname[0]
            if key=='loads':
                pq_df = self.produce_load_frame([filepath], data_path)
                self.pq_df = pq_df
            else:
                df = self.produce_frame([filepath], data_path)
                if key=='voltages':
                    self.v_df = df
                elif key=='pflow':
                    self.pflow_df = df
                    self.flow_present = True
                elif key=='qflow':
                    self.qflow_df = df

        graph_path_full = graph_pattern[0]+name+graph_pattern[1]
        graph_df = retrieve_loadBus_frame(data_path/directory/graph_path_full)
        self.graph_df = graph_df

    def produce_frame(self, file_paths, data_path):
        '''
        Create and return a singular dataframe from the files in file_paths.
        '''
        df = pd.read_csv(data_path / file_paths[0],index_col=[0],dtype=np.float64)
        for idx,path in enumerate(file_paths[1:]):
            print(idx)
            df = df.append(pd.read_csv(data_path / path,index_col=[0],dtype=np.float64),ignore_index=True)
        return df 

    def produce_load_frame(self, file_paths, data_path):
        '''
        Create and return a singular dataframe from the load files in file_paths with a column name derived from the first two lines in the load file.
        '''
        df = pd.read_csv(data_path / file_paths[0],index_col=[0])
        cols = df.columns
        cols_sub = df.iloc[0,:].values
        columns_new = [val+'_'+cols_sub[idx][0] for idx,val in enumerate(cols)]
        
        df = pd.read_csv(data_path / file_paths[0],skiprows=1,dtype=np.float64,index_col=[0])
        df.index = range(len(df.index))
        df.columns = columns_new
        if len(file_paths)>1:
            for idx,path in enumerate(file_paths[1:]):
                print(idx)
                df_raw = pd.read_csv(data_path / path,skiprows=1,dtype=np.float64,index_col=[0])
                df_raw.columns = columns_new
                df = df.append(df_raw,ignore_index=True)
        return df 



class DatasetGasen(Dataset):
    """
    Dataset generation class for Gasen datasets with single table containing all data and multi-index columns.
    """
    def __init__(self):
        
        super().__init__()

    def create_dataset(self, 
        data_path:Path,
        directory:str,
        name:str,
        graph_name:str):
        '''
        Look in data_path / directory for single data file with name and load it into dataframes.
        Create a dataset object to store the dataframes in.
        '''   
        self.df = pd.read_csv(data_path / directory / name, header=[0,1,2], index_col=[0,1,2])
        df = pd.read_csv(data_path / directory / name)

        ####check for 0, NaN or infinte values
        if True in df.isnull().any():
            print('Zero values in Data!')
        dataset = np.array(df)
        count = 0
        rows = len(dataset)
        columns = len(dataset[1])
        for x in range(0, rows):
            for y in range(0, columns):
                if (dataset[x][y] in {'NaN', 'infinity', 'nan', 'Infinity'}):
                    dataset = np.delete(dataset, x, 0)
                    count = count + 1
        print("Deleted Rows: " + str(count))
        df = df.dropna(axis='rows')
        ######
        
        p_df = df.loc[2:,df.loc[1,:]=='m:P:bus1']
        p_df.columns = [element+'_P' for element in p_df.columns]

        q_df = df.loc[2:,df.loc[1,:]=='m:Q:bus1']
        q_df.columns = [element[:-2]+'_Q' for element in q_df.columns]

        pq_df = p_df.merge(q_df,left_index=True,right_index=True)
        pq_df = pq_df.sort_index(axis='columns').reset_index(drop=True)
        self.pq_df = pq_df.astype('float')

        v_df = df.loc[2:,df.loc[1,:]=='m:u'].reset_index(drop=True)
        v_df.columns = [element+'_V' for element in v_df.columns]
        self.v_df = v_df.astype('float')

        pflow_df = df.loc[2:,df.loc[1,:]=='m:Pflow'].reset_index(drop=True)
        pflow_df.columns = [element[:-2]+'_p' for element in pflow_df.columns]
        self.pflow_df = pflow_df.astype('float')

        qflow_df = df.loc[2:,df.loc[1,:]=='m:Qflow'].reset_index(drop=True)
        qflow_df.columns = [element[:-2]+'_q' for element in qflow_df.columns]
        self.qflow_df = qflow_df.astype('float')

        self.flow_present = True
        self.graph_df = retrieve_loadBus_frame(data_path / directory / graph_name)