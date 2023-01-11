''' 
A module for creating, training and evaluating artificial neural networks based on keras.
'''
import pandas as pd
import os, re

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from keras.models import Sequential, load_model
from keras.layers import Dense, Input, BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam   
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import initializers

from joblib import Parallel, delayed
import json

class NNDTO:
    ''' 
    Data transfer object carries information through neural network training and evaluation process.
    '''
    def __init__(self):
        self.scalers = None        # df of scalers for X and y data (only relevant for rescaling on evaluation)
        self.gridparams = None     # hyperparameter tuning grid dictionary

        self.results_df = None
        self.results_models = None
        self.results_models_paths = None
        self.Xtest = None
        self.ytest = None
        
        self.df_avg = None
        self.best_model_path = None  # single or list
        self.best_model = None      # single or list
        self.model_linreg = None

        self.metric_df = None
        self.metrics_linreg_df = None

        self.fraction_known = None  # for load estimation - fraction of known loads


def save_nndto(nndto,path='',name=''):
    ''' 
    Stores fields fron a NNDTO object at path. Stored are the gridparams, results_df, results_models.
    '''
    path.mkdir(exist_ok=True)
    with open(path / f"parameters {name}.json",'w') as outfile:
        dump_dict = nndto.gridparams.copy()
        json.dump(dump_dict,outfile)
    for idx,model in enumerate(nndto.results_models):
        model.save(path / f"model {name}_{idx}.h5")
    nndto.results_df.to_csv(path / f"results_df {name}.csv")
    # try:
    #     nndto.metric_df.to_csv(path / f"metric_df {name}.csv")
    # except:
    #     print("No metric_df found to store.")
    # try:
    #     nndto.metrics_linreg_df.to_csv(path / f"metrics_linreg_df {name}.csv")
    # except:
    #     print("No metrics_linreg_df found to store.")

def save_nndto_metric_df(nndto,path,name,idx):
    '''
    Stores fields fron a NNDTO object at path. Stored is the metric_df.
    '''
    path.mkdir(exist_ok=True)
    nndto.metric_df.to_csv(path / f"metric_df {name}_{idx}.csv")


def get_resultfile_paths(model_path,name,id=''):
    ''' 
    Find the paths to files containing results from training. id is an optional parameter to identify the file (load estimation).
    '''
    if name=='': raise ValueError('Check name, got empty string.')    
    if id=='':  # case load flow
        metric_df_path = [path for path in os.listdir(model_path) \
            if len(re.findall(r'metric_df\s\w*'+name+r'\.csv',path))>0]
        try:
            metric_linreg_df_path = [path for path in os.listdir(model_path) \
            if len(re.findall(r'metrics_linreg_df\s\w*'+name+r'\.csv',path))>0]
        except:
            metric_linreg_df_path = ''
        results_df_path = [path for path in os.listdir(model_path) \
            if len(re.findall(r'results_df\s\w*'+name+r'\.csv',path))>0]
    else:        # case load estimation
        metric_df_path = [path for path in os.listdir(model_path) \
            if len(re.findall(r'metric_df\s\w*'+name+f'_{id}.csv',path))>0]    
        try:
            metric_linreg_df_path = [path for path in os.listdir(model_path) \
            if len(re.findall(r'metrics_linreg_df\s\w*'+name+f'_{id}.csv',path))>0]
        except:
            metric_linreg_df_path = ''
        results_df_path = [path for path in os.listdir(model_path) \
            if len(re.findall(r'results_df\s\w*'+name+f'_{id}.csv',path))>0]
    return metric_df_path,metric_linreg_df_path,results_df_path


def load_nndto(model_path,name,id,load_models_too=False):
    '''
    Load stored data in model_path to an NNDTO and return it.
    '''
    nndto = NNDTO()
    metric_df_path,metric_linreg_df_path,results_df_path = get_resultfile_paths(model_path,name,id) 
    try:
        nndto.metric_df = pd.read_csv(model_path / metric_df_path[0] ,index_col=[0])
    except:
        pass
    try:
        nndto.metrics_linreg_df = pd.read_csv(model_path / metric_linreg_df_path[0] ,index_col=[0])
    except:
        pass
    nndto.results_df = pd.read_csv(model_path / results_df_path[0] ,index_col=[0])
    nndto.results_df = nndto.results_df.set_index([nndto.results_df.index,'cv'])
    if load_models_too:
        nndto.results_models_paths = load_models_nndto(model_path,name,id)
    return nndto

def get_model_ids(model_path,name,id):
    ''' 
    Get a sorted list of paths to the models in model_path that contain 'name' and 'id'.
    '''
    if id=='': raise ValueError()
    models_path_list = [path for path in os.listdir(model_path) \
        if len(re.findall(r'model\s\w*'+name+f'_{id}'+r'[_]?\d*.h5',path))>0]   

    models_ids = []
    for mod_path in sorted(models_path_list):
        mod_ids = mod_path.split('.')[0].split('_')[-2:]
        model_id = (mod_path,[int(id) for id in mod_ids])
        models_ids.append(model_id)
 
    # filter for those that contain the id as first list element of model_ids[1]
    models_ids = [element for element in models_ids if element[1][0]==id]
    models_ids_sorted = sorted(models_ids,key=lambda x:x[1][1])
    return models_ids_sorted


def get_all_model_ids(model_path,name):
    ''' 
    Get a sorted list of paths to the models in model_path that contain 'name'.
    '''
    models_path_list = [path for path in os.listdir(model_path) \
        if len(re.findall(r'model\s\w*'+name+'_',path))>0] 

    models_ids = []
    for mod_path in sorted(models_path_list):
        mod_ids = mod_path.split('.')[0].split('_')[-2:]
        model_id = (mod_path,[int(id) for id in mod_ids])
        models_ids.append(model_id)
    models_ids_sorted = sorted(models_ids,key=lambda x:x[1][0])
    #print (models_ids_sorted)
    return models_ids_sorted

def load_models_nndto(model_path,name,id):
    models_ids_sorted = get_model_ids(model_path,name,id)
    models_paths = []
    for path in models_ids_sorted:
        model_path =  path[0]
        models_paths.append(model_path)
    return models_paths

def load_specific_models(model_path,paths):
    models = []
    for path in paths:
        model = load_model(model_path / path)
        models.append(model)
    if len(models)==0:
        raise ValueError()
    return models


##  Neural net creation and training functions 
##############################################


def create_layers(hidden_layers,inodes):
    '''
    Create a tuple with layer and nodes information for the grid parameters.
    '''
    layers_nodes = tuple( [ tuple([inodes]*layer) for layer in hidden_layers] )
    return layers_nodes


def create_neural(layers_nodes,Xtest,ytest,activation='relu',opt='adam',loss='mse',metric='mse'):
    '''
    Creates and returns a neural network with hidden layers equal to len(leyers_nodes) and nodes according to the values in layers_nodes.
    Args:
        layers_nodes (tuple): needs to be specified as () to only produce input and output layers or as iterable tuple to produce hidden layers, e.g. (10,)
    '''
    model = Sequential()

    # input layer
    print(f"model.add: Length of Xtest columns: {len(Xtest[0,:])}")
    model.add(Input(shape=len(Xtest[0,:])))
    model.add(Dense(len(Xtest[0,:]),activation=activation))#, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())
    """model.add(BatchNormalization(axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None))"""

    # optional hidden layers
    for nodes in layers_nodes:
        model.add(Dense(nodes,activation=activation))
        """model.add(BatchNormalization(axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None))"""

    # output layer
    if len(ytest.shape)==1:
        model.add(Dense(1))
    else:
        model.add(Dense(len(ytest[0,:])))

    model.compile(optimizer=opt,loss=loss ,metrics=metric)
    return model


def train_neural(Xtrain,ytrain,model,epochs=5,batch=32,patience=8,val_split=0.15,valdata=None):
    earl = EarlyStopping(patience=patience)
    history = model.fit(
        Xtrain,
        ytrain,
        epochs=epochs,
        batch_size=batch,
        callbacks=[earl,],
        validation_split=val_split,
        validation_data=valdata,
        verbose=0
        )
    return history,model


def eval_neural(model,Xtest,ytest,metric=''):
    if metric== 'mae':
        metric = mean_absolute_error
    elif metric == 'mse':
        metric = mean_squared_error
    else: raise("Metric needs to be defined in eval_neural")

    ypredict = model.predict(Xtest)
    print(f'Training MSE: {metric(ytest,ypredict)}')
    return metric(ytest,ypredict)


def grid_point_cv_neural(grid_element,cv_data,Xtest,ytest,cv_count):
    '''
    Do a cross-validated neural net training run of one set of grid parameters.

    Args:
        grid_element (dict): a single dictionary of the ParameterGrid
        cv_data (tuple): a tuple of training, validation and test data
    '''
    from time import time
    print(grid_element)
    Xtrain,ytrain,Xval,yval = cv_data

    layers_nodes = grid_element['layers_nodes']
    act = grid_element['activation']
    loss = grid_element['loss']
    if grid_element['opt']=='adam':
        opt = Adam(learning_rate=grid_element['learning_rate'])#, clipvalue=1)
    elif grid_element['opt']=='sgd':
        opt = SGD(learning_rate=grid_element['learning_rate'])
    elif grid_element['opt']=='rmsprop':
        opt = RMSprop(learning_rate=grid_element['learning_rate']) #clipnorm=1)#, clipvalue=0.05)

    model = create_neural(layers_nodes,Xtest,ytest,activation=act,opt=opt,loss=loss,metric=loss)
    start = time()
    history, model = train_neural(Xtrain,ytrain,model,batch=grid_element['batch_size'], epochs=100,valdata=(Xval,yval))

    """reg = LinearRegression(normalize=True)
    reg.fit(Xtrain, ytrain)"""
    #model.optimizer.weights

    metric_val = eval_neural(model,Xtest,ytest,metric=loss)
    grid_element_return = grid_element.copy()
    grid_element_return['metric']=metric_val
    grid_element_return['time']=time()-start
    grid_element_return['epochs']=len(history.history['loss'])
    grid_element_return['cv']=cv_count+1
    grid_element_return['layers_nodes']=str(grid_element_return['layers_nodes'])
        
    return [grid_element_return, model]


def cross_val_split(X,y,test_size=0.15,cv=3):
    '''Creates tuples of training and validation data with number equal to cv and a test data set.'''
    cv_data = []
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=test_size,random_state=3,shuffle=True)
    
    folder = KFold(n_splits=cv,shuffle=True)
    split_incices = folder.split(Xtrain)
    for train_index,val_index in split_incices:
        Xt, Xv = Xtrain[train_index], Xtrain[val_index]
        yt, yv = ytrain[train_index], ytrain[val_index]
        cv_data.append((Xt,yt,Xv,yv))
    return cv_data,Xtest,ytest


def grid_cv_neural_parallel(grid,X,y,cv=3):
    ''' Create and train neural nets with parameters according to a ParameterGrid grid and evaluate. Cross-validation is performed.'''
    cv_data,Xtest,ytest = cross_val_split(X,y,test_size=0.15,cv=cv)

    grid_elements_df = pd.DataFrame()
    models = []
    grid_num = len(list(grid))
    for id_grid,grid_element in enumerate(list(grid)):
        
        print(f"Starting to train grid element number {id_grid} of {grid_num} in total")
        grid_element_return_list = Parallel(n_jobs=cv, prefer="threads")(
            delayed(grid_point_cv_neural)(grid_element,cv_data[cv_count],Xtest,ytest,cv_count) for cv_count in range(cv))
        # for debugging with cv=1:
        #return_list = grid_point_cv_neural(grid_element,cv_data[0],Xtest,ytest,0) 
        #grid_element_return_list = [return_list]
        print(f"Finished training grid element number {id_grid}")
        
        for grid_element_return in grid_element_return_list:
            grid_element_df = pd.DataFrame(grid_element_return[0],index=[id_grid])
            grid_elements_df = grid_elements_df.append(grid_element_df)
            models.append(grid_element_return[1])
    
    grid_elements_df = grid_elements_df.set_index([grid_elements_df.index,'cv'])
    return grid_elements_df, models, Xtest, ytest


##  Neural net performance evaluation functions
############################################


def get_best_model(df,models_paths,load_models=True):
    ''' 
    Selects the best performing model from result (averaged over all CV attempts). Averages over all CV runs in the result dataframe.
    Args:
        df (dataframe): result df from a nn training run
        models (list): trained models
    '''
    df_avg = df.groupby(level=0).mean()
    df_new = df.loc[(slice(None), 1), :].reset_index(level=1,drop=True)
    df_new.loc[:,['metric','time','epochs']]=df_avg.loc[:,['metric','time','epochs']].values
    if load_models:
        factor = int(len(df)/len(df_avg))    # for scaling back to the df indices before averaging (=CV)
        best_model = []
        for offset in range(factor):
            target_index = df_avg['metric'].idxmin()*factor+offset
            model_path = models_paths[target_index]
            best_model.append(model_path)
    else: best_model = None
    return (df_new,best_model)


def create_scalers(df,scaler):
    '''
    Create a DataFrame with fit scalers for each column of df.
    Args:
        scaler (string): identification of scaler, either 'minmax' or 'std' currently
    '''
    scaler_df = pd.DataFrame(columns=df.columns)
    for col in df.columns.values:
        sc = MinMaxScaler() if scaler=='minmax' else StandardScaler()
        sc.fit(df.loc[:,col].values.reshape(-1,1))
        scaler_df.loc[0,col] = sc
    return scaler_df


def model_eval_metric_R2_scaled(prediction,ytest_data,idx,metric=mean_squared_error):
    ''' Derive metric and R2 for a prediction (from scaled data).'''
    r2 = r2_score(ytest_data[:,idx],prediction[:,idx])
    metric_score = metric(ytest_data[:,idx],prediction[:,idx])
    return (metric_score,r2)

def model_eval_metric_R2_rescaled(prediction,ytest_data,idx,scaler_y,metric=mean_squared_error):
    ''' Derive metric and R2 for a prediction (from rescaled data). Scaler_y is a single column fit scaler'''
    ytest_rs = scaler_y.inverse_transform(ytest_data[:,idx].reshape(-1,1))
    predict_rs = scaler_y.inverse_transform(prediction[:,idx].reshape(-1,1))
    r2 = r2_score(ytest_rs,predict_rs)
    metric_score = metric(ytest_rs,predict_rs)
    return (metric_score,r2)


def get_metrics_df(columns,nndto,model,metric=mean_squared_error):
    ''' Construct a dataframe containing columns (e.g. y-data columns or unknown loads), a defined metric and r2 as columns.'''
    metric_df = pd.DataFrame()
    prediction = model.predict(nndto.Xtest.values)
    ytest_data = nndto.ytest.values
    for idx,column in enumerate(columns):
        scaler_y = nndto.scalers[1].loc[0,column]
        metric_val_sc, r2_sc = model_eval_metric_R2_scaled(prediction,ytest_data,idx,metric)
        metric_val_rs, r2_rs = model_eval_metric_R2_rescaled(prediction,ytest_data,idx,scaler_y,metric)
        metric_df = metric_df.append({'ycolumn':column,'metric':metric_val_rs,'r2':r2_rs,
                                    'metric_sc':metric_val_sc,'r2_sc':r2_sc},ignore_index=True)
    return metric_df
    

def metrics_df_avg(metric_df_list):
    '''
    Takes multiple metric_df and produces a single metric_df with averaged values.
    It includes new columns with variance on the rescaled metric and r2 values.
    '''
    metric_df_cat = pd.concat(metric_df_list)
    metric_df = metric_df_cat.groupby(level=0).mean()
    metric_df_var = metric_df_cat.groupby(level=0).var()
    metric_df['metric_var'] = metric_df_var['metric']
    metric_df['r2_var'] = metric_df_var['r2']
    metric_df['ycolumn'] = metric_df_list[0]['ycolumn']
    return metric_df
