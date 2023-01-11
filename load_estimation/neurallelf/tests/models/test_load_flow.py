import pytest, os, sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from neurallelf.models.load_flow import *



###  Module fixtures section  ##############
##################################

@pytest.fixture
def tempdir_with_files(tmpdir,request,scope='module'):
    #sys.stdout.write("\n"+str(type(request.param))+"\n")
    for filename in request.param:
        file = tmpdir.join(filename)
        with open(file,'w'):
            pass
    yield tmpdir


@pytest.fixture
def get_pq_v_df():
    pq_df = pd.DataFrame(data = np.random.random(100).reshape(10,10),
                         columns=['UL01_p','UL01.1_q',
                                  'UL02_p','UL02.1_q',
                                  'UL03_p','UL03.1_q',
                                  'UL04_p','UL04.1_q',
                                  'UL05_p','UL05.1_q'])
    v_df = pd.DataFrame(data = np.random.random(30).reshape(10,3),
                        columns=['Bus01',
                                 'Bus02',
                                 'Bus04'])
    yield pq_df,v_df



###  Tests section  ##############
##################################

@pytest.mark.xfail
def test_save_nndto():
    # this function is low priority for testing
    pass

@pytest.mark.xfail
def test_save_nndto_metric_df():
    pass



FILE_LIST_LF = ['metric_df grid01.csv',
                'metric_df grid10.csv',
                'metrics_linreg_df grid01.csv',
                'metrics_linreg_df grid10.csv',
                'results_df grid01.csv',
                'results_df grid02.csv',
                'results_df grid10.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_LF],indirect=True)
def test_get_resultfile_paths_loadflow(tempdir_with_files):
    # if case load flow
    assert len(os.listdir(tempdir_with_files)) == 7
    assert get_resultfile_paths(tempdir_with_files,'01','') == (['metric_df grid01.csv'],
                                                        ['metrics_linreg_df grid01.csv'],
                                                            ['results_df grid01.csv'])

FILE_LIST_LE = ['metric_df grid01_5.csv',
                'metric_df grid10_0.csv',
                'metrics_linreg_df grid01_5.csv',
                'metrics_linreg_df grid10_0.csv',
                'results_df grid01_5.csv',
                'results_df grid10_0.csv',
                'results_df grid01_1.csv',
                'results_df grid01_15.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_LE],indirect=True)
def test_get_resultfile_paths_loadestimation(tempdir_with_files):
    # else case load estimation
    assert len(os.listdir(tempdir_with_files)) == 8
    assert get_resultfile_paths(tempdir_with_files,'01',5) == (['metric_df grid01_5.csv'],
                                                        ['metrics_linreg_df grid01_5.csv'],
                                                        ['results_df grid01_5.csv'])

def test_get_resultfile_paths_loadestimation_empty():
    with pytest.raises(ValueError):
        get_resultfile_paths(tempdir_with_files,'',5) 



@pytest.mark.xfail
def test_load_nndto():
    # this function is low priority for testing
    pass




MODEL_LIST_LE = [ 'model grid01_0_0.h5',
                  'model grid01_0_2.h5',
                  'model grid01_0_1.h5',
                  'model grid01_0_3.h5',
                  'model grid01_1_0.h5',
                  'model grid01_1_1.h5',
                  'model grid05_0_2.h5',
                  'model grid05_0_0.h5',
                  'model grid05_0_1.h5']

@pytest.mark.parametrize('tempdir_with_files',[MODEL_LIST_LE],indirect=True)
def test_get_model_ids_loadestimation_grid01_id0(tempdir_with_files):
    assert len(os.listdir(tempdir_with_files)) == 9
    assert get_model_ids(tempdir_with_files,'01',0) == [('model grid01_0_0.h5',[0,0]),
                                                           ('model grid01_0_1.h5',[0,1]),
                                                           ('model grid01_0_2.h5',[0,2]),
                                                           ('model grid01_0_3.h5',[0,3])]

@pytest.mark.parametrize('tempdir_with_files',[MODEL_LIST_LE],indirect=True)
def test_get_model_ids_loadestimation_grid01_id1(tempdir_with_files):
    assert len(os.listdir(tempdir_with_files)) == 9
    assert get_model_ids(tempdir_with_files,'01',1) == [('model grid01_1_0.h5',[1,0]),
                                                        ('model grid01_1_1.h5',[1,1])]

MODEL_LIST_LE_2 = [ 'model grid_01_0_0.h5',
                  'model grid_01_0_2.h5',
                  'model grid_01_1_1.h5',
                  'model grid_01_1_3.h5']
@pytest.mark.parametrize('tempdir_with_files',[MODEL_LIST_LE_2],indirect=True)
def test_get_model_ids_loadestimation_grid01_id0advanced(tempdir_with_files):
    assert len(os.listdir(tempdir_with_files)) == 4
    assert get_model_ids(tempdir_with_files,'01',0) == [('model grid_01_0_0.h5',[0,0]),
                                                        ('model grid_01_0_2.h5',[0,2])]




MODEL_LIST_1 = [  'model grid01_0_0.h5',
                  'model grid01_0_2.h5',
                  'model grid01_0_1.h5',
                  'model grid02_0_3.h5']
@pytest.mark.parametrize('tempdir_with_files',[MODEL_LIST_1],indirect=True)
def test_get_all_model_ids_normal(tempdir_with_files):
    assert get_all_model_ids(tempdir_with_files,'01') == [  ('model grid01_0_0.h5',[0,0]),
                                                            ('model grid01_0_1.h5',[0,1]),
                                                            ('model grid01_0_2.h5',[0,2])]

MODEL_LIST_2 = [  'model grid10_2_0.h5',
                  'model grid01_0_2.h5',
                  'model grid01_0_1.h5',
                  'model grid02_0_10.h5']
@pytest.mark.parametrize('tempdir_with_files',[MODEL_LIST_2],indirect=True)
def test_get_all_model_ids_name(tempdir_with_files):
    assert get_all_model_ids(tempdir_with_files,'10') == [  ('model grid10_2_0.h5',[2,0])]



@pytest.mark.xfail
def test_load_models_nndto():
    # this function is low priority for testing
    pass


##  Neural net creation and training functions 
##############################################


def test_create_layers_normal1():
    assert create_layers([1,3],5) == ((5,),(5,5,5))

def test_create_layers_normal2():
    assert create_layers([1,5],100) == ((100,),(100,100,100,100,100))

def test_create_layers_normal3():
    assert create_layers([1],5) == ((5,),)

def test_create_layers_nohidden():
    assert create_layers([],5) == ()



def test_create_neural_y1():
    # test case: input layer with 4 /output with 1 node
    Xtest = np.ones(100).reshape(25,4)
    ytest = np.ones(30).reshape(30,1)
    layers_nodes = ()
    model = create_neural(layers_nodes,Xtest,ytest)
    assert model.layers[0].output_shape[-1] == 4
    assert model.layers[1].output_shape[-1] == 1

def test_create_neural_y4():
    # test case: input layer with 5 /output with 4 nodes
    Xtest = np.ones(100).reshape(20,5)
    ytest = np.ones(40).reshape(10,4)
    layers_nodes = ()
    model = create_neural(layers_nodes,Xtest,ytest)
    assert model.layers[0].output_shape[-1] == 5
    assert model.layers[1].output_shape[-1] == 4

def test_create_neural_1hidden():
    # test case: a single hidden layer
    Xtest = np.ones(100).reshape(20,5)
    ytest = np.ones(40).reshape(10,4)
    layers_nodes = (10,)
    model = create_neural(layers_nodes,Xtest,ytest)
    assert model.layers[0].output_shape[-1] == 5
    assert model.layers[1].output_shape[-1] == 10
    assert model.layers[2].output_shape[-1] == 4



@pytest.mark.xfail
def test_train_neural():
    # this function is low priority for testing
    pass

@pytest.mark.xfail
def test_eval_neural():
    # this function is low priority for testing
    pass

@pytest.mark.xfail
def test_grid_point_cv_neural():
    # this function is low priority for testing
    pass

@pytest.mark.xfail
def test_cross_val_split():
    # TODO, but how?
    pass

@pytest.mark.xfail
def test_grid_cv_neural_parallel():
    # this function is low priority for testing
    pass



def test_get_best_model_dfnew_columns():
    data = [
        [0,1,'relu',32,'(15,)',0.001,'mse','adam',0.033,20,45],
        [0,2,'relu',32,'(15,)',0.001,'mse','adam',0.034,21,48],
        [0,3,'relu',32,'(15,)',0.001,'mse','adam',0.035,24,42],
        [1,1,'relu',16,'(15,12)',0.001,'mse','adam',0.020,30,55],
        [1,2,'relu',16,'(15,12)',0.001,'mse','adam',0.022,34,65],
        [1,3,'relu',16,'(15,12)',0.001,'mse','adam',0.021,32,75]
    ]
    df = pd.DataFrame(data=data,
        columns=['',
            'cv',
            'activation',
            'batch_size',
            'layers_nodes',
            'learning_rate',
            'loss',
            'opt',
            'metric',
            'time',
            'epochs'])
    df = df.set_index(['','cv'])
    models = [1,2,3,4,5,6]
    df_new, best_model = get_best_model(df,models,load_models=True)
    target_cols = ['activation', 'batch_size', 'layers_nodes',
                    'learning_rate', 'loss', 'opt',
                    'metric', 'time', 'epochs']
    for idx,col in enumerate(df_new.columns.values):
        assert col == target_cols[idx]

def test_get_best_model_dfnew_values():
    data = [
        [0,1,'relu',32,'(15,)',0.001,'mse','adam',0.033,20,45],
        [0,2,'relu',32,'(15,)',0.001,'mse','adam',0.034,22,50],
        [0,3,'relu',32,'(15,)',0.001,'mse','adam',0.035,24,55],
        [1,1,'relu',16,'(15,12)',0.001,'mse','adam',0.020,30,55],
        [1,2,'relu',16,'(15,12)',0.001,'mse','adam',0.022,34,65],
        [1,3,'relu',16,'(15,12)',0.001,'mse','adam',0.021,32,75]
    ]
    df = pd.DataFrame(data=data,
        columns=['',
            'cv',
            'activation',
            'batch_size',
            'layers_nodes',
            'learning_rate',
            'loss',
            'opt',
            'metric',
            'time',
            'epochs'])
    df = df.set_index(['','cv'])
    models = [1,2,3,4,5,6]
    df_new, best_model = get_best_model(df,models,load_models=True)
    target_vals = [
        ['relu',32,'(15,)',0.001,'mse','adam',0.034,22,50],
        ['relu',16,'(15,12)',0.001,'mse','adam',0.021,32,65]
    ]
    for idx,vals in enumerate(df_new.values):
        for idxb, val in enumerate(vals):
            assert val == target_vals[idx][idxb]

def test_get_best_model_best_model():
    data = [
        [0,1,'relu',32,'(15,)',0.001,'mse','adam',0.033,20,45],
        [0,2,'relu',32,'(15,)',0.001,'mse','adam',0.034,22,50],
        [0,3,'relu',32,'(15,)',0.001,'mse','adam',0.035,24,55],
        [1,1,'relu',16,'(15,12)',0.001,'mse','adam',0.020,30,55],
        [1,2,'relu',16,'(15,12)',0.001,'mse','adam',0.022,34,65],
        [1,3,'relu',16,'(15,12)',0.001,'mse','adam',0.021,32,75]
    ]
    df = pd.DataFrame(data=data,
        columns=['',
            'cv',
            'activation',
            'batch_size',
            'layers_nodes',
            'learning_rate',
            'loss',
            'opt',
            'metric',
            'time',
            'epochs'])
    df = df.set_index(['','cv'])
    models = [0,1,2,3,4,5]
    df_new, best_model = get_best_model(df,models,load_models=True)
    assert best_model == [3,4,5]





def test_create_scalers_length(get_pq_v_df):
    pq_df,v_df = get_pq_v_df
    assert len(create_scalers(pq_df,'std')) == 1

def test_create_scalers_columns(get_pq_v_df):
    pq_df,v_df = get_pq_v_df
    assert list(create_scalers(pq_df,'std').columns.values) == ['UL01_p','UL01.1_q',
                                                                'UL02_p','UL02.1_q',
                                                                'UL03_p','UL03.1_q',
                                                                'UL04_p','UL04.1_q',
                                                                'UL05_p','UL05.1_q']

def test_create_scalers_scaler(get_pq_v_df):
    pq_df,v_df = get_pq_v_df
    assert isinstance(create_scalers(v_df,'minmax').loc[0,'Bus01'], MinMaxScaler)

def test_create_scalers_scaler2(get_pq_v_df):
    pq_df,v_df = get_pq_v_df
    assert isinstance(create_scalers(v_df,'std').loc[0,'Bus01'], StandardScaler)




@pytest.fixture
def get_prediciton_ytestdata():
    prediction = np.array([[0,0.25,0.35],
                           [0.45,0.55,0.65],
                           [0.75,0.85,1]])
    y_test_data = np.array([[0,0.2,0.3],
                           [0.4,0.5,0.6],
                           [0.7,0.8,1]])  
    yield prediction, y_test_data


def test_model_eval_metric_R2_scaled_idx0(get_prediciton_ytestdata):
    prediction, y_test_data = get_prediciton_ytestdata
    metric_sc, r2_sc = model_eval_metric_R2_scaled(prediction,y_test_data,0,mean_squared_error)
    assert metric_sc == pytest.approx(0.0016666666666666676)
    assert r2_sc == pytest.approx(0.9797297297297297)

def test_model_eval_metric_R2_scaled_idx1(get_prediciton_ytestdata):
    prediction, y_test_data = get_prediciton_ytestdata
    metric_sc, r2_sc = model_eval_metric_R2_scaled(prediction,y_test_data,1,mean_squared_error)
    assert metric_sc == pytest.approx(0.0024999999999999988)
    assert r2_sc == pytest.approx(0.9583333333333334)



@pytest.mark.xfail
def test_model_eval_metric_R2_rescaled_idx0():
    # this function is hard to test
    pass


@pytest.mark.xfail
def test_get_metrics_df():
    # this function is low priority for testing
    pass


def test_metrics_df_avg_normal():
    df1 = pd.DataFrame({'ycolumn':['UL01','UL02'],'metric':[0.1,0.2],'r2':[0.8,0.9]})
    df2 = pd.DataFrame({'ycolumn':['UL01','UL02'],'metric':[0.4,0.5],'r2':[0.6,0.7]})
    metric_df = metrics_df_avg([df1,df2])
    assert list(metric_df.columns) == ['metric','r2','metric_var','r2_var','ycolumn']
    assert metric_df.loc[metric_df.ycolumn=='UL01','metric'].values == 0.25
    assert metric_df.loc[metric_df.ycolumn=='UL02','metric'].values == 0.35
    assert metric_df.loc[metric_df.ycolumn=='UL01','r2'].values == 0.7
    assert metric_df.loc[metric_df.ycolumn=='UL02','r2'].values == 0.8

    assert metric_df.loc[metric_df.ycolumn=='UL01','metric_var'].values[0] == pytest.approx(0.045)
    assert metric_df.loc[metric_df.ycolumn=='UL01','r2_var'].values[0] == pytest.approx(0.02)