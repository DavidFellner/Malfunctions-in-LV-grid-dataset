import pytest

from neurallelf.models.load_estimation import *
from neurallelf.data.dataset_generators import DatasetTestGrids as Dataset


###  Module fixtures section  ##############
############################################

@pytest.fixture
def tempdir_scenario_1(tmpdir):
    file = tmpdir.join('scenarios grid01.json')
    with open(file,'w') as outfile:
        write_str = json.dumps([['load02_P','load02_Q'],['load01_P','load01_Q','load33_Q','load33_P']])
        json.dump(write_str,outfile)
    yield tmpdir

@pytest.fixture
def tempdir_scenario_single(tmpdir):
    file = tmpdir.join('scenarios grid01.json')
    with open(file,'w') as outfile:
        write_str = json.dumps([['load01_P','load01_Q','load33_Q','load33_P']])
        json.dump(write_str,outfile)
    yield tmpdir


@pytest.fixture
def tempdir_scenario_and_files(tmpdir,request,scope='module'):
    file = tmpdir.join('scenarios grid01.json')
    with open(file,'w') as outfile:
        write_str = json.dumps(request.param['scenarios'])
        json.dump(write_str,outfile)
    for filename in request.param['filenames']:
        file = tmpdir.join(filename)
        with open(file,'w'):
            pass
    yield tmpdir


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
                         columns=['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q'])
    v_df = pd.DataFrame(data = np.random.random(30).reshape(10,3),
                        columns=['Bus01_V',
                                 'Bus02_V',
                                 'Bus04_V'])
    yield pq_df,v_df

@pytest.fixture
def get_flow_df():
    pflow_df = pd.DataFrame(data = np.random.random(30).reshape(10,3),
                        columns=['Bus01_p',
                                 'Bus02_p',
                                 'Bus04_p'])
    qflow_df = pd.DataFrame(data = np.random.random(30).reshape(10,3),
                    columns=['Bus01_q',
                            'Bus02_q',
                            'Bus04_q'])
    yield pflow_df,qflow_df


###  Tests for ledto functions 
##################################

@pytest.mark.xfail
def test_save_ledto():
    # this function is low priority for testing
    pass


@pytest.mark.xfail
def test_save_ledto_metrics_summary():
    # this function is low priority for testing
    pass



def test_load_scenarios_grid01_multi(tempdir_scenario_1):
    # scenarios contains: [['load02_P','load02_Q'],['load01_P','load01_Q','load33_Q','load33_P']]
    dataset = Dataset()
    dataset.pq_df = pd.DataFrame(columns=[  'load01_P','load01_Q',
                                            'load02_P','load02_Q',
                                            'load33_P','load33_Q'])
    known_loads_scen = load_scenarios(tempdir_scenario_1,'01',dataset)
    assert known_loads_scen == [[2,3],[0,1,4,5]]

def test_load_scenarios_grid01_single(tempdir_scenario_single):
    # scenarios contains: [['load01_P','load01_Q','load33_Q','load33_P']]
    dataset = Dataset()
    dataset.pq_df = pd.DataFrame(columns=[  'load01_P','load01_Q',
                                            'load02_P','load02_Q',
                                            'load33_P','load33_Q'])
    known_loads_scen = load_scenarios(tempdir_scenario_single,'01',dataset)
    assert known_loads_scen == [[0,1,4,5]]



def test_scenarios_encoding_encode():
    dataset = Dataset()
    dataset.pq_df = pd.DataFrame(columns=[  'load03_P','load03_Q',
                                            'load22_P','load22_Q',
                                            'load11_P','load11_Q'])
    scenarios = [['load22_P','load22_Q'],
                ['load03_P','load03_Q','load11_P','load11_Q']]
    encode = True
    assert scenarios_encoding(dataset,scenarios,encode) == [[2,3],[0,1,4,5]]

def test_scenarios_encoding_decode():
    dataset = Dataset()
    dataset.pq_df = pd.DataFrame(columns=[  'load11_P','load11_Q',
                                            'load22_P','load22_Q',
                                            'load03_P','load03_Q'])
    scenarios = [[2,3],[0,1,4,5]]
    encode = False
    assert scenarios_encoding(dataset,scenarios,encode) == [['load22_P','load22_Q'],
                                                   ['load11_P','load11_Q','load03_P','load03_Q']]



@pytest.mark.xfail
def test_load_ledto():
    # this function is low priority for testing
    pass



def test_search_scenario_file_normal(tempdir_scenario_1):
    run_path = tempdir_scenario_1
    name = 'grid01'
    assert search_scenario_file(run_path,name)[0]

def test_search_scenario_file_short(tempdir_scenario_1):
    run_path = tempdir_scenario_1
    name = '01'
    assert search_scenario_file(run_path,name)[0]

def test_search_scenario_file_invalidname1(tempdir_scenario_1):
    run_path = tempdir_scenario_1
    name = '1'
    assert search_scenario_file(run_path,name)[0]

def test_search_scenario_file_invalidname2(tempdir_scenario_1):
    run_path = tempdir_scenario_1
    name = 'gridgrid01'
    assert search_scenario_file(run_path,name)[0] == False

def test_search_scenario_file_empty(tempdir_scenario_1):
    run_path = tempdir_scenario_1
    name = ''
    assert search_scenario_file(run_path,name)[0] == False



def test_scenario_validation_4loads_range():
    # expected: 4*1 + 4*2 elements
    attempts = 4
    n_known = -1                 # range
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q','UL02_P','UL02_Q'])
    pq_ind_known = [[0,1],[2,3],[4,5],[6,7]]
    assert scenario_validation(pq_ind_known,attempts,n_known,pq_df)

def test_scenario_validation_4loads_range_faulty():
    attempts = 4
    n_known = -1                 # range
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q','UL02_P','UL02_Q'])
    pq_ind_known = [[0,1],[4,5],[6,7],
                                [0,1,2,3],[2,3,6,7],[4,5,8,9],[0,1,4,5]]
    assert scenario_validation(pq_ind_known,attempts,n_known,pq_df) == False

def test_scenario_validation_4loads_nknown1():
    attempts = 3
    n_known = 1
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q'])
    pq_ind_known = [[2,3],[4,5],[0,1]]
    assert scenario_validation(pq_ind_known,attempts,n_known,pq_df)

def test_scenario_validation_4loads_nknown1_faulty():
    attempts = 4
    n_known = 1
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q','UL02_P','UL02_Q'])
    pq_ind_known = [[1,2,3],[1,3,4],[1,2,4]]
    assert scenario_validation(pq_ind_known,attempts,n_known,pq_df) == False

def test_scenario_validation_4loads_nknown0(get_pq_v_df):
    attempts = 4
    n_known = 0                 
    pq_df, _ = get_pq_v_df
    pq_df = pq_df.drop(['UL05_P','UL05_Q'],axis=1)
    pq_ind_known = [[1,2,3,4]]
    assert scenario_validation(pq_ind_known,attempts,n_known,pq_df) == False



FILE_LIST_NORMAL = ['model grid02_33_7.h5','parameters grid02_33.json','results_df grid02_33.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_NORMAL],indirect=True)
def test_confirm_models_results_params_normal(tempdir_with_files):
    run_path = tempdir_with_files
    assert confirm_models_results_params(run_path,'02')

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_NORMAL],indirect=True)
def test_confirm_models_results_params_no_file(tempdir_with_files):
    run_path = tempdir_with_files
    assert confirm_models_results_params(run_path,'01') == False

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_NORMAL],indirect=True)
def test_confirm_models_results_params_error(tempdir_with_files):
    run_path = tempdir_with_files
    with pytest.raises(ValueError):
        assert confirm_models_results_params(run_path,'')

FILE_LIST_MISSING = ['model grid02_33_7.h5','parameters grid02_33.json']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_MISSING],indirect=True)
def test_confirm_models_results_params_normal(tempdir_with_files):
    run_path = tempdir_with_files
    assert confirm_models_results_params(run_path,'02') == False



##  Tests for scenario creation functions 
#####################################################

SCENARIOS_1 = [['UL01_P','UL01_Q'],['UL03_P','UL03_Q'],['UL01_P','UL01_Q','UL02_P','UL02_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q'],['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q','UL05_P','UL05_Q']]
FILE_LIST_SCEN_1 = ['model grid01_33_7.h5','parameters grid01_33.json','results_df grid01_33.csv']
SCENARIOS_2 = [['UL05_P','UL05_Q'],['UL03_P','UL03_Q'],['UL05_P','UL05_Q','UL02_P','UL02_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q'],
                ['UL04_P','UL04_Q','UL02_P','UL02_Q','UL03_P','UL03_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL03_P','UL03_Q','UL02_P','UL02_Q','UL04_P','UL04_Q','UL05_P','UL05_Q']]
FILE_LIST_SCEN_2 = ['model grid01_33_7.h5','parameters grid01_01.json','results_df grid01_99.csv']
@pytest.mark.parametrize('tempdir_scenario_and_files',
    [{'scenarios':SCENARIOS_1,
    'filenames':FILE_LIST_SCEN_1},
    {'scenarios':SCENARIOS_2,
    'filenames':FILE_LIST_SCEN_2}],indirect=True)
def test_look_for_valid_scenarios_allTrue(tempdir_scenario_and_files,get_pq_v_df):
    run_path = tempdir_scenario_and_files
    name = '01'
    attempts = 2
    n_known = -1
    dataset = Dataset()
    dataset.pq_df, _ = get_pq_v_df
    scenario_found, scenarios_valid, models_results_parameters_present = look_for_valid_scenarios(run_path,name,attempts,n_known,dataset)
    assert scenario_found
    assert scenarios_valid
    assert models_results_parameters_present

FILE_LIST_SCEN_3 = ['model grid01_33_7.h5','parameters grid01_33.json','results_df grid01_33.csv']
@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_SCEN_3],indirect=True)
def test_look_for_valid_scenarios_noScen(tempdir_with_files,get_pq_v_df):
    run_path = tempdir_with_files
    name = '01'
    attempts = 2
    n_known = -1
    dataset = Dataset()
    dataset.pq_df, _ = get_pq_v_df
    scenario_found, scenarios_valid, models_results_parameters_present = look_for_valid_scenarios(run_path,name,attempts,n_known,dataset)
    assert scenario_found == False
    assert scenarios_valid == False
    assert models_results_parameters_present == False



SCENARIOS_3 = [['UL03_P','UL03_Q'],['UL01_P','UL01_Q','UL02_P','UL02_Q'],['UL02_P','UL02_Q','UL03_P','UL03_Q'],
                ['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q','UL05_P','UL05_Q']]
SCENARIOS_4 = [['UL05_P','UL05_Q'],['UL03_P','UL03_Q'],['UL01_P','UL01_Q'],['UL05_P','UL05_Q','UL02_P','UL02_Q'],['UL02_P','UL02_Q','UL03_P','UL03_Q'],
                ['UL04_P','UL04_Q','UL02_P','UL02_Q','UL03_P','UL03_Q'],['UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL01_P','UL01_Q','UL02_P','UL02_Q','UL03_P','UL03_Q','UL04_P','UL04_Q'],
                ['UL03_P','UL03_Q','UL02_P','UL02_Q','UL04_P','UL04_Q','UL05_P','UL05_Q']]
@pytest.mark.parametrize('tempdir_scenario_and_files',
    [{'scenarios':SCENARIOS_3,
    'filenames':FILE_LIST_SCEN_1},
    {'scenarios':SCENARIOS_4,
    'filenames':FILE_LIST_SCEN_2}],indirect=True)
def test_look_for_valid_scenarios_scenWrong(tempdir_scenario_and_files,get_pq_v_df):
    run_path = tempdir_scenario_and_files
    name = '01'
    attempts = 2
    n_known = -1
    dataset = Dataset()
    dataset.pq_df, _ = get_pq_v_df
    scenario_found, scenarios_valid, models_results_parameters_present = look_for_valid_scenarios(run_path,name,attempts,n_known,dataset)
    assert scenario_found 
    assert scenarios_valid == False
    assert models_results_parameters_present == False

FILE_LIST_SCEN_4 = ['model grid02_33_7.h5','parameters grid01_33.json','results_df grid01_33.csv']
FILE_LIST_SCEN_5 = ['parameters grid01_01.json','results_df grid01_99.csv']
@pytest.mark.parametrize('tempdir_scenario_and_files',
    [{'scenarios':SCENARIOS_1,
    'filenames':FILE_LIST_SCEN_4},
    {'scenarios':SCENARIOS_2,
    'filenames':FILE_LIST_SCEN_5}],indirect=True)
def test_look_for_valid_scenarios_wrongFiles(tempdir_scenario_and_files,get_pq_v_df):
    run_path = tempdir_scenario_and_files
    name = '01'
    attempts = 2
    n_known = -1
    dataset = Dataset()
    dataset.pq_df, _ = get_pq_v_df
    scenario_found, scenarios_valid, models_results_parameters_present = look_for_valid_scenarios(run_path,name,attempts,n_known,dataset)
    assert scenario_found
    assert scenarios_valid
    assert models_results_parameters_present == False



@pytest.mark.xfail
def test_load_or_create_scenarios():
    pass



@pytest.mark.parametrize('tempdir_scenario_and_files',
    [{'scenarios':SCENARIOS_1,
    'filenames':FILE_LIST_SCEN_1},
    {'scenarios':SCENARIOS_2,
    'filenames':FILE_LIST_SCEN_2}],indirect=True)
def test_load_scenarios_LFfromLE_normal(tempdir_scenario_and_files,get_pq_v_df):
    results_path = tempdir_scenario_and_files
    le_model_dir = ''
    name = '01'
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.graph_df = pd.DataFrame(data=[['UL01','Bus01'],
                                         ['UL02','Bus02'],
                                         ['UL03','Bus02'],
                                         ['UL04','Bus02'],
                                         ['UL05','Bus04']],columns=['load','bus'])
    attempts = 2
    n_known = -1
    ledto = load_scenarios_LFfromLE(results_path,results_path,le_model_dir,name,dataset,attempts,n_known)
    assert len(ledto.pq_ind_known) == 8
    assert ledto.last_index == 33

FILE_LIST_SCEN_3 = ['model grid01_33_7.h5','parameters grid01_33.json','results_df grid01_33.csv']
@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST_SCEN_3],indirect=True)
def test_load_scenarios_LFfromLE_noScen(tempdir_with_files,get_pq_v_df):
    results_path = tempdir_with_files
    le_model_dir = ''
    name = '01'
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.graph_df = pd.DataFrame(data=[['UL01','Bus01'],
                                         ['UL02','Bus02'],
                                         ['UL03','Bus02'],
                                         ['UL04','Bus02'],
                                         ['UL05','Bus04']],columns=['load','bus'])
    attempts = 2
    n_known = -1
    with pytest.raises(ValueError):
        ledto = load_scenarios_LFfromLE(results_path,results_path,le_model_dir,name,dataset,attempts,n_known)


@pytest.mark.parametrize('tempdir_scenario_and_files',
    [{'scenarios':SCENARIOS_1,
    'filenames':FILE_LIST_SCEN_1}],indirect=True)
def test_load_scenarios_LFfromLE_continuation(tempdir_scenario_and_files,get_pq_v_df):
    run_path = tempdir_scenario_and_files
    le_model_dir = ''
    name = '01'
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.graph_df = pd.DataFrame(data=[['UL01','Bus01'],
                                         ['UL02','Bus02'],
                                         ['UL03','Bus02'],
                                         ['UL04','Bus02'],
                                         ['UL05','Bus04']],columns=['load','bus'])
    attempts = 2
    n_known = -1
    ledto = load_scenarios_LFfromLE(run_path,run_path,le_model_dir,name,dataset,attempts,n_known)
    assert ledto.last_index == 33




def test_random_scenarios_if():
    # from 1-4 known loads with 3 attempts
    assert len(random_scenarios(3,5)) == 12

def test_random_scenarios_else():
    # 4 known random loads with 3 attempts
    assert len(random_scenarios(3,5,4)) == 3
    assert len(random_scenarios(3,5,4)[0]) == 4

def test_random_scenarios_elseValues():
    # 3 known random loads with 4 attempts
    assert random_scenarios(4,5,3) == [[1,2,4],[1,3,4],[2,3,4],[0,2,4]]

def test_random_scenarios_known_larger():
    # n_knwon > n_loads
    with pytest.raises(ValueError):
        random_scenarios(3,5,6)
    
def test_random_scenarios_equal():
    # n_knwon == n_loads
    with pytest.raises(ValueError):
        random_scenarios(3,5,5)

def test_random_scenarios_attempts0():
    # attempts == 0
    with pytest.raises(ValueError):
        random_scenarios(0,5)

def test_random_scenarios_loads0():
    # n_loads == 0
    with pytest.raises(ValueError):
        random_scenarios(3,0,1)

def test_random_scenarios_loadsunknown():
    # n_known == 0
    assert len(random_scenarios(3,5,0)) == 1



@pytest.mark.xfail
def test_create_random_scenarios():
    # this function is low priority for testing
    pass



def test_create_ledto_normal():
    pq_ind_known = [[0,1],[2,3],[0,1,4,5]]
    graph = pd.DataFrame(data=[['UL01','Bus02'],
                               ['UL02','Bus03'],
                               ['UL04','Bus11']],columns=['load','bus'])
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q','UL02_P','UL02_Q','UL04_P','UL04_Q'])
    v_df  = pd.DataFrame(columns=['Bus02_V','Bus03_V','Bus11_V'])       
    ledto = create_ledto(pq_ind_known,graph,pq_df,v_df)
    assert ledto.pq_ind_known == [[0,1],[2,3],[0,1,4,5]]
    assert ledto.v_ind_known == [[0],[1],[0,2]]




def test_known_loads_to_indices():
    known_loads_scenarios = [[0],[2],[5,7]]
    assert known_loads_to_indices(known_loads_scenarios) == [[0,1],[4,5],[10,11,14,15]]




def test_match_loads_buses_normal1():
    df =  pd.DataFrame(data=[['UL01','Bus02'],
                             ['UL02','Bus03'],
                             ['UL05','Bus11']],columns=['load','bus'])
    assert len(df) == 3
    assert df.iloc[0,:].values[0] == 'UL01'
    assert df.iloc[0,:].values[1] == 'Bus02'
    assert match_loads_buses(df)['UL01'] == 'Bus02'

def test_match_loads_buses_normal2():
    df =  pd.DataFrame(data=[['UL01','Bus02'],
                             ['UL02','Bus03'],
                             ['UL55','Bus11']],columns=['load','bus'])
    assert match_loads_buses(df)['UL55'] == 'Bus11'




def test_select_uloads_ubuses_scenario01():
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q'])
    v_df = pd.DataFrame(columns=['Bus01_V',
                                 'Bus02_V',
                                 'Bus04_V'])
    load_to_bus = {'UL01':'Bus01',
                   'UL02':'Bus04'}
    v_ind = select_loads_buses_scenario([0,1],pq_df,v_df,load_to_bus)
    assert v_ind == [0]

def test_select_uloads_ubuses_scenario0105():
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q'])
    v_df = pd.DataFrame(columns=['Bus01_V',
                                 'Bus02_V',
                                 'Bus04_V'])
    load_to_bus = {'UL01':'Bus01',
                   'UL02':'Bus04',
                   'UL05':'Bus02'}
    v_ind = select_loads_buses_scenario([0,1,8,9],pq_df,v_df,load_to_bus)
    assert v_ind == [0,1]
    
def test_select_uloads_ubuses_scenario_empty():
    pq_df = pd.DataFrame(columns=['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q'])
    v_df = pd.DataFrame(columns=['Bus01_V',
                                 'Bus02_V',
                                 'Bus04_V'])
    load_to_bus = {'UL01':'Bus01',
                   'UL02':'Bus04',
                   'UL05':'Bus02'}        
    v_ind = select_loads_buses_scenario([],pq_df,v_df,load_to_bus)
    assert v_ind == []





##  Tests for feature selection functions 
#####################################################


# test schema for select_feature_label_scaled_scenario:
    # test cases without pflow/qflow data present:
    #     mode LE
    #         X columns right
    #         y columns right
    #         scaler columns right
    #     mode DLEF
    #         X columns right
    #         y columns right
    #         scaler columns right
    #     mode DLF
    #         X columns right
    #         y columns right
    #         scaler columns right
    #     mode BusLF
    #         X columns right
    #         y columns right
    #         scaler columns right
    # test cases with pflow/qflow data present:
    #     mode LE
    #         X columns right
    #         y columns right
    #         scaler columns right
    #     mode DLEF
    #         X columns right
    #         y columns right
    #         scaler columns right
    #     mode DLF
    #         X columns right
    #         y columns right
    #         scaler columns right
    #     mode BusLF
    #         X columns right
    #         y columns right
    #         scaler columns right
    #
    #
    # test exteme inputs
    #   all buses known
    #   no load known
    #   all loads known
    #   no voltage known
    #   all voltages known
    #   all known / all unknown


# test cases without flow data present:

def test_select_feature_label_scaled_modeLE_noflow_noknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeLE_noflow_oneknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [6,7]
    v_ind_known = [1]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL04_P','UL04_Q','Bus02_V','Bus04_V']
    assert list(y_le.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLEF_noflow_noknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q',
                                        'Bus02_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q',
                                        'Bus02_V','Bus04_V']

def test_select_feature_label_scaled_modeDLEF_noflow_twoknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [2,3,8,9]
    v_ind_known = [1]
    known_buses_ind = [0,2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL02_P','UL02_Q','UL05_P','UL05_Q','Bus01_V','Bus02_V','Bus04_V']
    assert list(y_le.columns.values) == ['UL01_P','UL01_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q']
    assert list(scaler_y.columns.values) == ['UL01_P','UL01_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q']

def test_select_feature_label_scaled_modeDLF_noflow_noknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V']
    assert list(y_le.columns.values) == ['Bus02_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus02_V','Bus04_V']

def test_select_feature_label_scaled_modeDLF_noflow_oneknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [2,3]
    v_ind_known = [0]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL02_P','UL02_Q','Bus01_V','Bus04_V']
    assert list(y_le.columns.values) == ['Bus02_V']
    assert list(scaler_y.columns.values) == ['Bus02_V']

def test_select_feature_label_scaled_modeBusLF_noflow_noknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V']
    assert list(y_le.columns.values) == ['Bus02_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus02_V','Bus04_V']

def test_select_feature_label_scaled_modeBusLF_noflow_oneknown(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    pq_ind_known = [0,1]
    v_ind_known = [1]
    known_buses_ind = [0]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus02_V']
    assert list(y_le.columns.values) == ['Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus04_V']


# test cases with pflow/qflow data present:


def test_select_feature_label_scaled_modeLE_flow_noknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeLE_flow_oneknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [6,7]
    v_ind_known = [1]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL04_P','UL04_Q','Bus02_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLEF_flow_noknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q',
                                        'Bus02_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q',
                                        'Bus02_V','Bus04_V']

def test_select_feature_label_scaled_modeDLEF_flow_twoknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [2,3,8,9]
    v_ind_known = [1]
    known_buses_ind = [0,2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL02_P','UL02_Q','UL05_P','UL05_Q','Bus01_V','Bus02_V','Bus04_V',
                                        'Bus01_p','Bus04_p','Bus01_q','Bus04_q']
    assert list(y_le.columns.values) == ['UL01_P','UL01_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q']
    assert list(scaler_y.columns.values) == ['UL01_P','UL01_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q']

def test_select_feature_label_scaled_modeDLF_flow_noknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V']
    assert list(y_le.columns.values) == ['Bus02_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus02_V','Bus04_V']

def test_select_feature_label_scaled_modeDLF_flow_oneknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [2,3]
    v_ind_known = [0]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL02_P','UL02_Q','Bus01_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['Bus02_V']
    assert list(scaler_y.columns.values) == ['Bus02_V']

def test_select_feature_label_scaled_modeBusLF_flow_noknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V']
    assert list(y_le.columns.values) == ['Bus02_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus02_V','Bus04_V']

def test_select_feature_label_scaled_modeBusLF_flow_oneknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [1]
    known_buses_ind = [0]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus02_V','Bus01_p','Bus01_q']
    assert list(y_le.columns.values) == ['Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus04_V']


# test exteme inputs - all buses observed (known_buses_ind)

def test_select_feature_label_scaled_modeLE_flow_allknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = [0,1,2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V','Bus02_V','Bus04_V','Bus01_p','Bus02_p','Bus04_p',
                                        'Bus01_q','Bus02_q','Bus04_q']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLEF_flow_allknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = [0,1,2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V','Bus02_V','Bus04_V',
                                        'Bus01_p','Bus02_p','Bus04_p','Bus01_q','Bus02_q','Bus04_q']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLF_flow_allknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = [0,1,2]
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)

def test_select_feature_label_scaled_modeBusLF_flow_allknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0]
    known_buses_ind = [0,1,2]
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)


# test exteme inputs - no loads known (pq_ind_known empty)

def test_select_feature_label_scaled_modeLE_flow_noload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = []
    v_ind_known = [0]
    known_buses_ind = [0]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus01_p','Bus01_q']
    assert list(y_le.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLEF_flow_noload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = []
    v_ind_known = [0]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q',
                                        'Bus02_V']
    assert list(scaler_y.columns.values) == ['UL01_P','UL01_Q',
                                        'UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q',
                                        'Bus02_V']

def test_select_feature_label_scaled_modeDLF_flow_noload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = []
    v_ind_known = [0]
    known_buses_ind = [1]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus02_V','Bus02_p','Bus02_q']
    assert list(y_le.columns.values) == ['Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus04_V']


def test_select_feature_label_scaled_modeBusLF_flow_noload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = []
    v_ind_known = [0]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['Bus02_V']
    assert list(scaler_y.columns.values) == ['Bus02_V']


# test exteme inputs - all loads known (pq_ind_known)

def test_select_feature_label_scaled_modeLE_flow_allload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = [0]
    known_buses_ind = [0]
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)

def test_select_feature_label_scaled_modeDLEF_flow_allload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = [0]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q',
                                  'Bus01_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['Bus02_V']
    assert list(scaler_y.columns.values) == ['Bus02_V']

def test_select_feature_label_scaled_modeDLF_flow_allload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = [0]
    known_buses_ind = [1]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q',
                                  'Bus01_V','Bus02_V','Bus02_p','Bus02_q']
    assert list(y_le.columns.values) == ['Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus04_V']


def test_select_feature_label_scaled_modeBusLF_flow_allload(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = [0]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['Bus01_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['Bus02_V']
    assert list(scaler_y.columns.values) == ['Bus02_V']


# test exteme inputs - no voltage known (v_ind_known empty)

def test_select_feature_label_scaled_modeLE_flow_novolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = []
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLEF_flow_novolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = []
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q',
                                  'Bus01_V','Bus02_V']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q',
                                  'Bus01_V','Bus02_V']

def test_select_feature_label_scaled_modeDLF_flow_novolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = []
    known_buses_ind = [1]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q',
                                  'Bus02_V','Bus02_p','Bus02_q']
    assert list(y_le.columns.values) == ['Bus01_V','Bus04_V']
    assert list(scaler_y.columns.values) == ['Bus01_V','Bus04_V']


def test_select_feature_label_scaled_modeBusLF_flow_novolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = []
    known_buses_ind = []
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)


# test exteme inputs - all voltages known (v_ind_known)                                                               

def test_select_feature_label_scaled_modeLE_flow_allvolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0,1,2]
    known_buses_ind = []
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='LE',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V','Bus02_V','Bus04_V']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                        'UL03_P','UL03_Q',
                                        'UL04_P','UL04_Q',
                                        'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLEF_flow_allvolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1]
    v_ind_known = [0,1,2]
    known_buses_ind = [2]
    X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)
    assert list(X_le.columns.values) == ['UL01_P','UL01_Q','Bus01_V','Bus02_V','Bus04_V','Bus04_p','Bus04_q']
    assert list(y_le.columns.values) == ['UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q']
    assert list(scaler_y.columns.values) == ['UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q']

def test_select_feature_label_scaled_modeDLF_flow_allvolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = [0,1,2]
    known_buses_ind = [1]
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLF',known_buses_ind=known_buses_ind)

def test_select_feature_label_scaled_modeBusLF_flow_allvolt(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7]
    v_ind_known = [0,1,2]
    known_buses_ind = []
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='BusLF',known_buses_ind=known_buses_ind)


#  test exteme inputs - all known / all unknown

def test_select_feature_label_scaled_modeDLEF_flow_allknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = [0,1,2,3,4,5,6,7,8,9]
    v_ind_known = [0,1,2]
    known_buses_ind = []
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)

def test_select_feature_label_scaled_modeDLEF_flow_noknown(get_pq_v_df,get_flow_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    dataset.pflow_df, dataset.qflow_df = get_flow_df
    pq_ind_known = []
    v_ind_known = []
    known_buses_ind = []
    with pytest.raises(ValueError):
        X_le,y_le, scaler_y = select_feature_label_scaled_scenario(pq_ind_known,v_ind_known,dataset,mode='DLEF',known_buses_ind=known_buses_ind)





@pytest.mark.xfail
def test_select_feature_label_scaled_LF():
    pass




def test_replace_columns_with_prediction_normal2(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    prediction_LE = np.ones(20).reshape(10,2)
    y_pr_columns = ['UL01_P','UL01_Q']

    pq_df = replace_columns_with_prediction(dataset,prediction_LE,y_pr_columns)
    assert pq_df.loc[0,'UL01_P'] == 1
    assert pq_df.loc[5,'UL01_P'] == 1
    assert pq_df.loc[9,'UL01_P'] == 1
    assert pq_df.loc[1,'UL01_Q'] == 1
    assert pq_df.loc[4,'UL01_Q'] == 1
    assert pq_df.loc[7,'UL01_Q'] == 1
    assert pq_df.loc[:,'UL02_P'].mean() < 1
    assert list(pq_df.columns) == ['UL01_P','UL01_Q',
                                  'UL02_P','UL02_Q',
                                  'UL03_P','UL03_Q',
                                  'UL04_P','UL04_Q',
                                  'UL05_P','UL05_Q']

def test_replace_columns_with_prediction_normal4(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, dataset.v_df = get_pq_v_df
    prediction_LE = np.ones(40).reshape(10,4)
    y_pr_columns = ['UL03_P','UL03_Q','UL05_P','UL05_Q']

    pq_df = replace_columns_with_prediction(dataset,prediction_LE,y_pr_columns)
    assert pq_df.loc[0,'UL03_P'] == 1
    assert pq_df.loc[5,'UL03_Q'] == 1
    assert pq_df.loc[9,'UL03_P'] == 1
    assert pq_df.loc[1,'UL05_Q'] == 1
    assert pq_df.loc[4,'UL05_P'] == 1
    assert pq_df.loc[7,'UL05_Q'] == 1