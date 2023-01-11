import pytest

from neurallelf.features.feature import *
from neurallelf.data.dataset_generators import DatasetTestGrids as Dataset


###  Fixture section  ##############
##################################

@pytest.fixture
def get_df(request,scope='module'):
    columns = request.param
    df = pd.DataFrame(data = np.random.random(len(columns)).reshape(1,len(columns)),
                         columns=columns)
    yield df


###  Tests section  ##############
##################################

# DF_COL_LIST_NUM1 = ['UL01_p','Charging03_p','PV_sys_02_p','PV_sys2_54_p','UL49685_p']
# @pytest.mark.parametrize('get_df',[DF_COL_LIST_NUM1],indirect=True)
# def test_extract_col_numbers_normal(get_df):
#     df = get_df
#     assert extract_col_numbers(df) == [1,3,2,2,54,49685]



# DF_COL_LIST_NUM2 = ['UL_p','Charging03_p','PV_sys_02_p','PV_sys2_54_p','UL049685_p']
# @pytest.mark.parametrize('get_df',[DF_COL_LIST_NUM2],indirect=True)
# def test_new_col_numbers_normal(get_df):
#     df = get_df
#     largest_num = 49685
#     assert new_col_numbers(df,largest_num) == ['49686','03','02','49687','49685']

# DF_COL_LIST_NUM3 = ['UL_p','Charging_p','PV_p']
# @pytest.mark.parametrize('get_df',[DF_COL_LIST_NUM3],indirect=True)
# def test_new_col_numbers_none(get_df):
#     df = get_df
#     largest_num = 0
#     assert new_col_numbers(df,largest_num) == ['01','02','03']



DF_COL_LIST_UL1 = ['UL01_p','UL01.1_q','UL03_p','UL03.1_q','UL02_p','UL02.1_q',
'UL04_p','UL04.1_q','UL05_p','UL05.1_q']
@pytest.mark.parametrize('get_df',[DF_COL_LIST_UL1],indirect=True)
def test_get_df_part_p_normal(get_df):
    df = get_df
    assert list(get_df_part(df,'p').columns) == ['UL01_p','UL03_p','UL02_p','UL04_p','UL05_p']

DF_COL_LIST_UL2 = ['UL01_p','UL01.1_q','P_p_p','P_p.1_q']
@pytest.mark.parametrize('get_df',[DF_COL_LIST_UL2],indirect=True)
def test_get_df_part_p_extreme(get_df):
    df = get_df
    assert list(get_df_part(df,'p').columns) == ['UL01_p','P_p_p']

@pytest.mark.parametrize('get_df',[DF_COL_LIST_UL1],indirect=True)
def test_get_df_part_q_normal(get_df):
    df = get_df
    assert list(get_df_part(df,'q').columns) == ['UL01.1_q','UL03.1_q','UL02.1_q','UL04.1_q','UL05.1_q']

DF_COL_LIST_UL3 = ['UL01_p','UL01.1_q','P1_q_p','P_q.1_q']
@pytest.mark.parametrize('get_df',[DF_COL_LIST_UL3],indirect=True)
def test_get_df_part_q_extreme(get_df):
    df = get_df
    assert list(get_df_part(df,'q').columns) == ['UL01.1_q','P_q.1_q']

DF_COL_LIST_BUS1 = ['Bus01','Bus02','Bus04','bus05','bBus06','Busbar07','Busbus']
@pytest.mark.parametrize('get_df',[DF_COL_LIST_BUS1],indirect=True)
def test_get_df_part_bus_extreme(get_df):
    df = get_df
    assert list(get_df_part(df,'bus').columns) == ['Bus01','Bus02','Bus04']




# DF_COL_LIST_UL10 = ['UL01_p','UL01.1_q','UL03_p','UL03.1_q','UL02_p','UL02.1_q',
# 'UL04_p','UL04.1_q','UL05_p','UL05.1_q']
# @pytest.mark.parametrize('get_df',[DF_COL_LIST_UL10],indirect=True)
# def test_encode_columns_UL_swapped(get_df):
#     # two columns are swapped
#     df = get_df
#     type = 'load'
#     df_encoded, encode_dict, decode_dict = encode_columns(df,type)
#     assert list(df_encoded.columns) == ['UL01_p','UL01_q','UL02_p','UL02_q','UL03_p','UL03_q',
# 'UL04_p','UL04_q','UL05_p','UL05_q']
#     assert encode_dict['UL03_p'] == 'UL03_p'
#     assert decode_dict['UL02_p'] == 'UL02_p'

# DF_COL_LIST_UL11 = ['UL001_p','UL001.1_q','PVsystem_p','PVsystem.1_q','UL023_p','UL023.1_q',
# 'UL_p04_p','UL_p04.1_q','UL_q05_p','UL_q05.1_q']
# @pytest.mark.parametrize('get_df',[DF_COL_LIST_UL11],indirect=True)
# def test_encode_columns_UL_extremes(get_df):
#     df = get_df
#     type = 'load'
#     df_encoded, encode_dict, decode_dict = encode_columns(df,type)
#     assert list(df_encoded.columns) == ['UL01_p','UL01_q','UL04_p','UL04_q','UL05_p','UL05_q',
#     'UL23_p','UL23_q','UL24_p','UL24_q']
#     assert encode_dict['PVsystem_p'] == 'UL24_p'
#     assert decode_dict['UL05_q'] == 'UL_q05.1_q'

# DF_COL_LIST_BUS10 = ['Bus04','Bus02','Bus4','Busbar','Bus004']
# @pytest.mark.parametrize('get_df',[DF_COL_LIST_BUS10],indirect=True)
# def test_encode_columns_bus(get_df):
#     df = get_df
#     type = 'bus'
#     df_encoded, encode_dict, decode_dict = encode_columns(df,type)
#     assert list(df_encoded.columns) == ['Bus02','Bus04','Bus05','Bus06','Bus07']
#     assert encode_dict['Bus04'] == 'Bus04'
#     assert decode_dict['Bus07'] == 'Bus004'

# @pytest.mark.parametrize('get_df',[DF_COL_LIST_BUS10],indirect=True)
# def test_encode_columns_bus_encode(get_df):
#     df = get_df
#     type = 'bus'
#     encode_dict = {'Bus04':'Bus04','Bus02':'Bus02','Bus4':'Bus00','Busbar':'Bus03',
#     'Bus004':'Bus99'}
#     df_encoded, encode_dict, decode_dict = encode_columns(df,type,encode_dict)
#     assert list(df_encoded.columns) == ['Bus00','Bus02','Bus03','Bus04','Bus99']
#     assert encode_dict['Bus04'] == 'Bus04'
#     assert decode_dict['Bus99'] == 'Bus004'



# def test_dataset_encode_columns_noflow_noknown():
#     dataset = Dataset()
#     dataset.pq_df = pd.DataFrame(data = np.arange(1,11).reshape(1,10),
#                          columns=['UL1_p','UL1.1_q',
#                                   'UL04_p','UL04.1_q',
#                                   'load_3_p','load_3.1_q',
#                                   'PV_sys34_p','PV_sys34.1_q',
#                                   'Charging_p','Charging.1_q'])
#     dataset.v_df = pd.DataFrame(data = np.arange(1,4).reshape(1,3),
#                         columns=['Bus02',
#                                  'node_1',
#                                  'bus_bar_'])
#     dataset.graph_df = pd.DataFrame(data={'load':['load_3','Charging','UL1'],
#                                            'bus':['node_1','Bus02','bus_bar_']})
#     dataset.flow_present = False
#     dataset.known_buses = []
#     dataset_encoded = dataset_encode_columns(dataset)  

#     assert list(dataset_encoded.pq_df.columns) == ['UL01_p','UL01_q',
#                                                     'UL03_p','UL03_q',
#                                                     'UL04_p','UL04_q',
#                                                     'UL34_p','UL34_q',
#                                                     'UL35_p','UL35_q']                   
#     assert list(dataset_encoded.pq_df.values[0]) == [1,2,5,6,3,4,7,8,9,10]
#     assert list(dataset_encoded.v_df.columns) == ['Bus01','Bus02','Bus03']
#     assert list(dataset_encoded.v_df.values[0]) == [2,1,3]
#     assert dataset_encoded.load_encode_dict['load_3_p'] == 'UL03_p'
#     assert list(dataset_encoded.graph_df.loc[:,'load'].values) == ['UL03','UL35','UL01']
#     assert list(dataset_encoded.graph_df.loc[:,'bus'].values) == ['Bus01','Bus02','Bus03']

# def test_dataset_encode_columns_flow_known():
#     dataset = Dataset()
#     dataset.pq_df = pd.DataFrame(data = np.arange(1,11).reshape(1,10),
#                          columns=['UL001_p','UL001.1_q',
#                                   '4_p','4.1_q',
#                                   'load4_3_p','load4_3.1_q',
#                                   'PV_p_sys3_p','PV_p_sys3.1_q',
#                                   'd_p','d.1_q'])
#     dataset.v_df = pd.DataFrame(data = np.arange(1,4).reshape(1,3),
#                         columns=['Bus002',
#                                  'node_10',
#                                  'bus12_bar10_'])
#     dataset.pflow_df = pd.DataFrame(data = np.arange(4,7).reshape(1,3),
#                         columns=['node_10',
#                                  'Bus002',
#                                  'bus12_bar10_'])                            
#     dataset.qflow_df = pd.DataFrame(data = np.arange(7,10).reshape(1,3),
#                         columns=['bus12_bar10_',
#                                  'node_10',
#                                  'Bus002'])    
#     dataset.graph_df = pd.DataFrame(data={'load':['load4_3','d','4'],
#                                            'bus':['node_10','bus12_bar10_','Bus002']})
#     dataset.flow_present = True
#     dataset.known_buses = ['node_10']
#     dataset_encoded = dataset_encode_columns(dataset)  

#     assert list(dataset_encoded.pq_df.columns) == ['UL01_p','UL01_q',
#                                                     'UL03_p','UL03_q',
#                                                     'UL04_p','UL04_q',
#                                                     'UL05_p','UL05_q',
#                                                     'UL06_p','UL06_q']                   
#     assert list(dataset_encoded.pq_df.values[0]) == [1,2,7,8,3,4,5,6,9,10]
#     assert list(dataset_encoded.v_df.columns) == ['Bus02','Bus10','Bus12']
#     assert list(dataset_encoded.v_df.values[0]) == [1,2,3]
#     assert list(dataset_encoded.pflow_df.columns) == ['Bus02','Bus10','Bus12']
#     assert list(dataset_encoded.pflow_df.values[0]) == [5,4,6]
#     assert list(dataset_encoded.qflow_df.columns) == ['Bus02','Bus10','Bus12']
#     assert list(dataset_encoded.qflow_df.values[0]) == [9,8,7]

#     assert dataset_encoded.load_encode_dict['PV_p_sys3.1_q'] == 'UL03_q'
#     assert list(dataset_encoded.graph_df.loc[:,'load'].values) == ['UL05','UL06','UL04']
#     assert list(dataset_encoded.graph_df.loc[:,'bus'].values) == ['Bus10','Bus12','Bus02']