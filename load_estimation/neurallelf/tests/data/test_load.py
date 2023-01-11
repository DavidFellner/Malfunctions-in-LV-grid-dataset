import pytest, os, sys

from neurallelf.data.load import *


###  Fixture section  ##############
##################################

@pytest.fixture
def tempdir_with_graph_data(tmpdir):
    file = tmpdir.join('graph.csv')
    df = pd.DataFrame(data=[['','','TypLod,TypLodind','Substation','',''],
                            ['UL05 ','Grid','','','Bus07',''],
                            [' UL04','Grid','','','\tBus06','']],
                    columns=['Name','Grid','Type','Terminal','Terminal.1','Zone'])
    df.to_csv(file,sep='\t')
    yield tmpdir


@pytest.fixture
def tempdir_with_files(tmpdir,request,scope='module'):
    #sys.stdout.write("\n"+str(type(request.param))+"\n")
    for filename in request.param:
        file = tmpdir.join(filename)
        with open(file,'w'):
            pass
    yield tmpdir


###  Tests section  ##############
##################################


FILE_LIST10 = [
    'CLUE_Test_01_df_load.csv',
    'CLUE_Test_01_full_df_load.csv', 
    'CLUE_Test_01_narrow_df_load.csv',
    'CLUE_Test_02_narrow_df_load.csv',
    'CLUE_Test_02narrow_df_load.csv',
    '04_df_load.csv',     
    '4_full_df_load.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST10],indirect=True)
def test_get_file_01_df_load(tempdir_with_files):
    path = tempdir_with_files
    name = '01'
    file_pattern = 'df_load'
    assert get_file(path,name,file_pattern) == ['CLUE_Test_01_df_load.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST10],indirect=True)
def test_get_file_01_full_df_load(tempdir_with_files):
    path = tempdir_with_files
    name = '01'
    file_pattern = 'full_df_load'
    assert get_file(path,name,file_pattern) == ['CLUE_Test_01_full_df_load.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST10],indirect=True)
def test_get_file_narrow_df_load(tempdir_with_files):
    path = tempdir_with_files
    name = '02'
    file_pattern = 'narrow_df_load'
    assert get_file(path,name,file_pattern) == ['CLUE_Test_02_narrow_df_load.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST10],indirect=True)
def test_get_file_04(tempdir_with_files):
    path = tempdir_with_files
    name = '04'
    file_pattern = 'df_load'
    assert get_file(path,name,file_pattern) == ['04_df_load.csv']

@pytest.mark.parametrize('tempdir_with_files',[FILE_LIST10],indirect=True)
def test_get_file_error(tempdir_with_files):
    path = tempdir_with_files
    name = ''
    file_pattern = 'df_load'
    with pytest.raises(ValueError):
        get_file(path,name,file_pattern)



def test_retrieve_loadBus_frame_len(tempdir_with_graph_data):
    data_path = tempdir_with_graph_data / 'graph.csv'
    df = retrieve_loadBus_frame(data_path)
    assert len(df)==2

def test_retrieve_loadBus_frame_load(tempdir_with_graph_data):
    data_path = tempdir_with_graph_data / 'graph.csv'
    df = retrieve_loadBus_frame(data_path)
    assert len(df.load) ==2
    assert list(df.loc[:,'load'].values) == ['UL05','UL04']

def test_retrieve_loadBus_frame_bus(tempdir_with_graph_data):
    data_path = tempdir_with_graph_data / 'graph.csv'
    df = retrieve_loadBus_frame(data_path)
    assert len(df.load) ==2
    assert list(df.loc[:,'bus'].values) == ['Bus07','Bus06']