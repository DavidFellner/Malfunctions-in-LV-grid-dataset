import pytest

from neurallelf.data.dataset_generators import *


###  Fixture section  ##############
##################################

@pytest.fixture
def get_datasetGasen():
    path = Path(os.path.dirname(__file__)) / "test_resources"
    filename = "df_res_10rows.csv"
    graph_name = "CLUE_Gasen_Load_Bus_Mapping.txt"
    dataset = DatasetGasen()
    dataset.create_dataset(
        data_path= path,
        directory="Gasen",
        name=filename,
        graph_name=graph_name)
    yield dataset

###  Tests section  ##############
##################################


def test_dataset_gasen_create_dataset_pqdf(get_datasetGasen):
    """
    Test if DataseGasen is loaded and established as intended from a CSV-file.
    """
    dataset = get_datasetGasen

    assert dataset.pq_df is not None
    assert len(dataset.pq_df.columns) == 150*2-4  # -4 from MS_Gasen since it is flows, V and phi
    assert dataset.pq_df.loc[0,'load_490_P']==pytest.approx(0.0011384012177586553)
    assert dataset.pq_df.loc[0,'load_462_P']==pytest.approx(0.0016329144127666948)
    assert dataset.pq_df.loc[0,'load_409_Q']==pytest.approx(0.0047902557998895645)
    assert dataset.pq_df.loc[0,'PV_499_Q']==pytest.approx(-0.004402207676321268)
    
def test_dataset_gasen_create_dataset_vdf(get_datasetGasen):
    """
    Test if DataseGasen is loaded and established as intended from a CSV-file.
    """
    dataset = get_datasetGasen

    assert dataset.v_df is not None
    assert len(dataset.v_df.columns) == 1436/4 + 1  # +1 from MS_Gasen since it is flows, V and phi
    assert dataset.v_df.loc[0,'node_183_V']==pytest.approx(0.9907160050302793)
    assert dataset.v_df.loc[0,'node_78_V']==pytest.approx(0.998201689801078)
    
def test_dataset_gasen_create_dataset_flowdf(get_datasetGasen):
    """
    Test if DataseGasen is loaded and established as intended from a CSV-file.
    """
    dataset = get_datasetGasen

    assert dataset.pflow_df is not None
    assert dataset.qflow_df is not None
    assert len(dataset.pflow_df.columns) == 1436/4 + 1  # +1 from MS_Gasen since it is flows, V and phi
    assert len(dataset.qflow_df.columns) == 1436/4 + 1  # +1 from MS_Gasen since it is flows, V and phi

    assert dataset.pflow_df.loc[0,'node_78_p']==pytest.approx(0.018540888791632456)
    assert dataset.qflow_df.loc[0,'node_78_q']==pytest.approx(0.02216468582664627)
    assert dataset.pflow_df.loc[1,'node_222_p']==pytest.approx(-1.1396602518780616e-08)
    assert dataset.qflow_df.loc[1,'node_222_q']==pytest.approx(6.734980671184163e-15)
    
def test_dataset_gasen_create_dataset_graphdf(get_datasetGasen):
    """
    Test if DataseGasen is loaded and established as intended from a CSV-file.
    """
    dataset = get_datasetGasen

    assert dataset.graph_df is not None
    assert list(dataset.graph_df.columns) == ['load','bus']
    assert dataset.graph_df.loc[dataset.graph_df.load=='PV_500','bus'].values == 'node_218'
    assert dataset.graph_df.loc[dataset.graph_df.load=='load_450','bus'].values == 'node_86'
    
    