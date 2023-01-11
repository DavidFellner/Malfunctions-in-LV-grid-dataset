import pytest

import pandas as pd
import numpy as np

from neurallelf.visualization.load_estimation_viz import *
from neurallelf.data.dataset_generators import DatasetTestGrids as Dataset
from neurallelf.models.load_estimation import LEDTO



###  Module fixtures section  ##############
############################################

@pytest.fixture
def get_pq_v_df():
    pq_df = pd.DataFrame(data = np.random.random(100).reshape(10,10),
                         columns=['UL01_p','UL01_q',
                                  'UL02_p','UL02_q',
                                  'UL03_p','UL03_q',
                                  'UL04_p','UL04_q',
                                  'UL05_p','UL05_q'])
    v_df = pd.DataFrame(data = np.random.random(30).reshape(10,3),
                        columns=['Bus01',
                                 'Bus02',
                                 'Bus04'])
    yield pq_df,v_df



###  Tests for visualization functions  ####
############################################


def test_get_x_labels(get_pq_v_df):
    dataset = Dataset()
    dataset.pq_df, _ = get_pq_v_df
    pq_ind_known = [[0,1],[6,7,2,3],[0,1,2,3,4,5,6,7,8,9]]
    ledto = LEDTO(pq_ind_known)
    a, b = get_x_labels(dataset,ledto)
    assert a[0][0] == 'UL01_p'
    assert a[0][1] == 'UL01_q'
    assert b == [[1],[2,4],[1,2,3,4,5]]