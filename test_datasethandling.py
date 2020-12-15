from main import load_dataset
from RNN import RNN

def test_load_dataset():
    '''
    Tests if malfunctions_in_LV_grid_dataset is correctly imported
    '''

    dataset, X, y = load_dataset(dataset='test')

    assert X.size == 1405400 or X.size == 2108100
    assert len(X) == 40 or len(X) == 60
    assert len(y) == 40 or len(y) == 60
    assert sum(y) == 10 or sum(y) == 15

def test_preprocessing():
    '''
    Tests if preprocessing is done correctly (zero meaning every sample by itself and max abs scaling fitted on the training set)
    '''

    X_train = [[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]]
    X_test = [[2,3,4,5], [0,6,7,8]]

    X_train, X_test = RNN.preprocess(X_train, X_test)

    assert all([i.mean() == 0 for i in X_train])
    assert all([i.mean() == 0 for i in X_test])
    assert max([i.max() for i in X_train]) == 1
    assert max([i.min() for i in X_train]) == -1
    assert max([i.max() for i in X_train]) > 1
    assert max([i.min() for i in X_train]) < -1


def test_postprocessing():



    return

