import config
from malfunctions_in_LV_grid_dataset import MlfctinLVdataset
from RNN import RNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import torch
from torch import nn

configuration = config.learning_config

def train():
    print("########## Configuration ##########")
    for key, value in configuration.items():
        print(key, ' : ', value)

    dataset, X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



    if configuration['classifier'] == 'RNN':
        clf = RNN(configuration['RNN model settings'][0], configuration['RNN model settings'][1], configuration['RNN model settings'][2], configuration['RNN model settings'][3]).fit(X_train,y_train)

    # Define hyperparameters
    n_epochs = 100
    lr = 0.01

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    y_pred = clf.predict(X_test)
    metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    print("########## Metrics ##########")
    print(
        "Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nFScore: {3}".format(accuracy, metrics[0], metrics[1],
                                                                         metrics[2]))

    return

def load_dataset():     #WRITE TESTCASE!!!

    if configuration['dataset'] == 'some_other_dataset':
        dataset = MlfctinLVdataset(configuration["some_other_dataset"])
    else:
        dataset = MlfctinLVdataset(configuration["malfunction_in_LV_grid_data"])    #default is created dataset

    X = dataset.get_x()
    y = dataset.get_y()
    return dataset, X, y
