import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

class MlfctinLVdataset:

    def __init__(self, file_path):
        self._file_path = file_path

        self._df = pd.read_csv(file_path, header=0, sep=';', decimal='.', low_memory=False)
        self._X = np.array(np.transpose(self._df.drop(["Unnamed: 0"], axis=1)[:-1].astype(np.float32)))
        self._y = list(self._df.iloc[-1].copy()[1:].astype(int))

        self._target_names = ['0: regular behaviour', '1: malfunction occured']

        self._le = LabelEncoder()
        self._y[:] = self._le.fit_transform(self._y)
        torch.load('data/' + ID + '.pt')

    def get_x(self):
        return self._X

    def get_target_names(self):
        return self._target_names

    def get_y(self):
        return self._y

    def get_le(self):
        return self._le
