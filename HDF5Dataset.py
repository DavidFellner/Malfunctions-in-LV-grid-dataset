import h5py
from pathlib import Path
import torch
from torch.utils import data

class HDF5Dataset(data.Dataset):
    def __init__(self, archive, type):
        self.archive = archive
        self.phase = type

    def __getitem__(self, index):
        with h5py.File(self.archive, 'r', libver='latest', swmr=True) as archive:
            datum = archive['x_' + str(self.phase)][index, :]
            datum_raw = archive['x_raw_' + str(self.phase)][index, :]
            label = archive['y_' + str(self.phase)][index]
            return datum, label, datum_raw

    def __len__(self):
        with h5py.File(self.archive, 'r', libver='latest', swmr=True) as archive:
            datum = archive['x_' + str(self.phase)]
            return len(datum)
