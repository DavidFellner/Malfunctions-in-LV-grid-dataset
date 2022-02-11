import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

class PCA_Dataset:
    '''
    PCA has been already perfomed on samples of this dataset; the features are variances explained by PCs
    '''

    def __init__(self, data, name, classes=None, bay='F2', Setup='A', labelling='classification'):

        self.name = name
        self.data = data
        self.labelling = labelling

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

        self.X = np.array([data[key][1] for key in data if key[-2:] == name.split('_')[2] and key.split(' ')[3] == name.split('_')[1] and key.split(' ')[0] in self.classes])
        self.y = []
        for key in data:
            if key[-2:] == bay and key.split(' ')[3] == Setup:
                if key.split(' ')[0] in classes and self.labelling=='classification':
                    self.y = self.y + [classes.index(key.split(' ')[0])]
                elif self.labelling == 'detection':
                    if key.split(' ')[0] == 'correct':
                        self.y = self.y + [0]
                    elif key.split(' ')[0] == 'wrong':
                        self.y = self.y + [1]
                    elif key.split(' ')[0] == 'inversed':
                        self.y = self.y + [1]
        self.y = np.array(self.y)

class Raw_Dataset:
    '''
        raw measurements
    '''

    def __init__(self, data, name, classes=None, bay='F2', Setup='A', labelling='classification'):

        self.name = name
        self.data = data
        self.labelling = labelling

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

        self.X = np.array([data[measurement] for measurement in data if measurement[-2:] == name.split('_')[2] and measurement.split(' ')[3] == name.split('_')[1] and measurement.split(' ')[0] in self.classes])
        self.y = []
        for measurement in self.X:
            if measurement.name[-2:] == bay and measurement.name.split(' ')[3] == Setup:
                if measurement.name.split(' ')[0] in classes and self.labelling=='classification':
                    self.y = self.y + [classes.index(measurement.name.split(' ')[0])]
                elif self.labelling == 'detection':
                    if measurement.name.split(' ')[0] == 'correct':
                        self.y = self.y + [0]
                    elif measurement.name.split(' ')[0] == 'wrong':
                        self.y = self.y + [1]
                    elif measurement.name.split(' ')[0] == 'inversed':
                        self.y = self.y + [1]
        self.y = np.array(self.y)

class Combined_Dataset:
    '''
    combines all data in one dataframe with the measurements as an index and the variables at timesteps as columns (as in ('var1',t1) ('var2',t1), ('var3',t1), ('var1',t2), ('var2',t2) ...)
    then pca is done reduce to the most important variables on the most important timesteps
    '''

    def __init__(self, data, variables, name, classes=None, bay='F2', setup='A', labelling='classification'):

        self.name = name
        self.variables = variables
        self.labelling = labelling
        self.bay = bay
        self.setup = setup

        if classes is not None:
            self.classes = classes
        else:
            self.classes = ['correct', 'wrong']

        self.data = {applicable_measurements.name: data[applicable_measurements.name] for applicable_measurements in [data[measurement] for measurement in data if
                                                               measurement[-2:] == name.split('_')[2] and
                                                               measurement.split(' ')[3] == name.split('_')[1] and
                                                               measurement.split(' ')[0] in self.classes]}

        measurements = {}
        for measurement in data:
            if measurement[-2:] == name.split('_')[2] and measurement.split(' ')[3] == name.split('_')[1] and measurement.split(' ')[0] in self.classes:
                reduced_measurement = pd.DataFrame(index=data[measurement].data.index,
                                                                     data=data[measurement].data[self.variables].values,
                                                                     columns=variables)
                measurements[measurement] = Combined_Dataset.flatten_df_into_row(self, reduced_measurement)

        self.combined_data = pd.DataFrame(index=[self.data[measurement].name for measurement in self.data], data=[measurements[measurement].values[0] for measurement in measurements], columns=measurements[list(measurements.keys())[0]].columns)




    def label(self):

        self.X = np.array(self.principalComponents_selection)

        """self.X = np.array([self.data[measurement] for measurement in self.data if
                           measurement[-2:] == self.name.split('_')[2] and measurement.split(' ')[3] == self.name.split('_')[
                               1] and measurement.split(' ')[0] in self.classes])"""
        self.y = []
        for measurement in self.data:
            if measurement[-2:] == self.bay and measurement.split(' ')[3] == self.setup:
                if measurement.split(' ')[0] in self.classes and self.labelling == 'classification':
                    self.y = self.y + [self.classes.index(measurement.split(' ')[0])]
                elif self.labelling == 'detection':
                    if measurement.split(' ')[0] == 'correct':
                        self.y = self.y + [0]
                    elif measurement.split(' ')[0] == 'wrong':
                        self.y = self.y + [1]
                    elif measurement.split(' ')[0] == 'inversed':
                        self.y = self.y + [1]
        self.y = np.array(self.y)

        return self.X, self.y

    def PCA(self, n_components=0.99):

        pca = PCA(n_components=n_components)

        self.principalComponents_selection = pca.fit_transform(self.combined_data_scaled)
        self.explained_variance = pca.explained_variance_ratio_

        return self.principalComponents_selection

    def scale(self):

        combined_data_scaled = StandardScaler().fit_transform(self.combined_data)
        self.combined_data_scaled = pd.DataFrame(index=self.combined_data.index, data=combined_data_scaled, columns=self.combined_data.columns)

        return self.combined_data_scaled

    def flatten_df_into_row(self, df):

        v = df.unstack().to_frame().sort_index(level=1).T
        v.columns = v.columns.map(str)

        return v