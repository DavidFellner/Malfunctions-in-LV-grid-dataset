from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import pandas as pd

class Measurement:

    def __init__(self, data, name):

        self.name = name
        self.data = data

    def pca(self, vars, var_numbers, analysis=False, n_components=2):

        try:
            selected_data = self.data.loc[:, self.data.columns[var_numbers]].values
        except IndexError:
            print('Check variables!')
            return ()
        selected_data = StandardScaler().fit_transform(selected_data)

        selected_data_normalised = pd.DataFrame(selected_data, columns=vars)

        if analysis:
            #to do analysis on the dimension reduction done on data to find most important features
            pca = PCA()     #to find out how many components needed
            X_pca = pca.fit(selected_data_normalised)
            dimensions_cut_off_value = np.where(np.around(np.cumsum(pca.explained_variance_ratio_), decimals=2) == 0.99)[0][0]
            #fig, ax = plt.subplots()
            #ax.plot(np.cumsum(pca.explained_variance_ratio_))
            #ax.set_xlabel('number of components')
            #ax.set_ylabel('cumulative explained variance');

            pca = PCA(n_components=0.99)
            X_pca = pca.fit_transform(selected_data_normalised)  # this will fit and reduce dimensions
            #print(pca.n_components_)  # one can print and see how many components are selected.

            contribution_to_component = pd.DataFrame(pca.components_, columns=selected_data_normalised.columns)

            n_pcs = pca.n_components_  # get number of component# get the index of the most important feature on EACH component
            most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
            initial_feature_names = selected_data_normalised.columns
            # get the most important feature names
            most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
        else:
            dimensions_cut_off_value = None
            most_important_names = None

        #to show more graphically
        pca = PCA(n_components=n_components)
        principalComponents_selection = pca.fit_transform(selected_data_normalised)
        explained_variance = pca.explained_variance_ratio_

        return (principalComponents_selection, explained_variance, dimensions_cut_off_value, most_important_names)

    def kpca(self, vars, var_numbers):

        try:
            selected_data = self.data.loc[:, self.data.columns[var_numbers]].values
        except IndexError:
            print('Check variables!')
            return ()
        selected_data = StandardScaler().fit_transform(selected_data)

        selected_data_normalised = pd.DataFrame(selected_data, columns=vars)

        kpca= KernelPCA(n_components=2, kernel='rbf')
        principalComponents_selection = kpca.fit_transform(selected_data_normalised)

        #variance of components > not explained variance in original space!
        explained_variance = np.var(principalComponents_selection, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)

        # cumulative proportion explained variance
        np.cumsum(explained_variance_ratio)

        return (principalComponents_selection, explained_variance)

    def ssa(self, vars, var_numbers):

        # Author: Johann Faouzi <johann.faouzi@gmail.com>
        # License: BSD-3-Clause

        import numpy as np
        import matplotlib.pyplot as plt
        from pyts.decomposition import SingularSpectrumAnalysis

        # Parameters
        n_samples, n_timestamps = 100, 48

        # Toy dataset
        rng = np.random.RandomState(41)
        X = rng.randn(n_samples, n_timestamps)

        # We decompose the time series into three subseries
        window_size = 15
        groups = [np.arange(i, i + 5) for i in range(0, 11, 5)]

        # Singular Spectrum Analysis
        ssa = SingularSpectrumAnalysis(window_size=15, groups=groups)
        X_ssa = ssa.fit_transform(X)

        # Show the results for the first time series and its subseries
        plt.figure(figsize=(16, 6))

        ax1 = plt.subplot(121)
        ax1.plot(X[0], 'o-', label='Original')
        ax1.legend(loc='best', fontsize=14)

        ax2 = plt.subplot(122)
        for i in range(len(groups)):
            ax2.plot(X_ssa[0, i], 'o--', label='SSA {0}'.format(i + 1))
        ax2.legend(loc='best', fontsize=14)

        plt.suptitle('Singular Spectrum Analysis', fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

        # The first subseries consists of the trend of the original time series.
        # The second and third subseries consist of noise.

        return 0