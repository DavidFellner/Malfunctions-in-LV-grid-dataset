import seaborn as sns
import scipy
import sklearn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib
from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

class Clustering:

    def __init__(self, data, variables, metric='ward', method='euclidean', num_of_clusters=2):

        self.dataset = data
        self.variables = variables
        self.metric = metric
        self.method = method
        self.num_of_clusters= num_of_clusters


    def cluster(self):

        dict_of_variable_dfs = Clustering.repack(self)
        similarity_matrix = Clustering.similarity_measure(self)
        linkage_matrix = Clustering.link_clusters(self)
        predictions = Clustering.form_clusters(self)
        #score = Clustering.rand_score()

        return self.assign_clusters

    def repack(self):

        dict_of_variable_dfs = {}
        for variable in self.variables:
            dict_of_variable_dfs[variable] = pd.DataFrame()

            for measurement, label in zip(self.dataset.X, self.dataset.y):
                dict_of_variable_dfs[variable] = pd.concat((dict_of_variable_dfs[variable], pd.DataFrame(index= measurement.data.index, data=measurement.data[variable].values, columns=[(measurement.name, str(label))])), axis=1)

        self.data_dict = dict_of_variable_dfs

        return dict_of_variable_dfs



    def similarity_measure(self):

        #average over all variable correlations
        columns_and_index = [(self.dataset.X[i].name, str(self.dataset.y[i])) for i in range(len(self.dataset.X))]
        corr_matrix_dict_sorted_by_var = {}
        corr_matrix_dict_sorted_by_measurement = {i: {i:[] for i in columns_and_index} for i in columns_and_index}
        corr_matrix = {i: {i:None for i in columns_and_index} for i in columns_and_index}
        df_corr_matrix = pd.DataFrame(index=columns_and_index)

        for variable in self.data_dict.keys():
            check_matrix = self.data_dict[variable].corr(method='pearson')
            corr_matrix_dict_sorted_by_var[variable] = self.data_dict[variable].corr(method='pearson').to_dict()    #split data up in dictionaries of same features

        for variable in corr_matrix_dict_sorted_by_var.keys():
            for measurement in columns_and_index:
                for other_measurement in corr_matrix_dict_sorted_by_var[variable][measurement]:
                    corr_matrix_dict_sorted_by_measurement[measurement][other_measurement].append(corr_matrix_dict_sorted_by_var[variable][measurement][other_measurement])

        for measurement in corr_matrix_dict_sorted_by_measurement:
            for other_measurement in corr_matrix_dict_sorted_by_measurement[measurement]:
                average_of_correlations_over_all_variables = np.nanmean(corr_matrix_dict_sorted_by_measurement[measurement][other_measurement])
                corr_matrix[measurement][other_measurement] = average_of_correlations_over_all_variables

        df_corr_matrix = df_corr_matrix.from_dict(corr_matrix)

        plt.figure()
        sns.heatmap(df_corr_matrix)
        plt.figure()
        sns.set(font_scale=1.175)
        ticks = [' '.join(i[0].split(' ')[:2] + i[0].split(' ')[4:6])[:-1] for i in df_corr_matrix.columns]
        ticks_coded = [ i.split(' ')[0][0] + '. c. S. '+ i.split(' ')[-1] for i in ticks]
        plot_matrix = pd.DataFrame(columns= ticks_coded, index = ticks_coded, data = df_corr_matrix.values)
        clustermap = sns.clustermap(plot_matrix, method='ward')
        ax = clustermap.ax_heatmap
        ax.set_ylabel("")
        ax.set_xlabel("")

        if config.save_figures:
            clustermap.savefig(os.path.join(config.raw_data_folder, 'Clustering',
                                                   learning_config[
                                                           'setup_chosen'] + '_' + learning_config[
                                                           'data_source'] + '_' + learning_config[
                                                           'mode']))
            clustermap.savefig(os.path.join(config.raw_data_folder, 'Clustering',
                                            learning_config[
                                                'setup_chosen'] + '_' + learning_config[
                                                'data_source'] + '_' + learning_config[
                                                'mode'] + '.pdf'), format='pdf')

        self.corr_matrix = df_corr_matrix

        return df_corr_matrix

    def link_clusters(self):

        Z = scipy.cluster.hierarchy.linkage(self.corr_matrix, method=self.method, metric=self.metric)
        self.linkage_matrix = Z

        return Z

    def form_clusters(self):

        clusters = scipy.cluster.hierarchy.fcluster(self.linkage_matrix, criterion='maxclust', t = self.num_of_clusters)
        self.clusters = clusters

        assigned_clusters = []

        self.cluster_correct, self.cluster_wrong, self.cluster_inversed = Clustering.assign_clusters(self)
        for observation in clusters:
            if observation == self.cluster_correct: assigned_clusters.append(0)
            elif observation == self.cluster_wrong: assigned_clusters.append(1)
            elif observation == self.cluster_inversed: assigned_clusters.append(2)

        self.assign_clusters = np.array(assigned_clusters)

        return clusters, assigned_clusters

    def assign_clusters(self):

        most_common_cluster_correct_samples = max(set(list(self.clusters[:15])), key=list(self.clusters[:15]).count)
        if self.num_of_clusters == 3:
            most_common_cluster_wrong_samples = max(set(list(self.clusters[15:30])), key=list(self.clusters[15:30]).count)
            most_common_cluster_inversed_samples = max(set(list(self.clusters[30:45])), key=list(self.clusters[30:45]).count)
        else:
            most_common_cluster_inversed_samples = 3
            if most_common_cluster_correct_samples == 2:
                most_common_cluster_wrong_samples = 1
            else:
                most_common_cluster_wrong_samples = 2

        return most_common_cluster_correct_samples, most_common_cluster_wrong_samples, most_common_cluster_inversed_samples



    def rand_score(self):

        score = sklearn.metrics.rand_score(self.dataset.y, self.assign_clusters) #sklearn.metrics.rand_score(labels_true, labels_pred)
        self.score = score

        return score






