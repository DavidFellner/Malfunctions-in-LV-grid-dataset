import importlib
from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
import pandas as pd
import numpy as np
import random
import os

def add_noise(df):

    if config.add_noise:
        biggest_error_value = (config.smartmeter_voltage_range[1] / config.smartmeter_ratedvoltage_range[
            0]) * config.accuracy
        times_std = 3  # means biggest error value (i.e maximum value of smartmeter scale) is bigger than 98.8% of distribution
        mean = 0
        num_samples = len(df)
        calibrated = False

        while calibrated == False:
            std = biggest_error_value / times_std   # biggest error should be 1% of expected value (or of maximum of scale of measuring device)
            samples = np.random.normal(mean, std, size=num_samples)
            if samples.max() > biggest_error_value:
                times_std = times_std + 0.1
            else:
                calibrated = True

        if config.just_voltages:
            df_noised = df[('ElmTerm', 'm:u')] + samples
        else:
            df_noised = df.drop([('ElmTerm', 'm:u')], axis = 1)
            df_noised[('ElmTerm', 'm:u')] = (df[('ElmTerm', 'm:u')] + samples)
        return df_noised
    else:
        if config.just_voltages:
            try:
                df_noised = df[('ElmTerm', 'm:u')]
            except KeyError:
                df_noised = df
        else:
            df_noised = df
        return df_noised

def add_samples(train_samples, test_samples, num_features, sample_dict, samples_per_term, samples_before, label, number_of_samples_per_file, positive_test_samples_so_far, test_samples_so_far, dummy=False):

    for key in random.sample(list(sample_dict), samples_per_term):
        sample = sample_dict[key]
        sample_number = int(len(train_samples.columns) + len(test_samples.columns) / num_features)

        noised_data = add_noise(sample)
        if config.train_test_split == int:
            share_of_test_samples = config.train_test_split / config.number_of_samples
        else:
            share_of_test_samples = config.train_test_split

        if int(1/share_of_test_samples) <= number_of_samples_per_file:
            criterion1 = (int(number_of_samples_per_file * samples_before * share_of_test_samples) < share_of_test_samples*config.number_of_samples) #should always be true
            criterion2 = True
        else:
            criterion1 = (test_samples_so_far < share_of_test_samples*config.number_of_samples)
            if (label == 1 and (positive_test_samples_so_far < share_of_test_samples*config.number_of_samples*config.share_of_positive_samples)):
                criterion2 = True
            elif (label == 0 and ((test_samples_so_far - positive_test_samples_so_far) < (share_of_test_samples*config.number_of_samples*(1-config.share_of_positive_samples)))):
                criterion2 = True
            else:
                criterion2 = False
        if sample_number % int(1/share_of_test_samples) == 0 and criterion1 and criterion2:
            if num_features > 1:
                for i in noised_data.columns:
                    if dummy:
                        test_samples[(str(sample_number + samples_before), i[1])] = \
                        [noised_data[i].values.tolist()[0]] * len(noised_data[i]) + [label]
                    else:
                        test_samples[(str(sample_number + samples_before), i[1])] = noised_data[i].values.tolist() + [label]

            else:
                if dummy:
                    test_samples[str(sample_number + samples_before)] = \
                    [noised_data.values.tolist()[0]] * len(noised_data) + [label]
                else:
                    test_samples[str(sample_number + samples_before)] = noised_data.values.tolist() + [
                                label]
        else:
            if num_features > 1:
                for i in noised_data.columns:
                    if dummy:
                        train_samples[(str(sample_number + samples_before), i[1])] = \
                            [noised_data[i].values.tolist()[0]] * len(noised_data[i]) + [label]
                    else:
                        train_samples[(str(sample_number + samples_before), i[1])] = noised_data[i].values.tolist() + [label]

            else:
                if dummy:
                    train_samples[str(sample_number + samples_before)] = \
                        [noised_data.values.tolist()[0]] * len(noised_data) + [label]
                else:
                    train_samples[str(sample_number + samples_before)] = noised_data.values.tolist() + [
                        label]

    return train_samples, test_samples

def extract_malfunction_data(df, combinations_already_in_dataset, number_of_samples_before, positive_test_samples_so_far, test_samples_so_far):
    '''

    :param df:
    :param combinations_already_in_dataset:
    :return:

    extracts data of interest (no duplicates) and labels it with 1 for malfunction present, and 0 with no malfunction present
    also adds noise to data
    '''

    try:
        metainfo = df[('metainfo', 'in the first', 'few indices')]
    except KeyError:
        metainfo = df[str(('metainfo', 'in the first', 'few indices'))]
    terminals_with_malfunctions = [i for i in metainfo.iloc[5].split("'") if 'Bus' in i]
    if config.type == 'PV':
        line = 6
    elif config.type == 'EV':
        line = 7
    terminals_with_devices = [i for i in metainfo.iloc[line].split("'") if 'Bus' in i]
    start_time = metainfo[3].split(': ')[1].split('+')[0]

    sample_length = config.sample_length
    samples_to_go = config.number_of_samples - number_of_samples_before
    share_from_df = 1 / (config.simruns * config.number_of_grids)                                          # share of samples taken from current df
    if samples_to_go < int(config.number_of_samples * share_from_df):
        samples_from_df = samples_to_go
    else:
        samples_from_df = int(config.number_of_samples * share_from_df)

    num_positive_samples = samples_from_df * config.share_of_positive_samples
    num_neg_samples = samples_from_df * (1 - config.share_of_positive_samples)
    pos_samples_per_term = num_positive_samples / len(terminals_with_malfunctions)
    neg_samples_per_term = num_neg_samples / (len(terminals_with_devices) - len(terminals_with_malfunctions))

    difference_by_flooring = int(pos_samples_per_term) * len(terminals_with_malfunctions) - int(neg_samples_per_term) * (
            len(terminals_with_devices) - len(terminals_with_malfunctions))

    train_samples = pd.DataFrame(index=df.index[:sample_length].append(pd.Index(['label'])))
    test_samples = pd.DataFrame(index=df.index[:sample_length].append(pd.Index(['label'])))
    try:
        features_per_sample = len(df[terminals_with_devices[0]].columns)
    except KeyError:
        features_per_sample = 1

    if len(combinations_already_in_dataset) > 0:
        for combination in combinations_already_in_dataset:
            if set(terminals_with_devices) == set(combination[0]) \
                    and (terminals_with_malfunctions) == combination[1] \
                        and (terminals_with_malfunctions) == combination[2]:
                print('Combination already in dataset, file skipped!')
                return train_samples, test_samples, combinations_already_in_dataset

    combinations_already_in_dataset.append((terminals_with_devices, terminals_with_malfunctions, start_time))

    for term in terminals_with_devices:
        try:
            sample_dict = {name: group for name, group in df[term].groupby(np.arange(len(df[term])) // sample_length) if
                           len(group) == sample_length}
        except KeyError:
            try:
                sample_dict = {name: group for name, group in df[str((term, 'L1'))].groupby(np.arange(len(df[str((term, 'L1'))])) // sample_length) if
                               len(group) == sample_length}
            except KeyError:
                print('Broken data file')

        if term in terminals_with_malfunctions:

            if difference_by_flooring < 0:
                difference_by_flooring = difference_by_flooring + 1
                pos_samples = int(pos_samples_per_term) + 1
            else:
                pos_samples = int(pos_samples_per_term)

            train_samples, test_samples = add_samples(train_samples, test_samples, features_per_sample, sample_dict, pos_samples,
                                                      number_of_samples_before, 1, len(sample_dict.keys()), positive_test_samples_so_far, test_samples_so_far)

        else:

            if difference_by_flooring > 0:
                difference_by_flooring = difference_by_flooring - 1
                neg_samples = int(neg_samples_per_term) + 1
            else:
                neg_samples = int(neg_samples_per_term)

            train_samples, test_samples = add_samples(train_samples, test_samples, features_per_sample, sample_dict, neg_samples,
                                                      number_of_samples_before, 0, len(sample_dict.keys()), positive_test_samples_so_far, test_samples_so_far)

    return train_samples, test_samples, combinations_already_in_dataset

def extract_PV_noPV_data(df, combinations_already_in_dataset, number_of_samples_before):
    '''

    :param df:
    :param combinations_already_in_dataset:
    :return:

    extracts data of interest (no duplicates) and labels it with 1 for a terminal with a PV, and 0 for a terminal without PV; optionally adds noise to data (see config)
    '''

    metainfo = df[('metainfo', 'in the first', 'few indices')]
    terminals_with_loads = [i for i in metainfo.iloc[6].split("'") if 'Bus' in i]
    terminals_with_PV = [i for i in metainfo.iloc[5].split("'") if 'Bus' in i]

    sample_length = config.sample_length
    samples_to_go = config.number_of_samples - number_of_samples_before
    share_from_df = 1 / config.simruns  # share of samples taken from current df
    if samples_to_go < int(config.number_of_samples * share_from_df):
        samples_from_df = samples_to_go
    else:
        samples_from_df = int(config.number_of_samples * share_from_df)

    num_positive_samples = samples_from_df * config.share_of_positive_samples
    num_neg_samples = samples_from_df * (1 - config.share_of_positive_samples)
    pos_samples_per_term = num_positive_samples / len(terminals_with_PV)
    neg_samples_per_term = num_neg_samples / (len(terminals_with_loads) - len(terminals_with_PV))

    difference_by_flooring = int(pos_samples_per_term) * len(terminals_with_PV) - int(neg_samples_per_term) * (
                len(terminals_with_loads) - len(terminals_with_PV))

    df_reduced = pd.DataFrame(index=df.index[:sample_length].append(pd.Index(['label'])))
    features_per_sample = len(df[terminals_with_loads[0]].columns)

    if len(combinations_already_in_dataset) > 0:
        for combination in combinations_already_in_dataset:
            if set(terminals_with_PV) == set(combination):
                return df_reduced, combinations_already_in_dataset
    else:
        combinations_already_in_dataset.append(terminals_with_PV)

    for term in terminals_with_loads:
        sample_dict = {name: group for name, group in df[term].groupby(np.arange(len(df[term])) // sample_length) if
                       len(group) == sample_length}

        if term in terminals_with_PV:

            if difference_by_flooring < 0:
                difference_by_flooring = difference_by_flooring + 1
                pos_samples = int(pos_samples_per_term) + 1
            else:
                pos_samples = int(pos_samples_per_term)

            df_reduced = add_samples(df_reduced, features_per_sample, sample_dict, pos_samples,
                                     number_of_samples_before, 1)

        else:

            if difference_by_flooring > 0:
                difference_by_flooring = difference_by_flooring - 1
                neg_samples = int(neg_samples_per_term) + 1
            else:
                neg_samples = int(neg_samples_per_term)

            df_reduced = add_samples(df_reduced, features_per_sample, sample_dict, neg_samples,
                                     number_of_samples_before, 0)

    return df_reduced, combinations_already_in_dataset


def extract_dummy_data(df, combinations_already_in_dataset, number_of_samples_before):
    '''

    :param df:
    :param combinations_already_in_dataset:
    :return:

    extracts data of interest (no duplicates) and labels it with 1 for actual data, and 0 for dummy data of constant value
    '''

    metainfo = df[('metainfo', 'in the first', 'few indices')]
    terminals_with_loads = [i for i in metainfo.iloc[4].split("'") if 'Bus' in i]
    terminals_with_PV = [i for i in metainfo.iloc[3].split("'") if 'Bus' in i]

    sample_length = config.sample_length
    samples_to_go = config.number_of_samples - number_of_samples_before
    share_from_df = 1 / config.simruns                                          # share of samples taken from current df
    if samples_to_go < int(config.number_of_samples * share_from_df):
        samples_from_df = samples_to_go
    else:
        samples_from_df = int(config.number_of_samples * share_from_df)

    num_positive_samples = samples_from_df * config.share_of_positive_samples
    num_neg_samples = samples_from_df * (1-config.share_of_positive_samples)
    pos_samples_per_term = num_positive_samples / len(terminals_with_PV)
    neg_samples_per_term = num_neg_samples / (len(terminals_with_loads) - len(terminals_with_PV))

    difference_by_flooring = int(pos_samples_per_term) * len(terminals_with_PV) - int(neg_samples_per_term) * (len(terminals_with_loads) - len(terminals_with_PV))

    df_reduced = pd.DataFrame(index= df.index[:sample_length].append(pd.Index(['label'])))
    features_per_sample = len(df[terminals_with_loads[0]].columns)

    if len(combinations_already_in_dataset) > 0:
        for combination in combinations_already_in_dataset:
            if set(terminals_with_PV) == set(combination):
                return df_reduced, combinations_already_in_dataset
    else:
        combinations_already_in_dataset.append(terminals_with_PV)

    for term in terminals_with_loads:
        sample_dict = {name: group for name, group in df[term].groupby(np.arange(len(df[term])) // sample_length) if len(group) == sample_length}

        if term in terminals_with_PV:

            if difference_by_flooring < 0:
                difference_by_flooring = difference_by_flooring + 1
                pos_samples = int(pos_samples_per_term) + 1
            else:
                pos_samples = int(pos_samples_per_term)

            df_reduced = add_samples(df_reduced, features_per_sample, sample_dict, pos_samples,
                                     number_of_samples_before, 1)

        else:

            if difference_by_flooring > 0:
                difference_by_flooring = difference_by_flooring - 1
                neg_samples = int(neg_samples_per_term) + 1
            else:
                neg_samples = int(neg_samples_per_term)

            df_reduced = add_samples(df_reduced, features_per_sample, sample_dict, neg_samples, number_of_samples_before, 0, dummy=True)

    return df_reduced, combinations_already_in_dataset

def create_samples(dir, file, combinations_already_in_dataset, number_of_samples_before, positive_test_samples_so_far, test_samples_so_far):

    if config.type == 'PV':
        df = pd.read_csv(os.path.join(dir, file), header = [0,1,2],sep=';', low_memory=False)
    else:
        df = pd.read_csv(os.path.join(dir, file), header=[0], sep=';', low_memory=False)

    if len(df.columns) > 2:
        if config.raw_data_set_name == 'PV_noPV':
            train_samples, test_samples, terminals_already_in_dataset = extract_PV_noPV_data(df, combinations_already_in_dataset,
                                                                    number_of_samples_before)
        elif config.raw_data_set_name == 'malfunctions_in_LV_grid_dataset':
            train_samples, test_samples, combinations_already_in_dataset = extract_malfunction_data(df, combinations_already_in_dataset, number_of_samples_before, positive_test_samples_so_far, test_samples_so_far)
        else:
            train_samples, test_samples, combinations_already_in_dataset = extract_dummy_data(df, combinations_already_in_dataset,
                                                                                number_of_samples_before)
    else:
        train_samples = pd.DataFrame()
        test_samples = pd.DataFrame()

    return train_samples, test_samples, combinations_already_in_dataset