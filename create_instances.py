import config
import pandas as pd
import numpy as np
import random

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
            df_noised = df[('ElmTerm', 'm:u')]
        else:
            df_noised = df
        return df_noised


def extract_malfunction_data(df, terminals_already_in_dataset, number_of_samples_before):
    '''

    :param df:
    :param terminals_already_in_dataset:
    :return:

    extracts data of interest (no duplicates) and labels it with 1 for malfunction present, and 0 with no malfunction present
    also adds noise to data
    '''

    metainfo = df[('metainfo', 'in the first', 'few indices')]
    #check if still correct
    #terminals_with_malfunctions = [i for i in metainfo.iloc[3].split("'") if 'Bus' in i]
    #terminals_with_devices = [i for i in metainfo.iloc[6].split("'") if 'Bus' in i]

    df_reduced = pd.DataFrame(index= df.index.append(pd.Index(['label'])))
    if len(terminals_already_in_dataset) > 0:
        for combination in terminals_already_in_dataset:
            if set(terminals_with_malfunctions) == set(combination):
                return df_reduced, terminals_already_in_dataset
    else:
        terminals_already_in_dataset.append(terminals_with_malfunctions)

    random.shuffle(terminals_with_devices)
    for term in terminals_with_devices:
        sample_number = int(len(df_reduced.columns) / len(df[term].columns))
        if term in terminals_with_malfunctions:
            label = 1       #means timeseries is of a terminal that experiences a malfunction
            noised_data = add_noise(df[term])
            if len(noised_data.columns) > 1:
                for i in noised_data.columns:
                    df_reduced[
                        (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                               i].values.tolist() + [
                                                                                               label]
            else:
                df_reduced[str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                    label]
        else:
            try:
                if int((df_reduced.iloc[-1] == 0).value_counts()[True] / len(noised_data.columns)) < int(1/config.share_of_positive_samples) - len(terminals_with_malfunctions):
                    label = 0
                    noised_data = add_noise(df[term])
                    if len(noised_data.columns) > 1:
                        for i in noised_data.columns:
                            df_reduced[
                                (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                                       i].values.tolist() + [
                                                                                                       label]
                    else:
                        df_reduced[
                            str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                            label]
                elif int((df_reduced.iloc[-1] == 1).value_counts()[True] / len(noised_data.columns)) == len(terminals_with_malfunctions):
                    return df_reduced, terminals_already_in_dataset
            except KeyError:
                if len(df_reduced.columns) < int(1/config.share_of_positive_samples) - len(terminals_with_malfunctions):
                    label = 0
                    noised_data = add_noise(df[term])
                    if len(noised_data.columns) > 1:
                        for i in noised_data.columns:
                            df_reduced[
                                (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                                       i].values.tolist() + [
                                                                                                       label]
                    else:
                        df_reduced[
                            str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                            label]
                else:
                    continue
            except IndexError:
                label = 0
                noised_data = add_noise(df[term])
                if len(noised_data.columns) > 1:
                    for i in noised_data.columns:
                        df_reduced[
                            (str(sample_number + number_of_samples_before), i[1])] = noised_data[i].values.tolist() + [
                            label]
                else:
                    df_reduced[str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                        label]

    return df_reduced, terminals_already_in_dataset

def extract_PV_noPV_data(df, terminals_already_in_dataset, number_of_samples_before):
    '''

    :param df:
    :param terminals_already_in_dataset:
    :return:

    extracts data of interest (no duplicates) and labels it with 1 for malfunction present, and 0 with no malfunction present
    also adds noise to data
    '''

    metainfo = df[('metainfo', 'in the first', 'few indices')]
    terminals_with_loads = [i for i in metainfo.iloc[4].split("'") if 'Bus' in i]
    terminals_with_PV = [i for i in metainfo.iloc[3].split("'") if 'Bus' in i]

    df_reduced = pd.DataFrame(index= df.index.append(pd.Index(['label'])))
    if len(terminals_already_in_dataset) > 0:
        for combination in terminals_already_in_dataset:
            if set(terminals_with_PV) == set(combination):
                return df_reduced, terminals_already_in_dataset
    else:
        terminals_already_in_dataset.append(terminals_with_PV)

    random.shuffle(terminals_with_PV)
    for term in terminals_with_loads:

        features_per_sample = len(df[term].columns)
        sample_number = int(len(df_reduced.columns) / features_per_sample)


        if term in terminals_with_PV:
            try:
                if int((df_reduced.iloc[-1] == 1).value_counts()[True] / features_per_sample) < config.positive_samples_per_simrun:
                    label = 1       #means timeseries is of a terminal that experiences a malfunction
                    noised_data = add_noise(df[term])
                    if isinstance(noised_data, pd.DataFrame) and len(noised_data.columns) > 1:
                        for i in noised_data.columns:
                            df_reduced[
                                (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                                       i].values.tolist() + [
                                                                                                       label]
                    else:
                        df_reduced[str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                            label]

                elif int((df_reduced.iloc[-1] == 0).value_counts()[True] / features_per_sample) == int(config.positive_samples_per_simrun * (1/config.share_of_positive_samples - 1)) and \
                        int((df_reduced.iloc[-1] == 0).value_counts()[True] / features_per_sample) == int(config.positive_samples_per_simrun * (1/config.share_of_positive_samples - 1)):
                    return df_reduced, terminals_already_in_dataset

            except KeyError:
                if len(df_reduced.columns) / features_per_sample < int(config.positive_samples_per_simrun * (1/config.share_of_positive_samples)):
                    label = 1
                    noised_data = add_noise(df[term])
                    if isinstance(noised_data, pd.DataFrame) and len(noised_data.columns) > 1:
                        for i in noised_data.columns:
                            df_reduced[
                                (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                                       i].values.tolist() + [
                                                                                                       label]
                    else:
                        df_reduced[
                            str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                            label]
                else:
                    continue

            except IndexError:
                label = 1
                noised_data = add_noise(df[term])
                if features_per_sample > 1:
                    for i in noised_data.columns:
                        df_reduced[
                            (str(sample_number + number_of_samples_before), i[1])] = noised_data[i].values.tolist() + [
                            label]
                else:
                    df_reduced[str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                        label]

        else:
            try:
                if int((df_reduced.iloc[-1] == 0).value_counts()[True] / features_per_sample) < int(config.positive_samples_per_simrun * (1/config.share_of_positive_samples - 1)):
                    label = 0
                    noised_data = add_noise(df[term])
                    if isinstance(noised_data, pd.DataFrame) and len(noised_data.columns) > 1:
                        for i in noised_data.columns:
                            df_reduced[
                                (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                                       i].values.tolist() + [
                                                                                                       label]
                    else:
                        df_reduced[
                            str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                            label]
                elif int((df_reduced.iloc[-1] == 1).value_counts()[True] / features_per_sample) == config.positive_samples_per_simrun:
                    return df_reduced, terminals_already_in_dataset

            except KeyError:
                if len(df_reduced.columns) / features_per_sample < int(config.positive_samples_per_simrun * (1/config.share_of_positive_samples - 1)):
                    label = 0
                    noised_data = add_noise(df[term])
                    if features_per_sample > 1:
                        for i in noised_data.columns:
                            df_reduced[
                                (str(sample_number + number_of_samples_before), i[1])] = noised_data[
                                                                                                       i].values.tolist() + [
                                                                                                       label]
                    else:
                        df_reduced[
                            str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                            label]
                else:
                    continue
            except IndexError:
                label = 0
                noised_data = add_noise(df[term])
                if features_per_sample > 1:
                    for i in noised_data.columns:
                        df_reduced[
                            (str(sample_number + number_of_samples_before), i[1])] = noised_data[i].values.tolist() + [
                            label]
                else:
                    df_reduced[str(sample_number + number_of_samples_before)] = noised_data.values.tolist() + [
                        label]

    return df_reduced, terminals_already_in_dataset


def create_samples(dir, file, terminals_already_in_dataset, number_of_samples_before):

    df = pd.read_csv(dir + '\\' + file, header = [0,1,2],sep=';', low_memory=False)
    if config.data_set_name == 'PV_noPV':
        df_treated, terminals_already_in_dataset = extract_PV_noPV_data(df, terminals_already_in_dataset,
                                                                number_of_samples_before)
    elif config.data_set_name == 'malfunctions_in_LV_grid_dataset':
        df_treated, terminals_already_in_dataset = extract_malfunction_data(df, terminals_already_in_dataset, number_of_samples_before)

    return df_treated, terminals_already_in_dataset