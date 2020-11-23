import config
import pandas as pd
import numpy as np
import random

def add_noise(df):

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

    df_noised = df[df.columns[0]] + samples
    return df_noised


def extract_data(df, terminals_already_in_dataset, number_of_samples_before):
    '''

    :param df:
    :param terminals_already_in_dataset:
    :return:

    extracts data of interest (no duplicates) and labels it with 1 for malfunction present, and 0 with no malfunction present
    also adds noise to data
    '''

    metainfo = df[('metainfo', 'in the first', 'few indices')]
    terminals_with_malfunctions = [i for i in metainfo.iloc[3].split("'") if 'Bus' in i]
    terminals_with_devices = [i for i in metainfo.iloc[6].split("'") if 'Bus' in i]

    df_reduced = pd.DataFrame(index= df.index.append(pd.Index(['label'])))
    if len(terminals_already_in_dataset) > 0:
        for combination in terminals_already_in_dataset:
            if set(terminals_with_malfunctions) == set(combination):
                return df_reduced, terminals_already_in_dataset
    else:
        terminals_already_in_dataset.append(terminals_with_malfunctions)

    random.shuffle(terminals_with_devices)
    for term in terminals_with_devices:
        if term in terminals_with_malfunctions:
            label = 1       #means timeseries is of a terminal that experiences a malfunction
            noised_data = add_noise(df[term])
            df_reduced[str(len(df_reduced.columns) + number_of_samples_before)] = noised_data.values.tolist() + [label]
        else:
            try:
                if (df_reduced.iloc[-1] == 0).value_counts()[True] < int(1/config.share_of_malfunction_samples) - len(terminals_with_malfunctions):
                    label = 0
                    noised_data = add_noise(df[term])
                    df_reduced[
                        str(len(df_reduced.columns) + number_of_samples_before)] = noised_data.values.tolist() + [label]
                elif (df_reduced.iloc[-1] == 1).value_counts()[True] == len(terminals_with_malfunctions):
                    return df_reduced, terminals_already_in_dataset
            except KeyError:
                if len(df_reduced.columns) < int(1/config.share_of_malfunction_samples) - len(terminals_with_malfunctions):
                    label = 0
                    noised_data = add_noise(df[term])
                    df_reduced[
                        str(len(df_reduced.columns) + number_of_samples_before)] = noised_data.values.tolist() + [label]
                else:
                    continue
            except IndexError:
                label = 0
                noised_data = add_noise(df[term])
                df_reduced[str(len(df_reduced.columns) + number_of_samples_before)] = noised_data.values.tolist() + [
                    label]

    return df_reduced, terminals_already_in_dataset


def create_samples(dir, file, terminals_already_in_dataset, number_of_samples_before):

    df = pd.read_csv(dir + '\\' + file, header = [0,1,2],sep=';')
    df_treated, terminals_already_in_dataset = extract_data(df, terminals_already_in_dataset, number_of_samples_before)

    return df_treated, terminals_already_in_dataset