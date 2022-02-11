import os
import pandas as pd


def extract_data(test_bay, path, skip_existing=False):

    full_path = os.path.join(path, 'Test_Bay_' + test_bay, 'Extracted_Measurements')
    #measurements = {}
    count = 0

    for file in os.listdir(full_path):
        measurement_name = '_'.join(file.split(' ')[0:2]).split('.')[0]

        if skip_existing and measurement_name + '.csv' in os.listdir(full_path):
            print(f"File {measurement_name} already exists and is therefore skipped")
            continue

        count += 1
        f = open(os.path.join(full_path, file), 'r')
        text = f.read().replace('\x00', '')

        lines = text.split('\n')

        columns = lines[0].split('\t')

        measurement = pd.DataFrame(columns=columns)

        i = 1
        for row in lines[1:]:
            if row == '':
                continue
            else:
                measurement.loc[i] = row.split('\t')
                i += 1
        measurement.to_csv(os.path.join(full_path, measurement_name + '.csv'))
        # reread = pd.read_csv(os.path.join(full_path, measurement + '.csv'), sep =',') to make sure
        print(f"{file} finished and saved")


    return count


if __name__ == '__main__':

    data = {}
    data_path = os.path.join(os.getcwd(), 'ERIGrid-Test-Results-26-11-2021-phase1_final')
    test_bays = ['B1', 'F1', 'F2']

    for test_bay in test_bays:
        count = extract_data(test_bay, data_path, skip_existing=True)
        print(f"{count} files done for Test Bay {test_bay}")