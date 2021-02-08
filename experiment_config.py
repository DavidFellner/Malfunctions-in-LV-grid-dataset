import os
import sys

#see experiment folder for all experiments (combinations of datasets, timeseries length and sample number)

#chosen_experiment = 'dummy_1day_5k'

#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_20k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_10k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_5k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_1k'

#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_10k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_5k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_1k'

chosen_experiment = 'PV_noPV_7day_20k'
#chosen_experiment = 'PV_noPV_7day_10k'
#chosen_experiment = 'PV_noPV_7day_5k'
#chosen_experiment = 'PV_noPV_7day_1k'

#chosen_experiment = 'PV_noPV_1day_10k'
#chosen_experiment = 'PV_noPV_1day_5k'
#chosen_experiment = 'PV_noPV_1day_1k'

experiments_folder = os.getcwd() + '\\experiments\\'

try:
    if sys._called_from_test:
        test_folder = os.getcwd() + '\\test\\'
        f = open(test_folder + "experiment.txt", "r")
        experiment = f.read()
        f.close()
        os.remove(test_folder + "experiment.txt")
        chosen_experiment = experiment
        experiments_folder = os.getcwd() + '\\test\\'
except AttributeError:
    pass

experiment_path = experiments_folder + chosen_experiment + '.py'
