import os

#see expweriment folder for all experiment (combinations of datasets, timeseries length and sample number)
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_15k'
chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_10k'
#chosen_experiment = 'PV_noPV_7day_15k'
#chosen_experiment = 'PV_noPV_7day_10k'

#Sytem settings
experiments_folder = os.getcwd() + '\\experiments\\'

experiment_path = experiments_folder + chosen_experiment + '.py'
