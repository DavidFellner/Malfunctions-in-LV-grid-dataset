import os

#see experiment folder for all experiments (combinations of datasets, timeseries length and sample number)

#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_10k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_5k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_1k'

chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_10k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_5k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_1k'

#chosen_experiment = 'PV_noPV_7day_10k'
#chosen_experiment = 'PV_noPV_7day_5k'
#chosen_experiment = 'PV_noPV_7day_1k'

#chosen_experiment = 'PV_noPV_1day_10k'
#chosen_experiment = 'PV_noPV_1day_5k'
#chosen_experiment = 'PV_noPV_1day_1k'

#Sytem settings
experiments_folder = os.getcwd() + '\\experiments\\'

experiment_path = experiments_folder + chosen_experiment + '.py'
