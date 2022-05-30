import os
import sys

#see experiment folder for all experiments (combinations of datasets, timeseries length and sample number)

#Deep Learning:
#chosen_experiment = 'dummy_1day_5k'

#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_200k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_20k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_10k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_5k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_7day_1k'

#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_200k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_20k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_10k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_5k'
#chosen_experiment = 'malfunctions_in_LV_grid_dataset_1day_1k'

#chosen_experiment = 'PV_noPV_7day_20k'
#chosen_experiment = 'PV_noPV_7day_10k'
#chosen_experiment = 'PV_noPV_7day_5k'
#chosen_experiment = 'PV_noPV_7day_1k'

#chosen_experiment = 'PV_noPV_1day_10k'
#chosen_experiment = 'PV_noPV_1day_5k'
#chosen_experiment = 'PV_noPV_1day_1k'

#Transformer Detection:
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_template'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_A_c_vs_w_real_world'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_A_c_vs_w_simulation'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_anomal_real_world'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_simulation'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_i_real_world'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_i_simulation'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_w_real_world'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_w_simulation'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_w_vs_i_real_world'
#chosen_experiment = 'malfunctions_in_LV_grid_detection_Setup_B_c_vs_w_vs_i_simulation'

#Disaggregation:
chosen_experiment = 'malfunctions_in_LV_grid_disaggregation_template'



experiments_folder = os.path.join(os.getcwd(), 'experiments')

try:
    if sys._called_from_test:
        test_folder = os.path.join(os.getcwd(), 'test')
        f = open(os.path.join(test_folder, "experiment.txt"), "r")
        experiment = f.read()
        f.close()
        os.remove(os.path.join(test_folder, "experiment.txt"))
        chosen_experiment = experiment
        experiments_folder = os.path.join(os.getcwd(), 'test')
except AttributeError:
    pass

experiment_path = os.path.join(experiments_folder, chosen_experiment) + '.py'
