"""
Author:
    David Fellner
Description:
    Set settings for QDS and elements and save results to file to create a dataset executing a QDS. At first the grid is
    prepared and scenario settings are set.
"""

import pflib.pf as pf
import config
from start_powerfactory import start_powerfactory
from grid_preparation import prepare_grid
from data_creation import create_data

import os

def main():

    for file in os.listdir(config.data_folder):
        if os.path.isdir(config.data_folder + file):

            print('Creating data using the grid %s' % file)
            app, study_case_obj, ldf, o_ElmNet = start_powerfactory(file)
            curves = prepare_grid(app, file, o_ElmNet)

            create_data(app, o_ElmNet, curves, study_case_obj, file)
            print('Done with grid %s' % file)

    print('Done with all grids')
    return 0


main() #see config file for settings





