import pandas as pd
import os

#Sytem settings
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
data_folder = os.getcwd() + '\\input\\'
results_folder = os.getcwd() + '\\output\\'
local_machine_tz = 'Europe/Berlin'                          #timezone; it's important for Powerfactory

#Powerfactory settings
user = 'FellnerD'
system_language = 0                     #chose 0 for english, 1 for german according to the lagnuage of powerfactory installed on the system
parallel_computing = True
cores = 12                              #cores to be used for parallel computing (when 64 available use 12 - 24)
reduce_result_file_size = True          #save results as integers to save memory in csv
just_voltages = True                    # if False also P and Q results given

# Simulation settings
start = 0                               #start = 5 yields result_run#5
simruns = 2                             #number of datasets produced
step_size = 15                           #simulation step size in minutes
percentage = 25                         #percentage of busses with active PVs (PV proliferation)
control_curve_choice = 0                #for all PVs: choose control curve for 0 = cos(phi)(P), 1 = Q(P), 2 = brokenQ(P) (flat curve)
broken_control_curve_choice = 3         #for broken PV: choose control curve for 0 = cos(phi)(P), 1 = Q(P), 2 = broken Q(P) (flat curve), 3 = wrong Q(P) (inversed curve)
number_of_broken_devices = 1            #define number of devices to expereince malfunctions during simulation
load_scaling = 100                      #general load scaling for all loads in simulation (does not apply to setup)
generation_scaling = 100                #general generation scaling for all generation units in simulation (does not apply to setup)

# Simulation time settings
t_start = None                          #default(None): times inferred from profiles in data
t_end = None
#t_start = pd.Timestamp('2017-01-01 00:00:00', tz='utc')                                 # example for custom sim time
#t_end = pd.Timestamp('2018-01-01 00:00:00', tz='utc') - pd.Timedelta(step_size + 'T')