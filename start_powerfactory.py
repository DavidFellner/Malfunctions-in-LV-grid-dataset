import pflib.pf as pf
import os
import importlib
from experiment_config import experiment_path, chosen_experiment
spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def set_load_flow_settings(ldf_com_obj, load_scaling, generation_scaling):

    ldf_com_obj.SetAttribute('scLoadFac', load_scaling)          # set load scaling factor
    ldf_com_obj.SetAttribute('scGenFac', generation_scaling)     # set generation scaling factor

    ldf_com_obj.SetAttribute('iopt_net', 0)  # AC Load Flow, balanced, positive sequence
    ldf_com_obj.SetAttribute('iopt_at', 0)  # Automatic tap adjustment of transformers
    ldf_com_obj.SetAttribute('iopt_pq', 0)  # Consider Voltage Dependency of Loads

    return

def start_powerfactory(file):


    pf.start(inMemoryInstance='a unique str')  # start pf in engine mode so as not make alterations last in the powerfactory file after the data generation
    app = pf.app

    try:
        pf.delete_project(file)  # delete if project already exists
    except NameError:
        pass

    pf.pfd_import(config.user, os.path.join(config.grid_data_folder, file) + '.pfd')              # freshly import project

    app.ActivateProject(file)
    project = app.GetActiveProject()

    study_case_obj = app.GetActiveStudyCase()                                   # Get active study case

    if not study_case_obj.SearchObject('*.ComLdf'):
        study_case_obj.CreateObject('ComLdf', 'load_flow_calculation')

    ldf = study_case_obj.SearchObject('*.ComLdf')                                  # Get load flow calculation object
    if config.deeplearning:
        set_load_flow_settings(ldf, config.load_scaling, config.generation_scaling)
    elif (config.detection_methods or config.detection_application) and (config.sim_setting == 'ERIGrid_phase_1' or config.pf_file in config.pf_file_dict.values()):
        set_load_flow_settings(ldf, 0.1, 100)   #set load flow settings in a  way to scale loads correctly
    else:
        set_load_flow_settings(ldf, 100, 100)  # Set default load flow settings

    o_IntPrjFolder_netdat = app.GetProjectFolder('netdat')
    o_ElmNet = o_IntPrjFolder_netdat.SearchObject('*.ElmNet')                   # Get network

    app.Hide()                                                                  # Hide GUI of powerfactory

    return app, study_case_obj, ldf, o_ElmNet