import pflib.pf as pf
import pflib.object_frame as pof
import pandas as pd
import numpy as np
import random, math
import datetime
import os
import importlib
from experiment_config import experiment_path, chosen_experiment
from detection_method_settings import Mapping_Fluke_to_PowerFactory

m = Mapping_Fluke_to_PowerFactory()

from raw_data_generation.Sim.core.run_multiple_simulations import run_simulations

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config


def set_QDS_settings(app, study_case_obj, t_start, t_end, step_unit=1, balanced=0, step_size=None, marker=None):
    if step_size is None:
        step_size = config.step_size

    local_machine_tz = config.local_machine_tz

    if study_case_obj.SearchObject('*.ComStatsim') is None:
        qds = app.GetFromStudyCase("ComStatsim")
        qds.Execute()

    qds_com_obj = study_case_obj.SearchObject('*.ComStatsim')
    qds_com_obj.SetAttribute('iopt_net', balanced)  # AC Load Flow, balanced, positive sequence
    qds_com_obj.SetAttribute('calcPeriod', 4)  # User defined time range

    qds_com_obj.SetAttribute('startTime', int(
        (pd.DatetimeIndex([(t_start - t_start.tz_convert(local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
            0]))
    qds_com_obj.SetAttribute('endTime', int(
        (pd.DatetimeIndex([(t_end - t_end.tz_convert(local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
            0]))

    qds_com_obj.SetAttribute('stepSize', step_size)
    qds_com_obj.SetAttribute('stepUnit', step_unit)  # Seconds, Minutes ....

    if config.parallel_computing == True:
        qds_com_obj.SetAttribute('iEnableParal', config.parallel_computing)  # settings for parallel computation

        user = app.GetCurrentUser()
        settings = user.SearchObject(r'Set\Def\Settings')
        settings.SetAttribute('iActUseCore', 1)
        settings.SetAttribute('iCoreInput', config.cores)

    if config.just_voltages:  # define result variables of interest; result file is generated accordingly
        result_variables = {
            'ElmTerm': [
                'm:u',  # voltage at terminal
            ],
        }
    elif config.deeplearning:
        result_variables = {
            'ElmTerm': [
                'm:u',  # voltage at terminal
                'm:Pgen',  # active power generated at terminal
                'm:Qgen',  # reactive power generated at terminal
            ],
        }
    elif config.sim_setting in ['stmk']:
        result_variables = {
            'ElmTerm': [
                'm:u',
                'm:Pflow',
                'm:Qflow'
            ],
            'ElmPvsys': [
                'm:P:bus1',
                'm:Q:bus1'
            ]
        }
    elif marker == 'load_estimation_training_data':
        result_variables = {
            'ElmTerm': [
                'm:u',
                'm:phiu',
                'm:Pflow',
                'm:Qflow'
            ],
            'ElmLod': [
                'm:P:bus1',
                'm:Q:bus1'
            ],
            'ElmPvsys': [
                'm:P:bus1',
                'm:Q:bus1'
            ]
        }
    else:
        result_variables = m.mapping()

    result = pf.app.GetFromStudyCase('ComStatsim').results  # should be language independent
    """if config.system_language == 0:
        #result = pf._resolve_result_object('Quasi-Dynamic Simulation AC')  # get result file
        result = pf.app.GetFromStudyCase('ComStatsim').presults #should be language independent
    else:
        if config.system_language == 1:
            result = pf.app.GetFromStudyCase('ComStatsim').presults #should be language independent
            #result = pf._resolve_result_object('Quasi-Dynamische Simulation AC')  # get result file"""

    pf.set_vars_of_result_obj(result,
                              result_variables=result_variables)

    return result


def create_malfunctioning_PVs(malfunctioning_devices, o_ElmNet, curves):
    terms_with_malfunction = []

    for i in malfunctioning_devices:
        cubicle = i.bus1
        connection = cubicle.cterm
        terms_with_malfunction.append(connection)
        name = i.loc_name

        o_ElmTerm = connection
        o_StaCubic = o_ElmTerm.CreateObject('StaCubic', i.bus1.loc_name + ' PVbroken')
        o_Elm = o_ElmNet.CreateObject('ElmGenstat', name + ' broken')
        o_Elm.SetAttribute('bus1', o_StaCubic)

        o_Elm.SetAttribute('av_mode', 'qpchar')  # Control activated
        o_Elm.SetAttribute('pQPcurve', curves[config.broken_control_curve_choice])  # broken Control activated

        o_ChaTime = pf.get_referenced_characteristics(i, 'pgini')[
            0]  # get characteristic from original PV
        pf.set_referenced_characteristics(o_Elm, 'pgini', o_ChaTime)  # set saecharacteristic for inserted PV

        o_Elm.pgini = i.pgini  # scale PV  output to the same
        o_Elm.sgn = i.sgn  # set apparent power as above

        o_Elm.outserv = 0  # initially in service,
        # if needed: deactivated right at start of simulation by event > otherwise not considered by simulation at all

    return terms_with_malfunction


def create_malfunctioning_EVCs(malfunctioning_devices):
    terms_with_malfunction = []

    for i in malfunctioning_devices:
        cubicle = i.bus1
        connection = cubicle.cterm
        terms_with_malfunction.append(connection)

    return terms_with_malfunction


def create_malfunctioning_devices(active_PVs, active_EVCs, o_ElmNet, curves):
    '''
    PV: Pick random active PV to have dysfunctional control;
        Therefore another equal PV is created with the malfunctioning control
    EV: just pick a EVCS, as individual ldfs are necessary anyways we only need to pick one that is then controlled improperly
    '''

    number_of_broken_devices = config.number_of_broken_devices_and_type[0]
    type_of_broken_devices = config.number_of_broken_devices_and_type[1]

    if type_of_broken_devices == 'PV':
        malfunctioning_devices = random.sample(active_PVs, number_of_broken_devices)
        terms_with_malfunction = create_malfunctioning_PVs(malfunctioning_devices, o_ElmNet, curves)
    elif type_of_broken_devices == 'EV':
        malfunctioning_devices = random.sample(active_EVCs, number_of_broken_devices)
        terms_with_malfunction = create_malfunctioning_EVCs(malfunctioning_devices)
    else:
        print('Undefined device category entered for malfunction!')
        malfunctioning_devices = []
        terms_with_malfunction = []

    return malfunctioning_devices, terms_with_malfunction


def set_times(file, element=None, chars_dict=None, scenario=None, control=None, phase=None):
    if not config.t_start and not config.t_end:  # default > simulation time inferred from available load/generation profile data
        if config.deeplearning:
            t_start = pd.Timestamp(
                pd.read_csv(os.path.join(config.grid_data_folder, file, 'LoadProfile.csv'), sep=';',
                            index_col='time').index[0], tz='utc')
            t_end = pd.Timestamp(
                pd.read_csv(os.path.join(config.grid_data_folder, file, 'LoadProfile.csv'), sep=';',
                            index_col='time').index[-1], tz='utc')

            if config.dev_mode:
                sim_length = 8 * 900
                t_end = t_start + pd.Timedelta(str(sim_length) + 's')

            if config.sim_length < 365:  # randomly choose period of defined length during maximum simulation period to simulate
                time_delta = int((t_end - t_start) / np.timedelta64(1, 's'))  # simulation duration converted to seconds
                seconds = random.sample([*range(0, time_delta, 1)], 1)[
                    0]  # pick random point of the maximum simulation period
                t_off = pd.Timedelta(str(seconds) + 's')  # set offset

                sim_start = (t_start + t_off).replace(hour=0, minute=0, second=0)

                if config.dev_mode:
                    sim_length = 2 * 900
                    sim_end = sim_start + pd.Timedelta(str(sim_length) + 's')
                else:
                    t_sim_length = pd.Timedelta(str(config.sim_length) + 'd')
                    sim_end = sim_start + t_sim_length

                if sim_end < t_end:
                    t_start = sim_start
                    t_end = sim_end
                    return t_start, t_end

        if config.detection_methods:
            """char = pf.get_referenced_characteristics(element, 'pgini')[0]
            t_start = char.scale.scale[0]
            t_end = char.scale.scale[-1]"""
            if config.sim_setting == 'ERIGrid_phase_2':
                t_start = chars_dict[element][f'{scenario}_{control}_t_start']
                t_end = chars_dict[element][f'{scenario}_{control}_t_end']
            elif config.sim_setting in ['stmk']:
                t_start = chars_dict[element][f'{scenario}_t_start']
                t_end = chars_dict[element][f'{scenario}_t_end']
            else:
                t_start = chars_dict[element][f't_{scenario}_start']
                t_end = chars_dict[element][f't_{scenario}_end']

        if config.detection_application:
            if scenario:
                if phase == 'phase2':
                    t_start = chars_dict[element][f'{scenario}_{control}_t_start']
                    t_end = chars_dict[element][f'{scenario}_{control}_t_end']
                else:
                    t_start = chars_dict[element][f't_{scenario}_start']
                    t_end = chars_dict[element][f't_{scenario}_end']
            else:
                t_start = chars_dict[element][f't_start']
                t_end = chars_dict[element][f't_end']

    else:
        t_start = config.t_start
        t_end = config.t_end

    return t_start, t_end


def create_malfunction_events(app, malfunctioning_devices, t_start, t_end):
    '''
    :param malfunctioning_devices:
    :return:
    event to change control at random point of time (correctly controlled object turned off, malfunctioning object turned on)
    '''

    time_delta = int((t_end - t_start) / np.timedelta64(1, 's'))  # simulation duration converted to seconds
    seconds = random.sample([*range(0, time_delta, 1)], 1)[0]  # pick random point of the simulation
    t_off = pd.Timedelta(str(seconds) + 's')  # set offset
    event_time = (pd.DatetimeIndex([(t_start + t_off)]).astype(np.int64) // 10 ** 9)[0]  # define event time in unix
    start_time = \
        (pd.DatetimeIndex([(t_start - t_start.tz_convert(config.local_machine_tz).utcoffset())]).astype(
            np.int64) // 10 ** 9)[
            0] + 1  # define start time in unix

    time_of_malfunction = str(pd.to_datetime(event_time, unit='s', utc=True).astimezone(datetime.datetime.now(
        datetime.timezone.utc).astimezone().tzinfo))  # PF converts event time to local time zone and executes the event accordingly
    # > if passed 8 as event time, event time is converted to 9 in winter and to 10 in summer

    evtFold = app.GetFromStudyCase('IntEvtqds')

    for i in malfunctioning_devices:
        if config.whole_year == True:
            event_time = start_time
            oEvent = evtFold.CreateObject('EvtOutage', i.loc_name + ' PVout')
            oEvent.tDateTime = int(event_time)
            oEvent.p_target = i
            oEvent.i_what = 0

        else:
            oEvent = evtFold.CreateObject('EvtOutage', i.loc_name + ' PVout')
            oEvent.tDateTime = int(event_time)
            oEvent.p_target = i
            oEvent.i_what = 0

            oEvent = evtFold.CreateObject('EvtOutage',
                                          i.loc_name + ' PVbrokenout')  # at first PV has to be active and taken out of service;
            oEvent.tDateTime = int(start_time)  # otherwise it is not considered by simulation at all
            oEvent.p_target = app.GetCalcRelevantObjects(i.loc_name + ' broken.ElmGenstat')[0]
            oEvent.i_what = 0

            oEvent = evtFold.CreateObject('EvtOutage', i.loc_name + ' PVbrokenin')
            oEvent.tDateTime = int(event_time)
            oEvent.p_target = app.GetCalcRelevantObjects(i.loc_name + ' broken.ElmGenstat')[0]
            oEvent.i_what = 1

    return time_of_malfunction


def run_QDS(app, run, result):
    '''
    
    :param result: 
    :return: 
    
    Run quasi dynamic simulation to produce the data wanted.
    '''

    qds = app.GetFromStudyCase("ComStatsim")
    print('Simulation started')
    qds.Execute()
    print('Simulation run number %d concluded and is being saved' % run)

    results = pf.get_results_as_data_frame(result)
    results.name = 'result_run#%d' % run

    list_of_power_variables = ['m:Pgen', 'm:Qgen', 'm:Pload', 'm:Qload', 'm:P:bus2', 'm:Q:bus2', 'm:Psum:bushv',
                               'm:Qsum:bushv', 'm:Psum:buslv', 'm:Qsum:buslv', 'm:Pflow', 'm:Qflow']
    for data in results.columns:
        if data[2] in list_of_power_variables:
            results[data] = results[data] * 1000 * 1000  # convert all powers from MW to Watts
            if config.reduce_result_file_size == True:
                results[data] = results[data].values.astype('int')

    return results


def pick_results(results, training_data=False, phase=None):
    test_bay_dfs = {}

    if config.detection_application:
        if phase=='phase1':
            map = {'B1_elements': {'lines': ['LV-028', 'LV-027'], 'term': 'Test Bay B1'},
                   'F1_elements': {'lines': ['LV-004', 'LV-005'], 'term': 'Test Bay F1'},
                   'F2_elements': {'lines': ['LV-001', 'LV-002'], 'term': 'Test Bay F2'}}
        else:
            map = {'B2_elements': {'lines': ['LV-024', 'LV-025'], 'term': 'Test Bay B2'},
                   'B1_elements': {'lines': ['LV-027', 'LV-028'], 'term': 'Test Bay B1'},
                   'A1_elements': {'lines': ['LV-033', 'LV-032'], 'term': 'Test Bay A1'},
                   'C1_elements': {'lines': ['LV-038'], 'term': 'Test Bay C1'}}
    else:
        if config.sim_setting == 'ERIGrid_phase_1':
            map = {'B1_elements': {'lines': ['LV-028', 'LV-027'], 'term': 'Test Bay B1'},
                   'F1_elements': {'lines': ['LV-004', 'LV-005'], 'term': 'Test Bay F1'},
                   'F2_elements': {'lines': ['LV-001', 'LV-002'], 'term': 'Test Bay F2'}}
        elif config.sim_setting == 'ERIGrid_phase_2':
            map = {'B2_elements': {'lines': ['LV-024', 'LV-025'], 'term': 'Test Bay B2'},
                   'B1_elements': {'lines': ['LV-027', 'LV-028'], 'term': 'Test Bay B1'},
                   'A1_elements': {'lines': ['LV-033', 'LV-032'], 'term': 'Test Bay A1'},
                   'C1_elements': {'lines': ['LV-038'], 'term': 'Test Bay C1'}}
        elif config.sim_setting in ['stmk']:
            if learning_config['setup_chosen']['stmk'] == 'Gleinz':
                map = {'NAP_elements': {'term': 'NAP_Gleinz_PV'}}
            else:
                map = {'NAP_elements': {'term': 'NAP_Neudau_PV'}}
        else:
            print('undefined sim setup chosen!')

    if config.sim_setting in ['stmk']:
        if not training_data:
            for test_bay in map:
                test_bay_df = pd.DataFrame()
                for element in map[test_bay]:
                    if element == 'lines':
                        # do calc and picking on line data
                        for var in m.NAPmap['ElmLne']:
                            try:
                                if test_bay.split('_')[0] == 'F2':
                                    test_bay_df[m.NAPmap['ElmLne'][var][1]] = results[
                                        (
                                            map[test_bay][element][0], 'ElmLne',
                                            var)]  # always upstream of terminal value used
                                else:
                                    test_bay_df[m.NAPmap['ElmLne'][var][0]] = results[
                                        (
                                            map[test_bay][element][0], 'ElmLne',
                                            var)]  # always upstream of terminal value used
                            except KeyError:
                                print(f"{(map[test_bay][element][0], 'ElmLne', var)} not defined")
                    if element == 'term':
                        for var in m.NAPmap['ElmTerm']:
                            try:
                                if test_bay.split('_')[0] == 'F2':
                                    test_bay_df[m.NAPmap['ElmTerm'][var][1]] = results[
                                        (map[test_bay][element], 'ElmTerm', var)]
                                else:
                                    test_bay_df[m.NAPmap['ElmTerm'][var][0]] = results[
                                        (map[test_bay][element], 'ElmTerm', var)]
                            except KeyError:
                                print(f"{(map[test_bay][element][0], 'ElmTerm', var)} not defined")
                test_bay_dfs[test_bay] = test_bay_df
        if training_data:
            test_bays_df = results
            """test_bays_df = pd.concat(test_bay_dfs,
                                     axis=1)  # merge all selected data into one DF for training data for load estimation NN
            test_bays_df.columns = [' '.join(col).strip() for col in test_bays_df.columns.values]"""
    else:
        if not training_data:
            for test_bay in map:
                test_bay_df = pd.DataFrame()
                for element in map[test_bay]:
                    if element == 'lines':
                        # do calc and picking on line data
                        for var in m.map['ElmLne']:
                            try:
                                if test_bay.split('_')[0] == 'F2':
                                    test_bay_df[m.map['ElmLne'][var][1]] = results[
                                        (
                                            map[test_bay][element][0], 'ElmLne',
                                            var)]  # always upstream of terminal value used
                                else:
                                    test_bay_df[m.map['ElmLne'][var][0]] = results[
                                        (
                                            map[test_bay][element][0], 'ElmLne',
                                            var)]  # always upstream of terminal value used
                            except KeyError:
                                print(f"{(map[test_bay][element][0], 'ElmLne', var)} not defined")
                    if element == 'term':
                        for var in m.map['ElmTerm']:
                            try:
                                if test_bay.split('_')[0] == 'F2':
                                    test_bay_df[m.map['ElmTerm'][var][1]] = results[
                                        (map[test_bay][element], 'ElmTerm', var)]
                                else:
                                    test_bay_df[m.map['ElmTerm'][var][0]] = results[
                                        (map[test_bay][element], 'ElmTerm', var)]
                            except KeyError:
                                print(f"{(map[test_bay][element][0], 'ElmTerm', var)} not defined")
                test_bay_dfs[test_bay] = test_bay_df
        if training_data:
            test_bays_df = results
            """test_bays_df = pd.concat(test_bay_dfs,
                                     axis=1)  # merge all selected data into one DF for training data for load estimation NN
            test_bays_df.columns = [' '.join(col).strip() for col in test_bays_df.columns.values]"""
            return test_bays_df

    return test_bay_dfs


def save_results(count, results, file, t_start, t_end, malfunctioning_devices=None, time_of_malfunction=None,
                 terminals_with_PVs=None, terminals_with_EVCs=None, terminals=None, control_curve=None, marker='',
                 setup=None, folder=None, phase=None):
    if config.deeplearning:
        if malfunctioning_devices:
            if config.type == 'PV':
                malfunction_type = {0: 'cos(phi)(P)', 1: 'Q(P)', 2: 'broken Q(P) (flat curve)',
                                    3: 'wrong Q(P) (inversed curve)'}
                malfunction = malfunction_type[config.broken_control_curve_choice]
            elif config.type == 'EV':
                malfunction = 'EVSE without P(U) control'
            else:
                print("Undefined type of malfunction chosen")

            try:
                terminals_with_malfunction = [i.bus1.cterm.loc_name for i in malfunctioning_devices]
            except AttributeError:
                terminals_with_malfunction = [i.loc_name.split('@ ')[1] for i in malfunctioning_devices]

        metainfo = ['simulation#%d' % count, 'comment data format: active and reactive powers in Watts',
                    'step time in minutes: %d' % config.step_size, 'start time of simulation: %s' % t_start,
                    'end time of simulation: %s' % t_end]

        if malfunctioning_devices:
            metainfo.append(['terminal(s) with malfunction: %s' % terminals_with_malfunction,
                             'type of malfunction: %s' % malfunction])
        if terminals_with_PVs:
            metainfo.append('terminals with PVs: %s' % terminals_with_PVs)
        if terminals_with_EVCs:
            metainfo.append('terminals with EVSE: %s' % terminals_with_EVCs)
        if terminals:
            metainfo.append('terminals with loads: %s' % terminals)
        if time_of_malfunction:
            metainfo.append('time of malfunction: %s' % time_of_malfunction)

        metainfo += [''] * (len(results[0]) - len(metainfo))
        for i in list(range(len(results))):
            results[i][('metainfo', 'in the first', 'few indices')] = metainfo

        results_folder, file_folder = create_results_folder(file)
        if config.just_voltages:
            results = results[0].dropna(axis=1)
        results.to_csv(os.path.join(file_folder, 'result_run#%d.csv' % count), header=True, sep=';', decimal='.',
                       float_format='%.' + '%sf' % config.float_decimal)
    if config.detection_methods:
        results_folder, file_folder = create_results_folder(file)

        test_bay_dfs = pick_results(results)

        if config.sim_setting == 'ERIGrid_phase_1':
            test_bays = {'Test_Bay_B1': 'B1dataframe', 'Test_Bay_F1': 'F1dataframe', 'Test_Bay_F2': 'F2dataframe'}
        elif config.sim_setting in ['stmk']:
            test_bays = {'NAP': 'NAPdataframe'}
        else:
            test_bays = {'Test_Bay_B2': 'B2dataframe', 'Test_Bay_B1': 'B1dataframe', 'Test_Bay_A1': 'A1dataframe', 'Test_Bay_C1': 'C1dataframe'}
        for test_bay in test_bays:
            bay_folder = os.path.join(file_folder, test_bay)
            if not os.path.isdir(bay_folder):
                os.mkdir(bay_folder)
            if config.sim_setting in ['stmk']:
                test_bay_dfs['NAP_elements'].to_csv(os.path.join(bay_folder,
                                                                                       f'scenario_{count}_{control_curve}_control_Setup.csv'),
                                                                          sep=',', decimal=',')
    if config.detection_application:

        if marker[:13] == 'training_data' or marker[:21] == 'estimation_input_data':
            test_bays_df = pick_results(results, training_data=True, phase=phase)
            if marker.split('_')[0] == 'training':
                test_bays_df.to_csv(os.path.join(folder,
                                                 f'load_estimation_training_data_setup_{marker.split("_")[-1]}.csv'))  # ,
                # sep=',', decimal=',')
            else:
                test_bays_df.to_csv(os.path.join(folder,
                                                 f'load_estimation_input_data_setup_{marker.split("_")[-1]}.csv'))  # ,
                # sep=',', decimal=',')
            return
        else:
            results_folder, file_folder = create_results_folder(file, phase=phase)

            test_bay_dfs = pick_results(results, phase=phase.split('_')[-1])

            if phase.split('_')[-1] == 'phase1':
                test_bays = {'Test_Bay_B1': 'B1dataframe', 'Test_Bay_F1': 'F1dataframe', 'Test_Bay_F2': 'F2dataframe'}
            else:
                test_bays = {'Test_Bay_B2': 'B2dataframe', 'Test_Bay_B1': 'B1dataframe', 'Test_Bay_A1': 'A1dataframe',
                             'Test_Bay_C1': 'C1dataframe'}
            for test_bay in test_bays:
                bay_folder = os.path.join(file_folder, test_bay)
                if not os.path.isdir(bay_folder):
                    os.mkdir(bay_folder)
                test_bay_dfs[test_bay.split('_')[2] + '_elements'].to_csv(os.path.join(bay_folder,
                                                                                       f'scenario_{count}_{control_curve}_control_Setup_{setup}_{marker}.csv'),
                                                                          sep=',', decimal=',')

    return


def clean_up(app, active_PVs, active_EVCS=None, malfunctioning_devices=None):
    '''

    :param active_Pvs:
    :return:

    reset everything that has been changed between simulation runs
    '''

    for o in active_PVs: o.outserv = 1  # reset PVs to be out of service
    if config.number_of_broken_devices_and_type[1] == 'PV':
        if malfunctioning_devices:
            for o in malfunctioning_devices:  # delete PVs (+cubicles) created for malfunctions
                inserted_dummy_device = app.GetCalcRelevantObjects(o.loc_name + ' broken' + '.ElmGenstat')[0]
                o_StaCubic = inserted_dummy_device.bus1
                inserted_dummy_device.Delete()
                o_StaCubic.Delete()

    if active_EVCS is not None:
        for o in active_EVCS: o.outserv = 1  # reset PVs to be out of service

    evtFold = app.GetFromStudyCase('IntEvtqds')
    for o in evtFold.GetContents():  # delete events set up
        o.Delete()

    return


def create_results_folder(file, phase=None, setup=None, estimate_input=False, training_data=False):
    if config.deeplearning:
        penetrations = ''
        for key, value in config.percentage.items():
            if value > 0:
                penetrations += '_' + key + '(' + str(value) + ')'

        results_folder = os.path.join(config.raw_data_folder,
                                      (config.raw_data_set_name + penetrations + '_raw_data'))
    elif config.detection_methods:
        results_folder = os.path.join(config.raw_data_folder,
                                      (config.sim_setting + '_sim_data'))
        file_folder = os.path.join(results_folder, file)
    elif config.detection_application:
        if estimate_input:
            results_folder = os.path.join(config.raw_data_folder, (
                    '_'.join(config.pf_file_dict[phase.split('_')[-1]].split('_')[1:]) + '_estimation_input_data'))
            file_folder = os.path.join(results_folder, f'Setup_{setup}')
        elif training_data:
            results_folder = os.path.join(config.raw_data_folder, (
                    '_'.join(config.pf_file_dict[phase.split('_')[-1]].split('_')[1:]) + '_training_data'))
            file_folder = os.path.join(results_folder, file)
        else:
            if phase.split('_')[-1] == 'phase1': phase_num = 'phase_1'
            else: phase_num = 'phase_2'
            results_folder = os.path.join(config.raw_data_folder, (
                    '_'.join([config.pf_file_dict[phase.split('_')[-1]].split('_')[1:][0], phase_num]) + '_sim_data'))
            file_folder = os.path.join(results_folder, file)

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    if not os.path.isdir(file_folder):
        os.mkdir(file_folder)

    return results_folder, file_folder


def create_deeplearning_data(app, o_ElmNet, grid_data, study_case_obj, file):
    '''

    begin of loop to vary malfunctioning device and point of malfunction between simulation runs

    '''

    curves = grid_data[0]
    EV_charging_stations = grid_data[1]
    loads = grid_data[2]

    if config.add_data:
        results_folder, file_folder = create_results_folder(file)
        count = len([name for name in os.listdir(file_folder) if os.path.isfile(os.path.join(file_folder, name))])
    else:
        count = 0

    list_of_PVs = [i for i in app.GetCalcRelevantObjects('*.ElmGenstat') if
                   i.loc_name.split(' ')[1] == 'SGen']  # get list of PVs

    sample_PVs = math.floor(len(list_of_PVs) * config.percentage['PV'] / 100)  # set % of terminals to have PV
    sample_EVCs = math.floor(
        len(EV_charging_stations) * config.percentage['EV'] / 100)  # set % of terminals to have EV charging stations

    while count < config.simruns:  # every simrun has a different malfunction location and time (& different PV, EVCs... locations in general)

        active_PVs = random.sample(list_of_PVs, sample_PVs)  # pick active PVs randomly
        active_EVCS = random.sample(EV_charging_stations, sample_EVCs)  # pick active PVs randomly
        current_grid_info = [active_PVs, active_EVCS]

        for o in active_PVs: o.outserv = 0  # PVs not outofservice and therefore active = installed PV
        terminals_with_PVs = list(set([i.bus1.cterm.loc_name for i in
                                       active_PVs]))  # set bc there can be 2 load on one terminal and therefore 2 PVs on one terminal
        terminals = list(set([i.bus1.cterm.loc_name for i in list_of_PVs]))

        for o in active_EVCS: o.outserv = 0  # EVCs not outofservice and therefore active = installed EVCs
        terminals_with_EVCs = list(set([i.bus1.cterm.loc_name for i in
                                        active_EVCS]))  # set bc there can be 2 load on one terminal and therefore 2 EVCs on one terminal

        t_start, t_end = set_times(file)
        current_grid_info.append(t_start)
        current_grid_info.append(t_end)
        if config.raw_data_set_name == 'malfunctions_in_LV_grid_dataset':
            malfunctioning_devices, terms_with_malfunction = create_malfunctioning_devices(active_PVs, active_EVCS,
                                                                                           o_ElmNet,
                                                                                           curves)  # only 1 malfunction looked at? also only 1 type of malfunction?
            current_grid_info.append(malfunctioning_devices)
            print(malfunctioning_devices[0].loc_name)
            # time_of_malfunction = create_malfunction_events(app, malfunctioning_devices, t_start, t_end)

        if config.number_of_broken_devices_and_type[
            1] == 'PV':  # only here QDS can be done, otherwise malfunction has to be produced in individual loadflows
            result = set_QDS_settings(app, study_case_obj, t_start, t_end)
            results = run_QDS(app, count, result)
        else:
            results = run_simulations(file, current_grid_info)

        if config.raw_data_set_name == 'malfunctions_in_LV_grid_dataset':
            save_results(count, results, file, t_start, t_end, malfunctioning_devices=malfunctioning_devices,
                         terminals_with_PVs=terminals_with_PVs, terminals_with_EVCs=terminals_with_EVCs)
            clean_up(app, active_PVs, active_EVCS, malfunctioning_devices=malfunctioning_devices)
        elif config.raw_data_set_name == 'PV_noPV':
            save_results(count, results, file, t_start, t_end, terminals_with_PVs=terminals_with_PVs,
                         terminals=terminals)
            clean_up(app, active_PVs, active_EVCS)

        count += 1

    return


def create_detectionmethods_data(app, o_ElmNet, grid_data, study_case_obj, file, setup=None, estimation=None,
                                 phase=None):
    '''

    create all data that was collected in the sim setup defined

    '''

    chars_dict = grid_data[0]
    if config.sim_setting not in ['stmk'] and (config.sim_setting == 'ERIGrid_phase_2' or phase.split('_')[-1] == 'phase2'):
        control_curves = {'DSM' : [], 'noDSM' : []}
        if config.detection_application: del control_curves['DSM']
    else:
        if config.sim_setting not in ['stmk'] and (estimation is not None or learning_config['setup_chosen'].split('_')[1] == 'A'):
            if config.extended:
                control_curves = {'correct': [1, 3, 0.9, 6], 'wrong': [1, 3, 0.999999, 6],
                                  'inversed': [0.9, 6, 0.999999, 3]}
            else:
                control_curves = {'correct': [1, 3, 0.9, 6], 'wrong': [1, 3, 0.999999, 6], }
        if config.sim_setting in ['stmk']:
            control_curves = config.setups[learning_config['setup_chosen']['stmk']]

        else:
            control_curves = {'correct': [1, 3, 0.9, 6], 'wrong': [1, 3, 0.999999, 6],
                              'inversed': [0.9, 6, 0.999999, 3]}

        if config.detection_application: del control_curves['correct']

    if config.sim_setting in ['stmk']:
        # ADD CHARS TO BASE VARIANT; ACTIVATE GIVEN VARIANT; PERFORM QDS
        PV = app.GetCalcRelevantObjects('*.ElmPvsys')[0]

        days = [i.split('_')[1] for i in chars_dict[PV].keys()][0::3]
        #days = days[:] #if not all scens to be simulated (pick up at some point)
        for day in days:
            for element in chars_dict.keys():

                if element in app.GetCalcRelevantObjects('.ElmPvsys'):
                    pf.set_referenced_characteristics(element, 'pgini', chars_dict[element][f'p_{day}'])
                elif element in app.GetCalcRelevantObjects('.ElmXnet'):
                    pf.set_referenced_characteristics(element, 'usetp', chars_dict[element][f'v_{day}'])
                elif element in app.GetCalcRelevantObjects('.ElmGenstat'):
                    pf.set_referenced_characteristics(element, 'pgini', chars_dict[element][f'p_{day}'])
                    pf.set_referenced_characteristics(element, 'qgini', chars_dict[element][f'q_{day}'])

            # do calc for correct, wrong, inversed, flat , as_is as in variants
            for control_curve in control_curves:
                pf.activate_variations(control_curve)


                t_start, t_end = set_times(file, element=PV, chars_dict=chars_dict, scenario=day,
                                           control=control_curve)

                result = set_QDS_settings(app, study_case_obj, t_start, t_end, step_unit=config.step_unit,
                                          balanced=config.balanced)  # set which vars and where!
                results = run_QDS(app, int(day), result)
                pf.deactivate_variations(control_curve)

                # save correctly as to be used by rest of framework
                if config.detection_methods:
                    save_results(int(day), results, file, t_start, t_end, control_curve=control_curve)


    else:
        if config.add_data:
            if config.sim_setting == 'ERIGrid_phase_2' or phase.split('_')[-1] == 'phase2':
                new_chars_dict = chars_dict
                results_folder, file_folder = create_results_folder(file, phase=phase)
                bay_folder = os.path.join(file_folder, 'Test_Bay_B2')
                if not os.path.isdir(bay_folder):
                    os.mkdir(bay_folder)

            else:
                new_chars_dict = {}
                results_folder, file_folder = create_results_folder(file, phase=phase)
                for element in chars_dict:
                    new_chars_dict[element] = {}
                    for char in chars_dict[element]:
                        bay_folder = os.path.join(file_folder, 'Test_Bay_F2')
                        if not os.path.isdir(bay_folder):
                            os.mkdir(bay_folder)

                        for control_curve in control_curves:
                            if config.detection_application:
                                if not os.path.isfile(os.path.join(bay_folder,
                                                                   f'scenario_{str(int(char.split("_")[1]) + 1)}_{control_curve}_control_Setup_{setup}_{estimation}.csv')) \
                                        and not os.path.isfile(os.path.join(bay_folder,
                                                                            f'scenario_{str(int(char.split("_")[1]) + 1)}_{control_curve}_control_Setup_{setup}_{estimation}.csv')):
                                    new_chars_dict[element][char] = chars_dict[element][char]
                            else:
                                name_1 = f'scenario_{char.split("_")[1]}_{control_curve}_control_Setup_{learning_config["setup_chosen"].split("_")[1]}.csv'
                                name_2 = f'scenario_{char.split("_")[0]}_{control_curve}_control_Setup_{learning_config["setup_chosen"].split("_")[1]}.csv'

                                if not os.path.isfile(os.path.join(bay_folder, name_1)) \
                                        and not os.path.isfile(os.path.join(bay_folder,
                                                                            name_2)):
                                    new_chars_dict[element][char] = chars_dict[element][char]
                                # del new_chars_dict[element][char]
            chars_dict = new_chars_dict

        PV = app.GetCalcRelevantObjects('*.ElmPvsys')[0]
        if len(app.GetCalcRelevantObjects('*.ElmPvsys')) > 1: print('too many PVs!')

        scenarios = [i.split('_')[1] for i in chars_dict[PV].keys()][0::3]
        #rerun only as_is
        scenarios = [i.split('_')[1] for i in chars_dict[PV].keys()][0::4]
        for scenario in scenarios:
            for element in chars_dict.keys():

                if element in app.GetCalcRelevantObjects('.ElmLod'):
                    if (config.sim_setting == 'ERIGrid_phase_2' or phase.split('_')[-1] == 'phase2') and element.loc_name != 'LB 3':
                        pf.set_referenced_characteristics(element, 'plini', chars_dict[element][f'p_{scenario}_noDSM'])
                        pf.set_referenced_characteristics(element, 'qlini', chars_dict[element][f'q_{scenario}_noDSM'])
                    else:
                        pf.set_referenced_characteristics(element, 'plini', chars_dict[element][f'p_{scenario}'])
                        pf.set_referenced_characteristics(element, 'qlini', chars_dict[element][f'q_{scenario}'])
                    load = element
                elif element in app.GetCalcRelevantObjects('.ElmPvsys'):
                    pf.set_referenced_characteristics(element, 'pgini', chars_dict[element][f'p_{scenario}'])

            # do calc for correct, wrong, inversed

            for control_curve in control_curves:
                do = True
                if config.add_data:

                    if (config.sim_setting == 'ERIGrid_phase_2' or phase.split('_')[-1] == 'phase2'): bay_folder = os.path.join(file_folder, 'Test_Bay_B2')
                    else: bay_folder = os.path.join(file_folder, 'Test_Bay_F2')

                    if config.detection_application:
                        if os.path.isfile(os.path.join(bay_folder,
                                                       f'scenario_{scenario}_{control_curve}_control_Setup_{setup}_{estimation}.csv')):
                            do = False
                    else:
                        if os.path.isfile(os.path.join(bay_folder,
                                                       f'scenario_{scenario}_{control_curve}_control_Setup_{learning_config["setup_chosen"].split("_")[1]}.csv')):
                            do = False

                if do:

                    if (config.sim_setting == 'ERIGrid_phase_2' or phase.split('_')[-1] == 'phase2'):
                        for element in chars_dict.keys():
                            if config.detection_application:
                                criteria = setup
                            else: criteria = learning_config["setup_chosen"].split("_")[1]
                            if criteria == 'A' :
                                if element.loc_name == 'LB 2':
                                    pf.set_referenced_characteristics(element, 'plini',
                                                                      chars_dict[element][f'p_{scenario}_{control_curve}'])
                                    pf.set_referenced_characteristics(element, 'qlini',
                                                                      chars_dict[element][f'q_{scenario}_{control_curve}'])
                                    load = element
                            if criteria == 'B' :
                                if element.loc_name == 'LB 1':
                                    pf.set_referenced_characteristics(element, 'plini',
                                                                      chars_dict[element][f'p_{scenario}_{control_curve}'])
                                    pf.set_referenced_characteristics(element, 'qlini',
                                                                      chars_dict[element][f'q_{scenario}_{control_curve}'])
                                    load = element

                    else:
                        PV.pf_over = control_curves[control_curve][0]
                        PV.p_over = control_curves[control_curve][1]
                        PV.pf_under = control_curves[control_curve][2]
                        PV.p_under = control_curves[control_curve][3]

                    if phase is not None:
                        t_start, t_end = set_times(file, element=load, chars_dict=chars_dict, scenario=scenario, control=control_curve, phase=phase.split('_')[-1])
                    else: t_start, t_end = set_times(file, element=load, chars_dict=chars_dict, scenario=scenario, control=control_curve)

                    result = set_QDS_settings(app, study_case_obj, t_start, t_end, step_unit=config.step_unit,
                                              balanced=config.balanced)  # set which vars and where!
                    results = run_QDS(app, int(scenario), result)

                    # save correctly as to be used by rest of framework
                    if config.detection_methods:
                        save_results(int(scenario), results, file, t_start, t_end, control_curve=control_curve)
                    if config.detection_application:
                        save_results(int(scenario), results, file, t_start, t_end, control_curve=control_curve,
                                     marker=estimation, setup=setup, phase=phase)

    return


def create_load_estimation_training_data(app, o_ElmNet, grid_data, study_case_obj, file, phase, setup):
    '''

    create data used for training the NN for load estimation

    '''

    chars_dict = grid_data[0]

    results_folder, file_folder = create_results_folder(file, phase=phase, training_data=True)
    training_data_folder = os.path.join(file_folder, 'Load_estimation_training_data')
    if not os.path.isdir(training_data_folder):
        os.mkdir(training_data_folder)

    PV = app.GetCalcRelevantObjects('*.ElmPvsys')[0]
    # if len(app.GetCalcRelevantObjects('*.ElmPvsys')) > 1: print('too many PVs!')

    for element in chars_dict.keys():
        if element in app.GetCalcRelevantObjects('.ElmLod'):
            pf.set_referenced_characteristics(element, 'plini', chars_dict[element][f'p'])
            pf.set_referenced_characteristics(element, 'qlini', chars_dict[element][f'q'])
        elif element in app.GetCalcRelevantObjects('.ElmPvsys'):
            pf.set_referenced_characteristics(element, 'pgini', chars_dict[element][f'p'])

        # run QDS for the same number of steps as there are load values per load

    t_start, t_end = set_times(file, element=PV, chars_dict=chars_dict)

    step_unit = 0  # 0 for seconds

    result = set_QDS_settings(app, study_case_obj, t_start, t_end, step_unit=step_unit,
                              balanced=0, step_size=1,
                              marker='load_estimation_training_data')  # set which vars and where!
    results = run_QDS(app, 1, result)

    # save correctly as to be used by load estimation
    if config.detection_application:
        save_results(1, results, file, t_start, t_end,
                     marker='training_data_setup_' + setup, folder=training_data_folder)

    return


def create_load_estimation_input_data(app, o_ElmNet, grid_data, study_case_obj, file, phase, setup):
    '''

    create data used for training the NN for load estimation

    '''

    chars_dict = grid_data[0]

    results_folder, file_folder = create_results_folder(file, phase=phase, setup=setup, estimate_input=True)
    estimation_input_data_folder = os.path.join(file_folder, 'Load_estimation_input_data')
    if not os.path.isdir(estimation_input_data_folder):
        os.mkdir(estimation_input_data_folder)

    PV = app.GetCalcRelevantObjects('*.ElmPvsys')[0]
    # if len(app.GetCalcRelevantObjects('*.ElmPvsys')) > 1: print('too many PVs!')

    for element in chars_dict.keys():
        if element in app.GetCalcRelevantObjects('.ElmLod'):
            pf.set_referenced_characteristics(element, 'plini', chars_dict[element][f'p'])
            pf.set_referenced_characteristics(element, 'qlini', chars_dict[element][f'q'])
        elif element in app.GetCalcRelevantObjects('.ElmPvsys'):
            pf.set_referenced_characteristics(element, 'pgini', chars_dict[element][f'p'])

        # run QDS for the same number of steps as there are load values per load

    t_start, t_end = set_times(file, element=PV, chars_dict=chars_dict)

    step_unit = 0  # 0 for seconds

    result = set_QDS_settings(app, study_case_obj, t_start, t_end, step_unit=step_unit,
                              balanced=0, step_size=1,
                              marker='load_estimation_training_data')  # set which vars and where!
    results = run_QDS(app, 1, result)

    # save correctly as to be used by load estimation
    if config.detection_application:
        save_results(1, results, file, t_start, t_end,
                     marker='estimation_input_data_setup_' + setup, folder=estimation_input_data_folder)

    return
