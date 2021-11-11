import pflib.pf as pf
import pflib.object_frame as pof
import pandas as pd
import numpy as np
import random, math
import datetime
import os
import importlib
from experiment_config import experiment_path, chosen_experiment

from raw_data_generation.Sim.core.run_multiple_simulations import run_simulations

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def set_QDS_settings(app, study_case_obj, t_start, t_end):
    local_machine_tz = config.local_machine_tz

    if study_case_obj.SearchObject('*.ComStatsim') is None:
        qds = app.GetFromStudyCase("ComStatsim")
        qds.Execute()

    qds_com_obj = study_case_obj.SearchObject('*.ComStatsim')
    qds_com_obj.SetAttribute('iopt_net', 0)  # AC Load Flow, balanced, positive sequence
    qds_com_obj.SetAttribute('calcPeriod', 4)  # User defined time range

    qds_com_obj.SetAttribute('startTime', int(
        (pd.DatetimeIndex([(t_start - t_start.tz_convert(local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
            0]))
    qds_com_obj.SetAttribute('endTime', int(
        (pd.DatetimeIndex([(t_end - t_end.tz_convert(local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
            0]))
    qds_com_obj.SetAttribute('stepSize', config.step_size)
    qds_com_obj.SetAttribute('stepUnit', 1)  # Minutes

    if config.parallel_computing == True:
        qds_com_obj.SetAttribute('iEnableParal', config.parallel_computing)  # settings for parallel computation

        user = app.GetCurrentUser()
        settings = user.SearchObject(r'Set\Def\Settings')
        settings.SetAttribute('iActUseCore', 1)
        settings.SetAttribute('iCoreInput', config.cores)

    if config.just_voltages == True:  # define result variables of interest; result file is generated accordingly
        result_variables = {
            'ElmTerm': [
                'm:u',  # voltage at terminal
            ],
        }
    else:
        result_variables = {
            'ElmTerm': [
                'm:u',  # voltage at terminal
                'm:Pgen',  # active power generated at terminal
                'm:Qgen',  # reactive power generated at terminal
            ],
        }

    if config.system_language == 0:
        result = pf._resolve_result_object('Quasi-Dynamic Simulation AC')  # get result file
    else:
        if config.system_language == 1:
            result = pf._resolve_result_object('Quasi-Dynamische Simulation AC')  # get result file

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


def set_times(file):
    if not config.t_start and not config.t_end:  # default > simulation time inferred from available load/generation profile data
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
            results[data] = results[data] * 1000 * 1000  # convert all powers from kW to Watts
            if config.reduce_result_file_size == True:
                results[data] = results[data].values.astype('int')

    return results


def save_results(count, results, file, t_start, t_end, malfunctioning_devices=None, time_of_malfunction=None,
                 terminals_with_PVs=None, terminals_with_EVCs=None, terminals=None):
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


def create_results_folder(file):
    penetrations = ''
    for key, value in config.percentage.items():
        if value > 0:
            penetrations += '_' + key + '(' + str(value) + ')'

    results_folder = os.path.join(config.raw_data_folder,
                                  (config.raw_data_set_name + penetrations + '_raw_data'))
    file_folder = os.path.join(results_folder, file)

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    if not os.path.isdir(file_folder):
        os.mkdir(file_folder)

    return results_folder, file_folder


def create_data(app, o_ElmNet, grid_data, study_case_obj, file):
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
    sample_EVCs = math.floor(len(EV_charging_stations) * config.percentage['EV'] / 100) # set % of terminals to have EV charging stations

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
            malfunctioning_devices, terms_with_malfunction = create_malfunctioning_devices(active_PVs, active_EVCS, o_ElmNet, curves) #only 1 malfunction looked at? also only 1 type of malfunction?
            current_grid_info.append(malfunctioning_devices)
            print(malfunctioning_devices[0].loc_name)
            # time_of_malfunction = create_malfunction_events(app, malfunctioning_devices, t_start, t_end)

        if config.number_of_broken_devices_and_type[1] == 'PV':    #only here QDS can be done, otherwise malfunction has to be produced in individual loadflows
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
