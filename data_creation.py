import pflib.pf as pf
import config
import pandas as pd
import numpy as np
import random, math
import datetime
import csv
import os

def set_QDS_settings(app, study_case_obj, file, t_start, t_end):

    local_machine_tz = config.local_machine_tz

    if study_case_obj.SearchObject('*.ComStatsim') is None:
        qds = app.GetFromStudyCase("ComStatsim")
        qds.Execute()

    qds_com_obj = study_case_obj.SearchObject('*.ComStatsim')
    qds_com_obj.SetAttribute('iopt_net', 0)                             # AC Load Flow, balanced, positive sequence
    qds_com_obj.SetAttribute('calcPeriod', 4)                           # User defined time range

    qds_com_obj.SetAttribute('startTime', int(
        (pd.DatetimeIndex([(t_start - t_start.tz_convert(local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
            0]))
    qds_com_obj.SetAttribute('endTime', int(
        (pd.DatetimeIndex([(t_end - t_end.tz_convert(local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
            0]))
    qds_com_obj.SetAttribute('stepSize', config.step_size)
    qds_com_obj.SetAttribute('stepUnit', 1)                                             # Minutes

    if config.parallel_computing == True:
        qds_com_obj.SetAttribute('iEnableParal', config.parallel_computing)                 # settings for parallel computation

        user = app.GetCurrentUser()
        settings = user.SearchObject(r'Set\Def\Settings')
        settings.SetAttribute('iActUseCore', 1)
        settings.SetAttribute('iCoreInput', config.cores)


    if config.just_voltages == True:        # define result variables of interest; result file is generated accordingly
        result_variables = {
            'ElmTerm': [
                'm:u',  # voltage at terminal
            ],
        }
    else:
        result_variables = {
            'ElmTerm': [
                'm:u',  # voltage at terminal
                'm:Pflow',  # active power at terminal
                'm:Qflow',  # reactive power at terminal
            ],
        }

    if config.system_language == 0:
        result = pf._resolve_result_object('Quasi-Dynamic Simulation AC')               # get result file
    else:
        if config.system_language == 1:
            result = pf._resolve_result_object('Quasi-Dynamische Simulation AC')        # get result file

    pf.set_vars_of_result_obj(result,
                                 result_variables=result_variables)

    return result

def create_malfunctioning_PVs(active_PVs, o_ElmNet, curves):
    '''
    Pick random active PV to have dysfunctional control;
    Therefore another equal PV is created with the malfunctioning control
    '''

    malfunctioning_devices = random.sample(active_PVs, config.number_of_broken_devices)
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

        o_Elm.pgini = i.pgini          # scale PV  output to the same
        o_Elm.sgn = i.sgn               # set apparent power as above

        o_Elm.outserv = 0               # initially in service, deactivated right at start of simulation > otherwise not considered by simulation at all

    return malfunctioning_devices, terms_with_malfunction

def create_malfunction_events(app, malfunctioning_devices, file):
    '''
    :param malfunctioning_devices:
    :return:
    event to change control at random point of time (correctly controlled object turned off, malfunctioning object turned on)
    '''

    if not config.t_start and not config.t_end:      #  default > simulation time inferred from available load/generation profile data
        t_start = pd.Timestamp(pd.read_csv(config.data_folder + file + '\\LoadProfile.csv', sep=';', index_col='time').index[0], tz='utc')
        t_end = pd.Timestamp(pd.read_csv(config.data_folder + file + '\\LoadProfile.csv', sep=';', index_col='time').index[-1], tz='utc')
    else:
        t_start = config.t_start
        t_end = config.t_end

    time_delta = int((t_end - t_start) / np.timedelta64(1, 's'))            # simulation duration converted to seconds
    seconds = random.sample([*range(0, time_delta, 1)], 1)[0]               # pick random point of the simulation
    t_off = pd.Timedelta(str(seconds) + 's')                                # set offset
    event_time = (pd.DatetimeIndex([(t_start + t_off)]).astype(np.int64) // 10 ** 9)[0]     # define event time in unix
    start_time = \
    (pd.DatetimeIndex([(t_start - t_start.tz_convert(config.local_machine_tz).utcoffset())]).astype(np.int64) // 10 ** 9)[
        0] + 1  # define start time in unix

    time_of_malfunction = str(pd.to_datetime(event_time, unit='s', utc=True).astimezone(datetime.datetime.now(
        datetime.timezone.utc).astimezone().tzinfo))  # PF converts event time to local time zone and executes the event accordingly
                                                      # > if passed 8 as event time, event time is converted to 9 in winter and to 10 in summer

    evtFold = app.GetFromStudyCase('IntEvtqds')

    for i in malfunctioning_devices:
        oEvent = evtFold.CreateObject('EvtOutage', i.loc_name + ' PVout')
        oEvent.tDateTime = int(event_time)
        oEvent.p_target = i
        oEvent.i_what = 0

        oEvent = evtFold.CreateObject('EvtOutage',  i.loc_name + ' PVbrokenout')  # at first PV has to be active and taken out of service;
        oEvent.tDateTime = int(start_time)                                         # otherwise it is not considered by simulation at all
        oEvent.p_target = app.GetCalcRelevantObjects(i.loc_name + ' broken.ElmGenstat')[0]
        oEvent.i_what = 0

        oEvent = evtFold.CreateObject('EvtOutage', i.loc_name + ' PVbrokenin')
        oEvent.tDateTime = int(event_time)
        oEvent.p_target = app.GetCalcRelevantObjects(i.loc_name + ' broken.ElmGenstat')[0]
        oEvent.i_what = 1

    return time_of_malfunction, t_start, t_end

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
            results[data] = results[data] * 1000 * 1000             # convert all powers from kW to Watts
            if config.reduce_result_file_size == True:
                results[data] = results[data].values.astype('int')

    return results

def save_results(count, malfunctioning_devices, time_of_malfunction, results, terminals_with_PVs, file):

    malfunction_type = {0 : 'cos(phi)(P)', 1 : 'Q(P)', 2 : 'broken Q(P) (flat curve)', 3 : 'wrong Q(P) (inversed curve)'}
    terminals_with_malfunction = [i.bus1.cterm.loc_name for i in malfunctioning_devices]

    metainfo = ['simulation#%d' % count, 'comment data format: active and reactive powers in Watts',
                'step time in minutes: %d' % config.step_size,
                'terminal(s) with malfunction: %s' % terminals_with_malfunction,
                'time of malfunction: %s' % time_of_malfunction,
                'type of malfunction: %s' % malfunction_type[config.broken_control_curve_choice],
                'terminals with PVs: %s' % terminals_with_PVs]

    metainfo += [''] * (len(results) - len(metainfo))
    results[('metainfo', 'in the first', 'few indices')] = metainfo

    results_folder = config.results_folder + file + '\\'
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    results.to_csv(results_folder + 'result_run#%d.csv' % count, header=True, sep=';', decimal='.', float_format='%.3f')
    csv.register_dialect('myDialect',
                         delimiter=';',
                         quoting=csv.QUOTE_NONE,
                         skipinitialspace=True)
    with open(results_folder + 'result_run#%d.csv' % count, 'a') as csvFile:       #add meta info
        writer = csv.writer(csvFile, dialect='myDialect')
        writer.writerow(metainfo)
    csvFile.close()

    return

def clean_up(app, active_PVs, malfunctioning_devices):
    '''

    :param active_Pvs:
    :return:

    reset everything that has been changed between simulation runs
    '''

    for o in active_PVs: o.outserv = 1                              # reset PVs to be out of service
    for o in malfunctioning_devices:                                # delete PVs (+cubicles) created for malfunctions
        inserted_dummy_device = app.GetCalcRelevantObjects(o.loc_name + ' broken' + '.ElmGenstat')[0]
        o_StaCubic = inserted_dummy_device.bus1
        inserted_dummy_device.Delete()
        o_StaCubic.Delete()

    evtFold = app.GetFromStudyCase('IntEvtqds')
    for o in evtFold.GetContents():                                #delete events set up
        o.Delete()

    return

def create_data(app, o_ElmNet, curves, study_case_obj, file):
    '''

    begin of loop to vary malfunctioning device and point of malfunction between simulation runs

    '''

    count = 0
    l_objects = [i for i in app.GetCalcRelevantObjects('*.ElmGenstat') if i.loc_name.split(' ')[1] == 'SGen']   # get list of PVs

    sample = math.floor(len(l_objects) * config.percentage / 100)                 # set % of terminals to have PV
    while count < config.simruns:                                    # every simrun has a different malfunction location and time (& different PV locations in general)

        active_PVs = random.sample(l_objects, sample)         # pick active PVs randomly
        for o in active_PVs: o.outserv = 0                    # PVs not outofservice and therefore active = installed PV
        terminals_with_PVs = [i.bus1.cterm.loc_name for i in active_PVs]

        malfunctioning_devices, terms_with_malfunction = create_malfunctioning_PVs(active_PVs, o_ElmNet, curves)
        time_of_malfunction, t_start, t_end = create_malfunction_events(app, malfunctioning_devices, file)
        result = set_QDS_settings(app, study_case_obj, file, t_start, t_end)

        results = run_QDS(app, count, result)
        save_results(count, malfunctioning_devices, time_of_malfunction, results, terminals_with_PVs, file)
        clean_up(app, active_PVs, malfunctioning_devices)

        count += 1

    return 0