import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import matplotlib.ticker as ticker
from statistics import mean, pstdev
import numpy as np
import importlib
from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

def adjust_yticks(ax, step_size=0.01, format='%0.2f'):


    start, end = ax.get_ylim()
    if len(np.arange(start, end, step_size)) < 3:
        ticks = np.append(np.arange(start, end, step_size), np.arange(start, end, step_size)[-1]+step_size)
    else: ticks = np.arange(start, end, step_size)
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(format))

    return ax

def adjust_xticks(ax,start_end, step_size=2000, format='%0.f'):


    start = start_end[0]
    end = round(start_end[1], -3) #round to full 1000 (6000 instead of 5999)
    if len(np.arange(start, end, step_size)) < 4: ticks = np.append(np.arange(start, end, step_size), end)
    else: ticks = np.arange(start, end, step_size)
    ax.xaxis.set_ticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format))

    return ax

def  plot_scenario_test_bay(measurements, fgs, axs, vars=None, phase='1', pu=True):

    if vars is None:
        vars = {'B1': ('Vrms ph-n AN Avg', 4), 'F1': ('Vrms ph-n AN Avg', 4), 'F2': ('Vrms ph-n AN Avg', 4)}

    X = [measurements[i].data.index for i in measurements]
    Y = [(measurements[i].name, measurements[i].data) for i in measurements]

    data = [i[1][i[1].columns[vars[i[0][-2:]][1]]] for i in Y]  # voltages_phase_1_avg

    if learning_config['data_source'] == 'simulation':
        data = [i *1000 for i in data]
    if pu == True and vars['B1'][1] <= 126:  # only for voltages
        data = [i / 230 for i in data]

    if config.plot_only_trafo_and_pv:
        fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2)
    else:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    #fgs[str(len(fgs.keys()) + 1)] = fig
    fgs[[i.split(' ')[5][:-1] for i in measurements.keys()][0]] = fig

    if pu:
        fig.suptitle(f"Variable: {vars['B1'][0]} in per unit")
    else:
        fig.suptitle(f"Variable: {vars['B1'][0]}")

    axs[str(len(fgs.keys())) + '_ax1'] = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax1'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax1'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax2'] = plt.subplot2grid((3, 4), (1, 0), colspan=2, fig=fig)
    if config.plot_only_trafo_and_pv: axs[str(len(fgs.keys())) + '_ax2'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax2'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax4'] = plt.subplot2grid((3, 4), (0, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax4'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax4'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax4'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax5'] = plt.subplot2grid((3, 4), (1, 2), colspan=2, fig=fig)
    if config.plot_only_trafo_and_pv: axs[str(len(fgs.keys())) + '_ax5'].set_xlabel('timestep')
    # axs[str(len(fgs.keys())) + '_ax5'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax5'].yaxis.tick_right()


    if not config.plot_only_trafo_and_pv:
        axs[str(len(fgs.keys())) + '_ax3'] = plt.subplot2grid((3, 4), (2, 0), colspan=2, fig=fig)
        axs[str(len(fgs.keys())) + '_ax3'].set_xlabel('timestep')
        axs[str(len(fgs.keys())) + '_ax3'].set_ylabel(vars['B1'][0])

        axs[str(len(fgs.keys())) + '_ax6'] = plt.subplot2grid((3, 4), (2, 2), colspan=2, fig=fig)
        axs[str(len(fgs.keys())) + '_ax6'].set_xlabel('timestep')
        #axs[str(len(fgs.keys())) + '_ax6'].set_ylabel(var[0])
        axs[str(len(fgs.keys())) + '_ax6'].yaxis.tick_right()


    if config.note_avg_and_std:
        label = lambda x, y: list(measurements.keys())[x][:y] + ' avg: ' + str(
            format(mean(data[x]), ".3f")) + ' std: ' + str(format(pstdev(data[x]), ".3f"))
    else:
        label = lambda x, y: list(measurements.keys())[x][:y]

    start_end = lambda x: [X[x][0], X[x][-1]]

    # Setup A Scenario x: Test Bay F1
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[1], data[1], 'tab:blue', label=label(1,15))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[4], data[4], 'tab:red', label=label(4,13))
    axs[str(len(fgs.keys())) + '_ax1'].set_title(list(measurements.keys())[1][16:].split(':')[0] + ': Load and PV ' + f'({list(measurements.keys())[1][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax1'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax1'], start_end=start_end(1))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax1'])

    # Setup A Scenario x: Test Bay F2
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[2], data[2], 'tab:blue', label=label(2,15))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[5], data[5], 'tab:red', label=label(5,13))
    axs[str(len(fgs.keys())) + '_ax2'].set_title(list(measurements.keys())[2][16:].split(':')[0] + ': Transformer ' + f'({list(measurements.keys())[2][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax2'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax2'], start_end=start_end(2))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax2'])

    # Setup B Scenario x: Test Bay B1
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[6], data[6], 'tab:blue', label=label(6,15))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[9], data[9], 'tab:red', label=label(9,13))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[12], data[12], 'tab:orange', label=label(12,16))
    axs[str(len(fgs.keys())) + '_ax4'].set_title('  ' + list(measurements.keys())[6][16:].split(':')[0] + ': Load and PV ' + f'({list(measurements.keys())[6][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax4'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax4'], start_end=start_end(6))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax4'])

    # Setup B Scenario x: Test Bay F2
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[8], data[8], 'tab:blue', label=label(8, 15))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[11], data[11], 'tab:red', label=label(11, 13))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[14], data[14], 'tab:orange', label=label(14, 16))
    axs[str(len(fgs.keys())) + '_ax5'].set_title('  ' + list(measurements.keys())[8][16:].split(':')[0] + ': Transformer ' + f'({list(measurements.keys())[8][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax5'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax5'], start_end=start_end(8))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax5'])

    if not config.plot_only_trafo_and_pv:

        # Setup A Scenario x: Test Bay B1
        axs[str(len(fgs.keys())) + '_ax3'].plot(X[0], data[0], 'tab:blue', label=label(0, 15))
        axs[str(len(fgs.keys())) + '_ax3'].plot(X[3], data[3], 'tab:red', label=label(3, 13))
        axs[str(len(fgs.keys())) + '_ax3'].set_title(list(measurements.keys())[0][16:].split(':')[0] + ': Load ' + f'({list(measurements.keys())[0][16:].split(":")[1].split(" ")[-1]})')
        axs[str(len(fgs.keys())) + '_ax3'].legend()
        adjust_xticks(axs[str(len(fgs.keys())) + '_ax3'], start_end=start_end(0))
        adjust_yticks(axs[str(len(fgs.keys())) + '_ax3'])

        # Setup B Scenario x: Test Bay F1
        axs[str(len(fgs.keys())) + '_ax6'].plot(X[7], data[7], 'tab:blue', label=label(7,15))
        axs[str(len(fgs.keys())) + '_ax6'].plot(X[10], data[10], 'tab:red', label=label(10,13))
        axs[str(len(fgs.keys())) + '_ax6'].plot(X[13], data[13], 'tab:orange', label=label(13,16))
        axs[str(len(fgs.keys())) + '_ax6'].set_title('  ' + list(measurements.keys())[7][16:].split(':')[0] + ': Load ' + f'({list(measurements.keys())[7][16:].split(":")[1].split(" ")[-1]})')
        axs[str(len(fgs.keys())) + '_ax6'].legend()
        adjust_xticks(axs[str(len(fgs.keys())) + '_ax6'], start_end=start_end(7))
        adjust_yticks(axs[str(len(fgs.keys())) + '_ax6'])


    #plt.show()
    return fgs, axs

def  plot_scenario_case(measurements, fgs, axs, vars=None, phase='1', pu=True):


    if vars is None:
        vars = {'B1': ('Vrms ph-n AN Avg', 4), 'F1': ('Vrms ph-n AN Avg', 4), 'F2': ('Vrms ph-n AN Avg', 4)}
    X = [measurements[i].data.index for i in measurements]
    Y = [(measurements[i].name ,measurements[i].data) for i in measurements]

    data = [i[1][i[1].columns[vars[i[0][-2:]][1]]] for i in Y]  # voltages_phase_1_avg
    if learning_config['data_source'] == 'simulation':
        data = [i *1000 for i in data]
    if pu == True and vars['B1'][1] <= 126:  # only for voltages
        data = [i / 230 for i in data]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    #fgs[str(len(fgs.keys())+1)] = fig
    fgs[[i.split(' ')[5][:-1] for i in measurements.keys()][0]] = fig

    if pu:
        fig.suptitle(f"Variable: {vars['B1'][0][0]} in per unit")
    else:
        fig.suptitle(f"Variable: {vars['B1'][0][0]}")

    axs[str(len(fgs.keys())) + '_ax1'] = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax1'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax1'].set_ylabel(vars['B1'][0][0])

    axs[str(len(fgs.keys())) + '_ax2'] = plt.subplot2grid((3, 4), (1, 0), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax2'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax2'].set_ylabel(vars['B1'][0][0])

    axs[str(len(fgs.keys())) + '_ax3'] = plt.subplot2grid((3, 4), (0, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax3'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax3'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax3'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax4'] = plt.subplot2grid((3, 4), (1, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax4'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax4'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax4'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax6'] = plt.subplot2grid((3, 4), (2, 2), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax6'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax6'].set_ylabel(vars['B1'][0][0])
    axs[str(len(fgs.keys())) + '_ax6'].yaxis.tick_right()

    if config.note_avg_and_std:
        label = lambda x: list(measurements.keys())[x].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[y]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[y]), ".3f"))
    else:
        label = lambda x: list(measurements.keys())[x].split(': ')[1]

    start_end = lambda x: [X[x][0], X[x][-1]]

    #correct Setup A Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[0], data[0], 'tab:blue', label=label(0))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[1], data[1], 'tab:green', label=label(1))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[2], data[2], 'tab:orange',
                                            label=label(2))
    axs[str(len(fgs.keys())) + '_ax1'].set_title(list(measurements.keys())[0].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax1'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax1'], start_end=start_end(0))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax1'])

    #wrong Setup A Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[3], data[3], 'tab:blue',
                                            label=label(3))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[4], data[4], 'tab:green',
                                            label=label(4))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[5], data[5], 'tab:orange',
                                            label=label(5))
    axs[str(len(fgs.keys())) + '_ax2'].set_title(list(measurements.keys())[3].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax2'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax2'], start_end=start_end(3))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax2'])

    #correct Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[6], data[6], 'tab:blue',
                                            label=label(6))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[7], data[7], 'tab:green',
                                            label=label(7))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[8], data[8], 'tab:orange',
                                            label=label(8))
    axs[str(len(fgs.keys())) + '_ax3'].set_title(list(measurements.keys())[6].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax3'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax3'], start_end=start_end(6))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax3'])

    #wrong Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[9], data[9], 'tab:blue',
                                            label=label(9))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[10], data[10], 'tab:green',
                                            label=label(10))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[11], data[11], 'tab:orange',
                                            label=label(11))
    axs[str(len(fgs.keys())) + '_ax4'].set_title(list(measurements.keys())[9].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax4'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax4'], start_end=start_end(9))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax4'])

    #inversed Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[12], data[12], 'tab:blue',
                                            label=label(12))
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[13], data[13], 'tab:green',
                                            label=label(13))
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[14], data[14], 'tab:orange',
                                            label=label(14))
    axs[str(len(fgs.keys())) + '_ax6'].set_title(list(measurements.keys())[12].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax6'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax6'], start_end=start_end(12))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax6'])

    ax5.set_axis_off()

    #plt.show()
    return fgs, axs

def scatter(number, colors, data, ax):

    if number == 0:
        label = 'correct'
    elif number == 1:
        label = 'flat'
    elif number == 2:
        label = 'inversed'
    else:
        label = 'unknown label'

    s = 25
    for point in data[:-1]:

        ax.scatter(point[0],point[1],c = colors[number], s = s)

    ax.scatter(data[-1][0], data[-1][1], c=colors[number], s=s, label=label)

    return ax

def plot_pca(data, type='PCA', number_of_vars=15):

    fgs = {}
    axs = {}

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fgs[str(len(fgs.keys()) + 1)] = fig

    fig.suptitle(f"Explained variances of {type} on measurements using a maximum of {number_of_vars} variables")

    if type == 'kPCA':
        y_label = 'variance of 2nd PC'
        x_label = 'variance of 1st PC'
    else:
        y_label = 'expl. variance by 2nd PC'
        x_label = 'expl. variance by 1st PC'


    axs[str(len(fgs.keys())) + '_ax1'] = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2, fig=fig)
    # axs[str(len(fgs.keys())) + '_ax1'].set_xlabel(x_label)
    axs[str(len(fgs.keys())) + '_ax1'].set_ylabel(y_label)

    axs[str(len(fgs.keys())) + '_ax2'] = plt.subplot2grid((3, 4), (1, 0), colspan=2, fig=fig)
    # axs[str(len(fgs.keys())) + '_ax2'].set_xlabel(x_label)
    # axs[str(len(fgs.keys())) + '_ax2'].set_ylabel(y_label)

    axs[str(len(fgs.keys())) + '_ax3'] = plt.subplot2grid((3, 4), (2, 0), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax3'].set_xlabel(x_label)
    axs[str(len(fgs.keys())) + '_ax3'].set_ylabel(y_label)

    axs[str(len(fgs.keys())) + '_ax4'] = plt.subplot2grid((3, 4), (0, 2), colspan=2, fig=fig)
    # axs[str(len(fgs.keys())) + '_ax4'].set_xlabel(x_label)
    # axs[str(len(fgs.keys())) + '_ax4'].set_ylabel(y_label)
    axs[str(len(fgs.keys())) + '_ax4'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax5'] = plt.subplot2grid((3, 4), (1, 2), colspan=2, fig=fig)
    # axs[str(len(fgs.keys())) + '_ax5'].set_xlabel(x_label)
    # axs[str(len(fgs.keys())) + '_ax5'].set_ylabel(y_label)
    axs[str(len(fgs.keys())) + '_ax5'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax6'] = plt.subplot2grid((3, 4), (2, 2), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax6'].set_xlabel(x_label)
    # axs[str(len(fgs.keys())) + '_ax6'].set_ylabel(y_label)
    axs[str(len(fgs.keys())) + '_ax6'].yaxis.tick_right()

    colors = ['g', 'r', 'orange']
    # Setup A Scenario x: Test Bay B1
    axs[str(len(fgs.keys())) + '_ax1'] = scatter(0, colors, data['correct control Setup A: Test Bay B1'], axs[str(len(fgs.keys())) + '_ax1'])
    axs[str(len(fgs.keys())) + '_ax1'] = scatter(1, colors, data['wrong control Setup A: Test Bay B1'], axs[str(len(fgs.keys())) + '_ax1'])
    axs[str(len(fgs.keys())) + '_ax1'].set_title('Setup A: Test Bay B1 (Load at end of feeder)')
    axs[str(len(fgs.keys())) + '_ax1'].legend()

    # Setup A Scenario x: Test Bay F1
    axs[str(len(fgs.keys())) + '_ax2'] = scatter(0, colors, data['correct control Setup A: Test Bay F1'],
                                                 axs[str(len(fgs.keys())) + '_ax2'])
    axs[str(len(fgs.keys())) + '_ax2'] = scatter(1, colors, data['wrong control Setup A: Test Bay F1'],
                                                 axs[str(len(fgs.keys())) + '_ax2'])
    axs[str(len(fgs.keys())) + '_ax2'].set_title('Setup A: Test Bay F1 (PV + Load at begin of feeder)')
    axs[str(len(fgs.keys())) + '_ax2'].legend()

    # Setup A Scenario x: Test Bay F2
    axs[str(len(fgs.keys())) + '_ax3'] = scatter(0, colors, data['correct control Setup A: Test Bay F2'],
                                                 axs[str(len(fgs.keys())) + '_ax3'])
    axs[str(len(fgs.keys())) + '_ax3'] = scatter(1, colors, data['wrong control Setup A: Test Bay F2'],
                                                 axs[str(len(fgs.keys())) + '_ax3'])
    axs[str(len(fgs.keys())) + '_ax3'].set_title('Setup A: Test Bay F2 (Substation)')
    axs[str(len(fgs.keys())) + '_ax3'].legend()

    # Setup B Scenario x: Test Bay B1
    axs[str(len(fgs.keys())) + '_ax4'] = scatter(0, colors, data['correct control Setup B: Test Bay B1'],
                                                 axs[str(len(fgs.keys())) + '_ax4'])
    axs[str(len(fgs.keys())) + '_ax4'] = scatter(1, colors, data['wrong control Setup B: Test Bay B1'],
                                                 axs[str(len(fgs.keys())) + '_ax4'])
    axs[str(len(fgs.keys())) + '_ax4'] = scatter(2, colors, data['inversed control Setup B: Test Bay B1'],
                                                 axs[str(len(fgs.keys())) + '_ax4'])
    axs[str(len(fgs.keys())) + '_ax4'].set_title('Setup B: Test Bay B1 (PV + Load at end of feeder)')
    axs[str(len(fgs.keys())) + '_ax4'].legend()

    # Setup B Scenario x: Test Bay F1
    axs[str(len(fgs.keys())) + '_ax5'] = scatter(0, colors, data['correct control Setup B: Test Bay F1'],
                                                 axs[str(len(fgs.keys())) + '_ax5'])
    axs[str(len(fgs.keys())) + '_ax5'] = scatter(1, colors, data['wrong control Setup B: Test Bay F1'],
                                                 axs[str(len(fgs.keys())) + '_ax5'])
    axs[str(len(fgs.keys())) + '_ax5'] = scatter(2, colors, data['inversed control Setup B: Test Bay F1'],
                                                 axs[str(len(fgs.keys())) + '_ax5'])
    axs[str(len(fgs.keys())) + '_ax5'].set_title('Setup B: Test Bay F1 (Load at begin of feeder)')
    axs[str(len(fgs.keys())) + '_ax5'].legend()

    # Setup B Scenario x: Test Bay F2"
    axs[str(len(fgs.keys())) + '_ax6'] = scatter(0, colors, data['correct control Setup B: Test Bay F2'],
                                                 axs[str(len(fgs.keys())) + '_ax6'])
    axs[str(len(fgs.keys())) + '_ax6'] = scatter(1, colors, data['wrong control Setup B: Test Bay F2'],
                                                 axs[str(len(fgs.keys())) + '_ax6'])
    axs[str(len(fgs.keys())) + '_ax6'] = scatter(2, colors, data['inversed control Setup B: Test Bay F2'],
                                                 axs[str(len(fgs.keys())) + '_ax6'])
    axs[str(len(fgs.keys())) + '_ax6'].set_title('Setup B: Test Bay F2 (Substation)')
    axs[str(len(fgs.keys())) + '_ax6'].legend()

    # plt.show()
    return fgs, axs

def plot_grid_search(var, scores, x_label='number of PCA components', y_label='score', title='SVM with a polynomial kernel'):

    fig, ax = plt.subplots()
    y = {}
    for score in scores[0].keys():
        y[score] = [scores[point-1][score] for point in var]
    for score in y:
        ax.plot(var, np.array(y[score]).mean(axis=1), label=score)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        #ax.set_xticks(var)
        ax.set_title(title)
        ax.legend()

    return fig, ax
