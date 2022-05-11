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

    numbers = {}
    for number in list(range(len(Y))):
        numbers[Y[number][0].split(' ')[3] + '_' + Y[number][0].split(' ')[-1] + '_' + Y[number][0].split(' ')[0]] = number


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
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[numbers['A_F1_correct']], data[numbers['A_F1_correct']], 'tab:blue', label=label(numbers['A_F1_correct'],15))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[numbers['A_F1_wrong']], data[numbers['A_F1_wrong']], 'tab:red', label=label(numbers['A_F1_wrong'],13))
    if learning_config['data_source'] == 'simulation' and config.extended:
        axs[str(len(fgs.keys())) + '_ax1'].plot(X[numbers['A_F1_inversed']], data[numbers['A_F1_inversed']], 'tab:orange',
                                                label=label(numbers['A_F1_inversed'], 16))
    axs[str(len(fgs.keys())) + '_ax1'].set_title(list(measurements.keys())[numbers['A_F1_correct']][16:].split(':')[0] + ': Load and PV ' + f'({list(measurements.keys())[numbers["A_F1_correct"]][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax1'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax1'], start_end=start_end(numbers['A_F1_correct']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax1'])

    # Setup A Scenario x: Test Bay F2
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[numbers['A_F2_correct']], data[numbers['A_F2_correct']], 'tab:blue', label=label(numbers['A_F2_correct'],15))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[numbers['A_F2_wrong']], data[numbers['A_F2_wrong']], 'tab:red', label=label(numbers['A_F2_wrong'],13))
    if learning_config['data_source'] == 'simulation' and config.extended:
        axs[str(len(fgs.keys())) + '_ax2'].plot(X[numbers['A_F2_inversed']], data[numbers['A_F2_inversed']], 'tab:orange',
                                                label=label(numbers['A_F2_inversed'], 16))
    axs[str(len(fgs.keys())) + '_ax2'].set_title(list(measurements.keys())[numbers['A_F2_correct']][16:].split(':')[0] + ': Transformer ' + f'({list(measurements.keys())[numbers["A_F2_correct"]][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax2'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax2'], start_end=start_end(numbers['A_F2_correct']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax2'])

    # Setup B Scenario x: Test Bay B1
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[numbers['B_B1_correct']], data[numbers['B_B1_correct']], 'tab:blue', label=label(numbers['B_B1_correct'],15))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[numbers['B_B1_wrong']], data[numbers['B_B1_wrong']], 'tab:red', label=label(numbers['B_B1_wrong'],13))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[numbers['B_B1_inversed']], data[numbers['B_B1_inversed']], 'tab:orange', label=label(numbers['B_B1_inversed'],16))
    axs[str(len(fgs.keys())) + '_ax4'].set_title('  ' + list(measurements.keys())[numbers['B_B1_correct']][16:].split(':')[0] + ': Load and PV ' + f'({list(measurements.keys())[numbers["B_B1_correct"]][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax4'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax4'], start_end=start_end(numbers['B_B1_correct']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax4'])

    # Setup B Scenario x: Test Bay F2
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[numbers['B_F2_correct']], data[numbers['B_F2_correct']], 'tab:blue', label=label(numbers['B_F2_correct'], 15))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[numbers['B_F2_wrong']], data[numbers['B_F2_wrong']], 'tab:red', label=label(numbers['B_F2_wrong'], 13))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[numbers['B_F2_inversed']], data[numbers['B_F2_inversed']], 'tab:orange', label=label(numbers['B_F2_inversed'], 16))
    axs[str(len(fgs.keys())) + '_ax5'].set_title('  ' + list(measurements.keys())[numbers['B_F2_correct']][16:].split(':')[0] + ': Transformer ' + f'({list(measurements.keys())[numbers["B_F2_correct"]][16:].split(":")[1].split(" ")[-1]})')
    axs[str(len(fgs.keys())) + '_ax5'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax5'], start_end=start_end(numbers['B_F2_correct']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax5'])

    if not config.plot_only_trafo_and_pv:

        # Setup A Scenario x: Test Bay B1
        axs[str(len(fgs.keys())) + '_ax3'].plot(X[numbers['A_B1_correct']], data[numbers['A_B1_correct']], 'tab:blue', label=label(numbers['A_B1_correct'], 15))
        axs[str(len(fgs.keys())) + '_ax3'].plot(X[numbers['A_B1_wrong']], data[numbers['A_B1_wrong']], 'tab:red', label=label(numbers['A_B1_wrong'], 13))
        if learning_config['data_source'] == 'simulation' and config.extended:
            axs[str(len(fgs.keys())) + '_ax3'].plot(X[numbers['A_B1_inversed']], data[numbers['A_B1_inversed']],
                                                    'tab:orange',
                                                    label=label(numbers['A_B1_inversed'], 16))
        axs[str(len(fgs.keys())) + '_ax3'].set_title(list(measurements.keys())[numbers['A_B1_correct']][16:].split(':')[0] + ': Load ' + f'({list(measurements.keys())[numbers["A_B1_correct"]][16:].split(":")[1].split(" ")[-1]})')
        axs[str(len(fgs.keys())) + '_ax3'].legend()
        adjust_xticks(axs[str(len(fgs.keys())) + '_ax3'], start_end=start_end(numbers['A_B1_correct']))
        adjust_yticks(axs[str(len(fgs.keys())) + '_ax3'])

        # Setup B Scenario x: Test Bay F1
        axs[str(len(fgs.keys())) + '_ax6'].plot(X[numbers['B_F1_correct']], data[numbers['B_F1_correct']], 'tab:blue', label=label(numbers['B_F1_correct'],15))
        axs[str(len(fgs.keys())) + '_ax6'].plot(X[numbers['B_F1_wrong']], data[numbers['B_F1_wrong']], 'tab:red', label=label(numbers['B_F1_wrong'],13))
        axs[str(len(fgs.keys())) + '_ax6'].plot(X[numbers['B_F1_inversed']], data[13], 'tab:orange', label=label(numbers['B_F1_inversed'],16))
        axs[str(len(fgs.keys())) + '_ax6'].set_title('  ' + list(measurements.keys())[numbers['B_F1_correct']][16:].split(':')[0] + ': Load ' + f'({list(measurements.keys())[numbers["B_F1_correct"]][16:].split(":")[1].split(" ")[-1]})')
        axs[str(len(fgs.keys())) + '_ax6'].legend()
        adjust_xticks(axs[str(len(fgs.keys())) + '_ax6'], start_end=start_end(numbers['B_F1_correct']))
        adjust_yticks(axs[str(len(fgs.keys())) + '_ax6'])


    #plt.show()
    return fgs, axs

def  plot_scenario_case(measurements, fgs, axs, vars=None, phase='1', pu=True):


    if vars is None:
        vars = {'B1': ('Vrms ph-n AN Avg', 4), 'F1': ('Vrms ph-n AN Avg', 4), 'F2': ('Vrms ph-n AN Avg', 4)}
    X = [measurements[i].data.index for i in measurements]
    Y = [(measurements[i].name ,measurements[i].data) for i in measurements]

    numbers = {}
    for number in list(range(len(Y))):
        numbers[
            Y[number][0].split(' ')[3] + '_' + Y[number][0].split(' ')[-1] + '_' + Y[number][0].split(' ')[0]] = number

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
    if not (learning_config['data_source'] == 'simulation' and config.extended):
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

    if learning_config['data_source'] == 'simulation' and config.extended:
        axs[str(len(fgs.keys())) + '_ax5'] = plt.subplot2grid((3, 4), (2, 0), colspan=2, fig=fig)
        axs[str(len(fgs.keys())) + '_ax5'].set_xlabel('timestep')
        axs[str(len(fgs.keys())) + '_ax5'].set_ylabel(vars['B1'][0][0])

    axs[str(len(fgs.keys())) + '_ax6'] = plt.subplot2grid((3, 4), (2, 2), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax6'].set_xlabel('timestep')
    if not (learning_config['data_source'] == 'simulation' and config.extended):
        axs[str(len(fgs.keys())) + '_ax6'].set_ylabel(vars['B1'][0][0])
    axs[str(len(fgs.keys())) + '_ax6'].yaxis.tick_right()

    if config.note_avg_and_std:
        label = lambda x: list(measurements.keys())[x].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[x]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[x]), ".3f"))
    else:
        label = lambda x: list(measurements.keys())[x].split(': ')[1]

    start_end = lambda x: [X[x][0], X[x][-1]]

    #correct Setup A Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[numbers['A_B1_correct']], data[numbers['A_B1_correct']], 'tab:blue', label=label(numbers['A_B1_correct']))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[numbers['A_F1_correct']], data[numbers['A_F1_correct']], 'tab:green', label=label(numbers['A_F1_correct']))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[numbers['A_F2_correct']], data[numbers['A_F2_correct']], 'tab:orange',
                                            label=label(numbers['A_F2_correct']))
    axs[str(len(fgs.keys())) + '_ax1'].set_title(list(measurements.keys())[0].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax1'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax1'], start_end=start_end(numbers['A_F2_correct']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax1'])

    #wrong Setup A Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[numbers['A_B1_wrong']], data[numbers['A_B1_wrong']], 'tab:blue',
                                            label=label(numbers['A_B1_wrong']))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[numbers['A_F1_wrong']], data[numbers['A_F1_wrong']], 'tab:green',
                                            label=label(numbers['A_F1_wrong']))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[numbers['A_F2_wrong']], data[numbers['A_F2_wrong']], 'tab:orange',
                                            label=label(numbers['A_F2_wrong']))
    axs[str(len(fgs.keys())) + '_ax2'].set_title(list(measurements.keys())[numbers['A_B1_wrong']].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax2'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax2'], start_end=start_end(numbers['A_F2_wrong']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax2'])

    #correct Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[numbers['B_B1_correct']], data[numbers['B_B1_correct']], 'tab:blue',
                                            label=label(numbers['B_B1_correct']))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[numbers['B_F1_correct']], data[numbers['B_F1_correct']], 'tab:green',
                                            label=label(numbers['B_F1_correct']))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[numbers['B_F2_correct']], data[numbers['B_F2_correct']], 'tab:orange',
                                            label=label(numbers['B_F2_correct']))
    axs[str(len(fgs.keys())) + '_ax3'].set_title(list(measurements.keys())[numbers['B_B1_correct']].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax3'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax3'], start_end=start_end(numbers['B_B1_correct']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax3'])

    #wrong Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[numbers['B_B1_wrong']], data[numbers['B_B1_wrong']], 'tab:blue',
                                            label=label(numbers['B_B1_wrong']))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[numbers['B_F1_wrong']], data[numbers['B_F1_wrong']], 'tab:green',
                                            label=label(numbers['B_F1_wrong']))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[numbers['B_F2_wrong']], data[numbers['B_F2_wrong']], 'tab:orange',
                                            label=label(numbers['B_F2_wrong']))
    axs[str(len(fgs.keys())) + '_ax4'].set_title(list(measurements.keys())[numbers['B_B1_wrong']].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax4'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax4'], start_end=start_end(numbers['B_B1_wrong']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax4'])

    #inversed Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[numbers['B_B1_inversed']], data[numbers['B_B1_inversed']], 'tab:blue',
                                            label=label(numbers['B_B1_inversed']))
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[numbers['B_F1_inversed']], data[numbers['B_F1_inversed']], 'tab:green',
                                            label=label(numbers['B_F1_inversed']))
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[numbers['B_F2_inversed']], data[numbers['B_F2_inversed']], 'tab:orange',
                                            label=label(numbers['B_F2_inversed']))
    axs[str(len(fgs.keys())) + '_ax6'].set_title(list(measurements.keys())[numbers['B_B1_inversed']].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax6'].legend()
    adjust_xticks(axs[str(len(fgs.keys())) + '_ax6'], start_end=start_end(numbers['B_B1_inversed']))
    adjust_yticks(axs[str(len(fgs.keys())) + '_ax6'])

    if learning_config['data_source'] == 'simulation' and config.extended:
        # inversed Setup A Scenario x: Test Bays B1,F1,F2
        axs[str(len(fgs.keys())) + '_ax5'].plot(X[numbers['A_B1_inversed']], data[numbers['A_B1_inversed']], 'tab:blue',
                                                label=label(numbers['A_B1_inversed']))
        axs[str(len(fgs.keys())) + '_ax5'].plot(X[numbers['A_F1_inversed']], data[numbers['A_F1_inversed']], 'tab:green',
                                                label=label(numbers['A_F1_inversed']))
        axs[str(len(fgs.keys())) + '_ax5'].plot(X[numbers['A_F2_inversed']], data[numbers['A_F2_inversed']], 'tab:orange',
                                                label=label(numbers['A_F2_inversed']))
        axs[str(len(fgs.keys())) + '_ax5'].set_title(list(measurements.keys())[numbers['A_B1_inversed']].split(': ')[0])
        axs[str(len(fgs.keys())) + '_ax5'].legend()
        adjust_xticks(axs[str(len(fgs.keys())) + '_ax5'], start_end=start_end(numbers['A_F2_inversed']))
        adjust_yticks(axs[str(len(fgs.keys())) + '_ax5'])

    else:
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
