import matplotlib.pyplot as plt
from statistics import mean, pstdev
import numpy as np
import importlib
from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
learning_config = config.learning_config

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

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    fgs[str(len(fgs.keys()) + 1)] = fig

    if pu:
        fig.suptitle(f"Variable: {vars['B1'][0]} in per unit")
    else:
        fig.suptitle(f"Variable: {vars['B1'][0]}")

    axs[str(len(fgs.keys())) + '_ax1'] = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax1'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax1'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax2'] = plt.subplot2grid((3, 4), (1, 0), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax2'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax2'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax3'] = plt.subplot2grid((3, 4), (2, 0), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax3'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax3'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax4'] = plt.subplot2grid((3, 4), (0, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax4'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax4'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax4'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax5'] = plt.subplot2grid((3, 4), (1, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax5'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax5'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax5'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax6'] = plt.subplot2grid((3, 4), (2, 2), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax6'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax6'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax6'].yaxis.tick_right()

    #Setup A Scenario x: Test Bay B1
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[0], data[0], 'tab:blue', label=list(measurements.keys())[0][:15] + ' avg: ' + str(format(mean(data[0]), ".3f")) + ' std: ' + str(format(pstdev(data[0]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[3], data[3], 'tab:red', label=list(measurements.keys())[3][:13] + ' avg: ' + str(format(mean(data[3]), ".3f")) + ' std: ' + str(format(pstdev(data[3]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax1'].set_title(list(measurements.keys())[0][16:])
    axs[str(len(fgs.keys())) + '_ax1'].legend()

    # Setup A Scenario x: Test Bay F1
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[1], data[1], 'tab:blue', label=list(measurements.keys())[1][:15] + ' avg: ' + str(format(mean(data[1]), ".3f")) + ' std: ' + str(format(pstdev(data[1]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[4], data[4], 'tab:red', label=list(measurements.keys())[4][:13] + ' avg: ' + str(format(mean(data[4]), ".3f")) + ' std: ' + str(format(pstdev(data[4]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax2'].set_title(list(measurements.keys())[1][16:])
    axs[str(len(fgs.keys())) + '_ax2'].legend()

    # Setup A Scenario x: Test Bay F2
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[2], data[2], 'tab:blue', label=list(measurements.keys())[2][:15] + ' avg: ' + str(format(mean(data[2]), ".3f")) + ' std: ' + str(format(pstdev(data[2]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[5], data[5], 'tab:red', label=list(measurements.keys())[5][:13] + ' avg: ' + str(format(mean(data[5]), ".3f")) + ' std: ' + str(format(pstdev(data[5]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax3'].set_title(list(measurements.keys())[2][16:])
    axs[str(len(fgs.keys())) + '_ax3'].legend()

    # Setup B Scenario x: Test Bay B1
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[6], data[6], 'tab:blue', label=list(measurements.keys())[6][:15] + ' avg: ' + str(format(mean(data[6]), ".3f")) + ' std: ' + str(format(pstdev(data[6]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[9], data[9], 'tab:red', label=list(measurements.keys())[9][:13] + ' avg: ' + str(format(mean(data[9]), ".3f")) + ' std: ' + str(format(pstdev(data[9]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[12], data[12], 'tab:orange', label=list(measurements.keys())[12][:16] + ' avg: ' + str(format(mean(data[12]), ".3f")) + ' std: ' + str(format(pstdev(data[12]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax4'].set_title(list(measurements.keys())[6][16:])
    axs[str(len(fgs.keys())) + '_ax4'].legend()

    # Setup B Scenario x: Test Bay F1
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[7], data[7], 'tab:blue', label=list(measurements.keys())[7][:15] + ' avg: ' + str(format(mean(data[7]), ".3f")) + ' std: ' + str(format(pstdev(data[7]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[10], data[10], 'tab:red', label=list(measurements.keys())[10][:13] + ' avg: ' + str(format(mean(data[10]), ".3f")) + ' std: ' + str(format(pstdev(data[10]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[13], data[13], 'tab:orange', label=list(measurements.keys())[13][:16] + ' avg: ' + str(format(mean(data[13]), ".3f")) + ' std: ' + str(format(pstdev(data[13]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax5'].set_title(list(measurements.keys())[7][16:])
    axs[str(len(fgs.keys())) + '_ax5'].legend()

    # Setup B Scenario x: Test Bay F"
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[8], data[8], 'tab:blue', label=list(measurements.keys())[8][:15] + ' avg: ' + str(format(mean(data[8]), ".3f")) + ' std: ' + str(format(pstdev(data[8]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[11], data[11], 'tab:red', label=list(measurements.keys())[11][:13] + ' avg: ' + str(format(mean(data[11]), ".3f")) + ' std: ' + str(format(pstdev(data[11]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax6'].plot(X[14], data[14], 'tab:orange', label=list(measurements.keys())[14][:16] + ' avg: ' + str(format(mean(data[14]), ".3f")) + ' std: ' + str(format(pstdev(data[14]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax6'].set_title(list(measurements.keys())[8][16:])
    axs[str(len(fgs.keys())) + '_ax6'].legend()

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
    fgs[str(len(fgs.keys())+1)] = fig

    if pu:
        fig.suptitle(f"Variable: {vars['B1'][0]} in per unit")
    else:
        fig.suptitle(f"Variable: {vars['B1'][0]}")

    axs[str(len(fgs.keys())) + '_ax1'] = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax1'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax1'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax2'] = plt.subplot2grid((3, 4), (1, 0), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax2'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax2'].set_ylabel(vars['B1'][0])

    axs[str(len(fgs.keys())) + '_ax3'] = plt.subplot2grid((3, 4), (0, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax3'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax3'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax3'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax4'] = plt.subplot2grid((3, 4), (1, 2), colspan=2, fig=fig)
    #axs[str(len(fgs.keys())) + '_ax4'].set_xlabel('timestep')
    #axs[str(len(fgs.keys())) + '_ax4'].set_ylabel(var[0])
    axs[str(len(fgs.keys())) + '_ax4'].yaxis.tick_right()

    axs[str(len(fgs.keys())) + '_ax5'] = plt.subplot2grid((3, 4), (2, 2), colspan=2, fig=fig)
    axs[str(len(fgs.keys())) + '_ax5'].set_xlabel('timestep')
    axs[str(len(fgs.keys())) + '_ax5'].set_ylabel(vars['B1'][0])
    axs[str(len(fgs.keys())) + '_ax5'].yaxis.tick_right()

    #correct Setup A Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[0], data[0], 'tab:blue', label=list(measurements.keys())[0].split(': ')[1] + ' avg: ' + str(format(mean(data[0]), ".3f")) + ' std: ' + str(format(pstdev(data[0]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[1], data[1], 'tab:green', label=list(measurements.keys())[1].split(': ')[1] + ' avg: ' + str(format(mean(data[1]), ".3f")) + ' std: ' + str(format(pstdev(data[1]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax1'].plot(X[2], data[2], 'tab:orange',
                                            label=list(measurements.keys())[2].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[2]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[2]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax1'].set_title(list(measurements.keys())[0].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax1'].legend()

    #wrong Setup A Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[3], data[3], 'tab:blue',
                                            label=list(measurements.keys())[3].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[3]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[3]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[4], data[4], 'tab:green',
                                            label=list(measurements.keys())[4].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[4]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[4]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax2'].plot(X[5], data[5], 'tab:orange',
                                            label=list(measurements.keys())[5].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[5]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[5]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax2'].set_title(list(measurements.keys())[3].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax2'].legend()

    #correct Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[6], data[6], 'tab:blue',
                                            label=list(measurements.keys())[6].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[6]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[6]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[7], data[7], 'tab:green',
                                            label=list(measurements.keys())[7].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[7]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[7]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax3'].plot(X[8], data[8], 'tab:orange',
                                            label=list(measurements.keys())[8].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[8]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[8]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax3'].set_title(list(measurements.keys())[6].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax3'].legend()

    #wrong Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[9], data[9], 'tab:blue',
                                            label=list(measurements.keys())[9].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[9]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[9]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[10], data[10], 'tab:green',
                                            label=list(measurements.keys())[10].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[10]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[10]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax4'].plot(X[11], data[11], 'tab:orange',
                                            label=list(measurements.keys())[11].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[11]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[11]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax4'].set_title(list(measurements.keys())[9].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax4'].legend()

    #inversed Setup B Scenario x: Test Bays B1,F1,F2
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[12], data[12], 'tab:blue',
                                            label=list(measurements.keys())[12].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[12]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[12]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[13], data[13], 'tab:green',
                                            label=list(measurements.keys())[13].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[13]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[13]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax5'].plot(X[14], data[14], 'tab:orange',
                                            label=list(measurements.keys())[14].split(': ')[1] + ' avg: ' + str(
                                                format(mean(data[14]), ".3f")) + ' std: ' + str(
                                                format(pstdev(data[14]), ".3f")))
    axs[str(len(fgs.keys())) + '_ax5'].set_title(list(measurements.keys())[12].split(': ')[0])
    axs[str(len(fgs.keys())) + '_ax5'].legend()

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
