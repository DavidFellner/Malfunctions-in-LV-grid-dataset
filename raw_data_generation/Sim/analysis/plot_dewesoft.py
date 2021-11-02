import os

from datetime import datetime
import matplotlib.pyplot as plt

import dwdatareader as dw
import pandas as pd


def main(data_dir, dewe_fn, control_curve_fn, dewe_channels: tuple or list, figname, ylabel,
         control_curve_inverted=False, start_sec=None, end_sec=None, save_data_too=True):
    """  -------------- READ THIS! --------------
    :param dewe_channels: Easiest way to determine the correct channel names is opening the dewe file with Dewesoft X,
        plotting the required channels, and then (IMPORTANT!) renaming them to unique names in the view at the right.
        Reason: the power channels have the same names by default (e.g P_L1) and renaming then enables access from here.
        If you put a - at the beginning of the channel string, the channel plot will be inverted.
    """
    out_dir = os.path.join(os.path.dirname(__file__), 'data', "dewesoft")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if control_curve_fn:
        control_curve = pd.read_csv(os.path.join(data_dir, control_curve_fn), index_col=0, parse_dates=True)
    else:
        control_curve = None
    with dw.open(os.path.join(data_dir, dewe_fn)) as df:
        # for ch in df.values():
        #     print(ch.name)
        datasets = {}
        for channel in dewe_channels:
            if channel[0] == "-":  # minus! invert variable
                sr = df[channel[1:]].series()
                sr *= -1
                datasets[channel[1:]] = sr
            else:
                datasets[channel] = df[channel].series()

    lab_start = 32.0
    lab_end = lab_start + 86400.0

    sim_start_dt = datetime(2018, 3, 3, 0)
    sim_end_dt = datetime(2018, 3, 4, 0)

    assert (sim_end_dt - sim_start_dt).total_seconds() == 86400.0

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Test time in seconds')
    ax1.set_ylabel(ylabel)

    for channel, dataset in datasets.items():
        label = f"measured {channel}"
        if start_sec and end_sec:
            dataset[start_sec:end_sec].plot(ax=ax1, label=label)
        else:
            dataset.plot(ax=ax1, label=label)
        if save_data_too:
            if not os.path.isdir(os.path.join(out_dir, figname)):
                os.makedirs(os.path.join(out_dir, figname))
            dataset.to_csv(os.path.join(out_dir, figname, label + ".csv"))

    if control_curve_fn:
        sim_control_curve = control_curve[sim_start_dt:sim_end_dt]
        if control_curve_inverted:
            sim_control_curve *= -1
        sim_control_curve = sim_control_curve.set_index(sim_control_curve.index.map(seconds_since))
        sim_control_curve = sim_control_curve["0"].rename("control_curve")
        if start_sec and end_sec:
            sim_control_curve[start_sec, end_sec].plot(ax=ax1, label="control curve")
        else:
            sim_control_curve.plot(ax=ax1, label="control curve")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, figname + "_lab_test.png"))
    print("finished", figname)


def seconds_since(dt, start=datetime(2018, 3, 3, 0)):
    return (dt - start).total_seconds()

def plot_dewetron_temp(data_dir, figname):
    out_dir = os.path.join(os.path.dirname(__file__), 'data', "dewesoft")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filename = "REACT_Temp_HP20210212_175552.dmd_export_20210215_112903.csv"
    df = pd.read_csv(os.path.join(data_dir, filename), index_col=0, parse_dates=True)
    df = df.resample("1S").fillna("nearest")
    sim_start_dt = datetime(2021, 2, 12, 20, 45)
    sim_end_dt = datetime(2021, 2, 13, 20, 45)
    df = df[sim_start_dt:sim_end_dt]
    df.index = (df.index - sim_start_dt).total_seconds()
    fig, ax1 = plt.subplots()

    for col in df:
        plt.plot(df[col], label=col)

    plt.grid()
    plt.tight_layout()
    plt.legend(ncol=2)

    plt.savefig(os.path.join(out_dir, figname + ".png"))


if __name__ == '__main__':
    base_dir = r"V:\flex_eve_lab\REACT\Results"  # ToDo: keep local and remote directory in sync
    #base_dir = r"C:\Users\StahlederD\Desktop\REACT\Results"

    plot_dewetron_temp(data_dir=os.path.join(base_dir, "MEL_24h"), figname="HP_all_temps")

    main(data_dir=os.path.join(base_dir, "MIDAC_24h"),
         dewe_fn="20210127_MIDAC_24h_v2.d7d",
         control_curve_fn="hil_net_demand_load_758_10.csv",
         dewe_channels=("P_L1_GRID",),
         figname="midac",
         ylabel="Active power in kW",
         control_curve_inverted=True)

    main(data_dir=os.path.join(base_dir, "MIDAC_24h"),
         dewe_fn="20210127_MIDAC_24h_v2.d7d",
         control_curve_fn=None,
         dewe_channels=("P_L1_GRID", "-Q_L1_GRID"),
         figname="midac_PQ_analysis",
         ylabel="Power in kW/kVAr")

    main(data_dir=os.path.join(base_dir, "Victron_24h"),
         dewe_fn="20210204_VIC_24h.d7d",
         control_curve_fn="hil_net_demand_load_758_10_no_PV.csv",
         dewe_channels=("P_PV", "P_L1_GRID", "P_BESS_out"),
         figname="victron",
         ylabel="Active power in kW",
         control_curve_inverted=False)

    main(data_dir=os.path.join(base_dir, "Victron_24h"),
         dewe_fn="20210204_VIC_24h.d7d",
         control_curve_fn=None,
         dewe_channels=("P_L1_GRID", "-Q_L1_GRID"),
         figname="victron_PQ_analysis",
         ylabel="Power in kW/kVAr")

    main(data_dir=os.path.join(base_dir, "Victron_24h"),
         dewe_fn="20210204_VIC_24h.d7d",
         control_curve_fn=None,
         dewe_channels=("P_L1_GRID", "-Q_L1_GRID"),
         figname="victron_PQ_analysis_zoom",
         ylabel="Power in kW/kVAr",
         start_sec=29000, end_sec=33500)

    main(data_dir=os.path.join(base_dir, "MEL_24h"),
         dewe_fn="20210212_MEL_24hTest.d7d",
         control_curve_fn=None,
         dewe_channels=("P_L1_GRID",),
         figname="mel",
         ylabel="Active power in kW",
         control_curve_inverted=True)

    main(data_dir=os.path.join(base_dir, "MEL_24h"),
         dewe_fn="20210212_MEL_24hTest.d7d",
         control_curve_fn=None,
         dewe_channels=("P_L1_GRID", "-Q_L1_GRID"),
         figname="mel_PQ_analysis",
         ylabel="Power in kW/kVAr")
