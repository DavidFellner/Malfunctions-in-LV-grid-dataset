import os

import datetime
import pandas as pd
import matplotlib.pyplot as plt


def get_summed_curve(scenario_setup: pd.DataFrame):
    """
    1. Calc the sum of the loads where intelligent components are located.
    2. Calc the sum of the pvs where intelligent components are located.
    3. Calc the sum of both sums to get a global net demand
    :return:
    """

    local_dir = r"C:\REACT\react-hil\input"
    collab_dir = r"V:\flex_eve_lab\REACT\Daten_Grid"

    load_df = pd.read_csv(os.path.join(local_dir, "React_unbalanced_neu.csv"), index_col=0)

    start_dt = datetime.datetime(2018, 1, 1)
    end_dt = datetime.datetime(2019, 1, 1, 23, 45)

    dt_index = pd.date_range(start=start_dt, end=end_dt, freq="15T")

    load_df = load_df.set_index(dt_index)

    relevant_loads = ["lod_758", "lod_792"]

    for load in scenario_setup["PF_Load"]:

        split_load_name = load.split("_")

        if split_load_name[0] == "lod":
            relevant_loads.append(f"{split_load_name[0]}_{split_load_name[1]}")

    filter_columns_1 = [i + "_p1" for i in relevant_loads]
    filter_columns_2 = [i + "_p2" for i in relevant_loads]
    filter_columns_3 = [i + "_p3" for i in relevant_loads]

    all_filter_columns = filter_columns_1 + filter_columns_2 + filter_columns_3

    filtered_mv_loads = load_df.filter(all_filter_columns)  # length is

    if len(filtered_mv_loads.columns) < len(all_filter_columns):
        raise AssertionError("all_filter_columns has objects which are not part of load_df")
    sum_load_curve = filtered_mv_loads.sum(1)
    sum_load_curve.loc[:] *= (1000 / 4.)  # from MW to kW and divided by 4 due to Sawsans zone Scaling choice in powerfactory
    sum_load_curve = sum_load_curve["2018-01-01 00:00:00":"2018-12-31 23:00:00"]

    pv_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "input", "pv_profile.csv"))
    pv_size = 1  # pv_df is normalized to 2.5 kWp - remember to also change in sim_comp_controller!
    pv_count = scenario_setup["n_PV"].sum()
    sum_pv_curve = pv_df["ac_power_kW"].multiply(pv_size*pv_count)
    sum_pv_curve.index = sum_load_curve.index

    # calc curve for hil node Load 758_10
    hil_load = load_df.filter(["lod_758_p1","lod_758_p2","lod_758_p3"]).sum(1)
    hil_load.loc[:] *= (1000 / 64.)
    hil_load = hil_load["2018-01-01 00:00:00":"2018-12-31 23:00:00"]
    hil_pv = pv_df["ac_power_kW"].multiply(pv_size)
    hil_pv.index = hil_load.index
    hil_net_demand = - (hil_load + hil_pv)
    hil_net_demand.to_csv(os.path.join(os.path.dirname(__file__), "..", "input", "hil_net_demand_load_758_10.csv"))
    hil_load.to_csv(os.path.join(os.path.dirname(__file__), "..", "input", "hil_net_demand_load_758_10_no_PV.csv"))

    return -(sum_load_curve + sum_pv_curve)


def get_demand_power_plot(demand_curve):

    outdoor_temp = pd.read_csv("../input/outdoor_temp.csv", index_col=0, parse_dates=True)

    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    ax1.set_xlabel('datetime')
    demand_curve.plot(ax=ax1, color=color1)
    ax1.set_ylabel('power in kW', color=color1)

    color2 = "tab:red"
    ax2 = ax1.twinx()
    outdoor_temp.plot(ax=ax2, color=color2)
    ax2.set_ylabel("temp in C", color=color2)

    fig.tight_layout()

    plt.show()

    print("bp")

if __name__ == "__main__":

    scenario_setup = pd.read_csv(f"../input/scenarios/scenario1.csv")
    net_demand = get_summed_curve(scenario_setup)

    get_demand_power_plot(net_demand)

    scenarios = pd.read_excel("../scenarios.xlsx")

    for scenario in scenarios["scenario_name"]:
        scenario_setup = pd.read_csv(f"../input/scenarios/{scenario}.csv")
        net_demand = get_summed_curve(scenario_setup)

        # 14th July is a good day for our simulations
        # No! We need a day with low temperatures - otherwise the HP don't show an effect!
        # 30th of March is a good day
        print("finished", scenario)
