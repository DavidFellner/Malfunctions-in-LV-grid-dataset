import os

import pandas as pd
import matplotlib as mpl
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np


improved_labels = {
    "trafo_loading": "Transformer Loading (%)",
    "line_loading": "Cable Loading (%)",
    "voltage": "Node voltage (p.u.)",
    "lv_voltage": "Node voltages of LV nodes (p.u.)",
    "mv_voltage": "Node voltages of MV nodes (p.u.)",
    "power": "Active Power (kW)",
    "hh": "Active Power (kW)",
    "hp": "Active Power (kW)",
    "bess": "Active Power (kW)",
    "pv": "Active Power (kW)",
}


def improve_label(var_type):
    try:
        return improved_labels[var_type]
    except KeyError:
        return var_type


def get_results(res_dir, systems: list, scenarios: list, scenario_loads: dict, local: bool):
    results = dict()
    for system in systems:
        sys_res = dict()
        for scenario in scenarios:
            sys_res[scenario] = dict()
            if local:
                scen_path = os.path.join(res_dir, scenario)
            else:
                scen_path = os.path.join(res_dir, system, scenario)
            df = pd.read_csv(os.path.join(scen_path, f"{scenario}_Grid_Voltages.csv"), index_col=0, parse_dates=True)
            relevant_nodes = pd.read_csv(os.path.join(os.path.dirname(__file__), "relevant_nodes.csv"))["Terminal"].to_list()  # Only Aran Island shall be plotted. node list has been extracted from the PowerFactory UI
            col_names = []
            for el in relevant_nodes:
                col_names += [f"('{el}', 'L1')", f"('{el}', 'L2')", f"('{el}', 'L3')"]
            sys_res[scenario]["voltage"] = df[col_names]
            sys_res[scenario]["temp"] = pd.read_csv(os.path.join(scen_path, f"{scenario}_HP_Temps.csv"),
                                                            index_col=0, parse_dates=True)
            sys_res[scenario]["power"] = pd.read_csv(os.path.join(scen_path, f"{scenario}_Sim_Powers.csv"),
                                                             index_col=0, parse_dates=True)
            sys_res[scenario]["trafo_loading"] = pd.read_csv(os.path.join(scen_path, f"{scenario}_Trafo_Loadings.csv"),
                                                             index_col=0, parse_dates=True)
            sys_res[scenario]["line_loading"] = pd.read_csv(os.path.join(scen_path, f"{scenario}_Cable_Loadings.csv"),
                                                             index_col=0, parse_dates=True)
            sys_res[scenario]["soc"] = pd.read_csv(os.path.join(scen_path, f"{scenario}_BESS_SoC.csv"),
                                                             index_col=0, parse_dates=True)
            smartest_load = ["('smartest_lab', 'L1')", "('smartest_lab', 'L2')", "('smartest_lab', 'L3')"]
            sys_res[scenario]["power"] = sys_res[scenario]["power"][
                get_columns_from_load_names(scenario_loads[scenario]["all"]) + smartest_load]
            for type in scenario_loads[scenario].keys():
                sys_res[scenario][type] = sys_res[scenario]["power"][
                    get_columns_from_load_names(scenario_loads[scenario][type])]
        results[system] = sys_res

    return results


def _fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def make_cumulative_plot(df,
                         additional_search_substring=None,
                         scenario_name="",
                         legend=True,
                         x_label="Cable Loading in %",
                         xlims=(0, 100),
                         plot_labels=None):

    # df[df.columns[0]].plot()
    # plt.show()
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plt.grid()
    for i, (col_name, col) in enumerate(df.iteritems()):
        if plot_labels:
            label = plot_labels[i]
        else:
            try:
                idx = col_name[0].find(additional_search_substring)
                label = col_name[0][idx + 1:]
            except TypeError:
                label = None
        format = "-"  # if i < 10 else "-."  # ToDo: change back for substation-specific plots (not for all element plots)
        n, bins, patches = ax.hist(col.dropna(), 100, density=True, histtype='step',
                                   cumulative=True, label=label, linestyle=format)
    lgd = ax.legend()  # weird workaround to remove duplicate legend which was placed due to some reason
    if lgd: lgd.remove()
    if legend:
            lgd = fig.legend(fontsize="small", frameon=False, ncol=2)
    _fix_hist_step_vertical_line_at_end(ax)
    plt.yticks(np.linspace(0.0, 1.0, 11))
    y_value = [str(100 * round(x, 2)) + '%' for x in ax.get_yticks()]
    ax.set_yticklabels(y_value)
    if xlims:
        ax.set_xlim(xlims)
    ax.set_ylabel("Amount of values")
    ax.set_xlabel(x_label)

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig", scenario_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filepath = os.path.join(out_dir, f"{scenario_name}_{x_label}.png")
    fig.savefig(filepath, dpi=300)


def make_single_element_plot(results, systems, scenarios,
                             var_type="voltage", phases=("L1", "L2", "L3"), pf_el="111"):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time of day')

    for system in systems:
        for scenario in scenarios:
            if phases:
                for phase in phases:
                    to_plot = results[system][scenario][var_type][f"('{pf_el}', '{phase}')"]
                    to_plot.plot(ax=ax1, label=f"{system}_{scenario}_{phase}")
            else:
                try:
                    values = [results[system][scenario][var_type][f"('{pf_el}', '{ph}')"] for ph in ("L1", "L2", "L3")]
                    if var_type == "voltage":
                        avg_volts = sum(values) / 3.
                        avg_volts.plot(ax=ax1, label=f"{system}_{scenario}")
                    else:
                        sum(values).plot(ax=ax1, label=f"{system}_{scenario}")
                except KeyError:  # some variables don't have phases (e.g. loadings)
                    results[system][scenario][var_type][pf_el].plot(ax=ax1, label=f"{system}_{scenario}")
    plt.tight_layout()
    ax1.set_ylabel(improve_label(var_type))
    ax1.xaxis.set(major_formatter=md.DateFormatter('%H:%M'), major_locator=md.HourLocator(range(0, 24, 3)))
    plt.legend()
    if len(systems) > 1:
        out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig")
        filename = f"{pf_el}_{var_type}.png"
    else:
        out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig", systems[0])
        filename = f"{systems[0]}_{pf_el}_{var_type}.png"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    # plt.show()

    # print("bp")

def make_scenario_comparison_plot(results, systems, scenarios, pf_elements,
                             var_type="voltage", phases=("L1", "L2", "L3")):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time of day')

    for system in systems:
        if system == "sim_local_opt":
            plot_label = "Local self consumption optimization"
        elif system == "sim_p_of_u_with_setp":
            plot_label = "Local self consumption optimization with droop control"
        else:
            plot_label = ""

        for scenario in scenarios:
            if phases:
                results[system][scenario][var_type][[f"('{pf_el}', '{phase}')" for pf_el in pf_elements for phase in phases]].sum(1).plot(ax=ax1, label=plot_label)
            else:
                try:
                    values = results[system][scenario][var_type][[f"('{pf_el}', '{ph}')" for ph in ("L1", "L2", "L3") for pf_el in pf_elements]]
                    if var_type == "voltage":
                        avg_volts = sum(values) / 3.
                        avg_volts.plot(ax=ax1, label=f"{system}_{scenario}")
                    else:
                        sum(values).plot(ax=ax1, label=f"{system}_{scenario}")
                except KeyError:  # some variables don't have phases (e.g. loadings)
                    raise
    plt.tight_layout()
    ax1.set_ylabel(improve_label(var_type))
    ax1.xaxis.set(major_formatter=md.DateFormatter('%H:%M'), major_locator=md.HourLocator(range(0, 24, 3)))
    plt.legend()
    if len(systems) > 1:
        out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig")
        filename = f"batt_scen_comp_{var_type}.png"
    else:
        out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig", systems[0])
        filename = f"{systems[0]}_batt_comp_{var_type}.png"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    # plt.show()

    # print("bp")

def make_multiple_elements_plot(results, system, scenario, var_type, phases):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time of day')

    for pf_el, data in results[system][scenario][var_type].iteritems():
        ax1.plot(data)
    plt.tight_layout()
    ax1.set_ylabel(improve_label(var_type))
    ax1.xaxis.set(major_formatter=md.DateFormatter('%H:%M'), major_locator=md.HourLocator(range(0, 24, 3)))
    out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig", system)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, f"{system}_mult_el_{var_type}.png"), bbox_inches="tight")


def make_overall_power_plot(results, systems, scenarios):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time of day')
    ax1.set_ylabel("active power in kW")

    for system in systems:
        for scenario in scenarios:
            # for phase in phases:
            results[system][scenario]["power"].sum(1).plot(ax=ax1,
                                                           label=f"{system}_{scenario}")  # _{phase}")

    plt.legend()
    # plt.show()
    ax1.xaxis.set(major_formatter=md.DateFormatter('%H:%M'), major_locator=md.HourLocator(range(0, 24, 3)))
    out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, f"overall_power_plot.png"), bbox_inches="tight")

    # print("bp")


def make_scen_power_type_plot(scenario_results, type_subsets, system, scenario):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time of day')
    ax1.set_ylabel("active power in kW")

    for name, columns in type_subsets.items():
        # for phase in phases:
        scenario_results["power"][get_columns_from_load_names(columns)].sum(1).plot(ax=ax1,
                                                                                            label=f"{name}")  # _{phase}")

    plt.legend()
    # plt.show()
    plt.tight_layout()
    ax1.xaxis.set(major_formatter=md.DateFormatter('%H:%M'), major_locator=md.HourLocator(range(0, 24, 3)))
    out_dir = os.path.join(os.path.dirname(__file__), 'data', "fig", system)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, f"all_loads_{system}_{scenario}.png"), bbox_inches="tight")

    # print("bp")


def get_scen_relevant_loads(input_params, phases=("L1", "L2", "L3")):
    hh_loads = []
    bess = []
    hp = []
    pv = []

    for _, row in input_params.iterrows():
        pf_load = row["PF_Load"]
        if " " in pf_load:
            load_name = pf_load
            load_nr = pf_load.split(" ")[1]

        else:
            split_load_name = pf_load.split("_")
            load_name = f"{split_load_name[0]}_{split_load_name[1]}"
            load_nr = split_load_name[1]

        hh_loads.append(load_name)
        bess.append(f"BESS_{load_nr}")
        hp.append(f"HP_{load_nr}")
        pv.append(f"PV_{load_nr}")

    all_loads = hh_loads + bess + hp + pv

    return {"all": all_loads, "hh": hh_loads, "bess": bess, "hp": hp, "pv": pv}


def get_columns_from_load_names(load_names, phases=("L1", "L2", "L3")):
    ret_list = []

    for load in load_names:
        for phase in phases:
            ret_list.append(f"('{load}', '{phase}')")

    return ret_list


if __name__ == '__main__':
    MV_part_connecting_node = "111"
    connecting_line = "lne_113_111_1"

    # LV_nodes = pd.read_csv()

    # local_res_dir = os.path.join(os.path.dirname(__file__), "..", "output", "scenarios")
    remote_res_dir = r"V:\flex_eve_lab\REACT\Results"
    # remote_res_dir = r"C:\Users\StahlederD\Desktop\REACT\Results"
    systems = ["MIDAC_24h",
               "Victron_24h",
               "MEL_24h",
               "sim",
               "sim_local_opt",
               "sim_p_of_u",
               "sim_p_of_u_with_setp"]
    scenarios = ["scenario1"]

    input_params = {
        scenario: pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "input", "scenarios", f"{scenario}.csv"))
        for scenario in scenarios}

    scenario_loads = {scenario: get_scen_relevant_loads(input_params[scenario]) for scenario in scenarios}

    # scenario = "scenario1"

    results = get_results(remote_res_dir, systems, scenarios, scenario_loads, local=False)
    # format of results: results[system_type][scenario]

    # fix broken p.u. voltages (factor 244.219/230.940)
    lv_names = None
    for system in systems:
        for scenario in scenarios:
            node_voltages = results[system][scenario]['voltage']
            node_names = node_voltages.columns
            lv_names = [el for el in node_names if " LV " in el or "(1)" in el]
            node_voltages[lv_names] *= (244.21916386721169838737/230.94010767585030580366)

    # compare single elements for all systems and scenarios in one plot
    make_single_element_plot(results, systems, scenarios, var_type="voltage", phases=None, pf_el="Urban LV B46(1)")
    make_single_element_plot(results, systems, scenarios, var_type="power", phases=None, pf_el="smartest_lab")
    make_single_element_plot(results, systems, scenarios, var_type="power", phases=None, pf_el="BESS_758_10")
    make_single_element_plot(results, systems, scenarios, var_type="power", phases=None, pf_el="PV_758_10")
    make_single_element_plot(results, systems, scenarios, var_type="power", phases=None, pf_el="HP_758_10")
    make_single_element_plot(results, systems, scenarios, var_type="trafo_loading", phases=None, pf_el="tr_758")
    make_single_element_plot(results, systems, scenarios, var_type="line_loading", phases=None, pf_el="Urban LV line B46(1)")

    make_scenario_comparison_plot(results, ["sim_local_opt", "sim_p_of_u_with_setp"], scenarios, var_type="power",
                                  pf_elements=scenario_loads["scenario1"]["bess"])

    make_overall_power_plot(results, systems, scenarios)
    for system in systems:
        for scenario in scenarios:
            for type in ["voltage", "trafo_loading", "line_loading", "hh", "bess", "hp", "pv"]:
                make_multiple_elements_plot(results, system, scenario, var_type=type, phases=None)

            # here this function is only used to plot one system and scenario at once
            make_single_element_plot(results, [system], [scenario], var_type="voltage", phases=None, pf_el="Urban LV B46(1)")
            make_single_element_plot(results, [system], [scenario], var_type="power", phases=None, pf_el="smartest_lab")
            make_single_element_plot(results, [system], [scenario], var_type="trafo_loading", phases=None, pf_el="tr_758")
            make_single_element_plot(results, [system], [scenario], var_type="line_loading", phases=None, pf_el="Urban LV line B46(1)")

            make_scen_power_type_plot(results[system][scenario], scenario_loads[scenario], system, scenario)
            make_cumulative_plot(results[system][scenario]["line_loading"],
                                 scenario_name=system,
                                 x_label=improve_label("line_loading"),
                                 legend=False)
            make_cumulative_plot(results[system][scenario]["trafo_loading"],
                                 scenario_name=system,
                                 x_label=improve_label("trafo_loading"),
                                 legend=False)
            make_cumulative_plot(results[system][scenario]["voltage"][lv_names],
                                 scenario_name=system,
                                 x_label=improve_label("lv_voltage"),
                                 xlims=(0.97, 1.01),
                                 legend=False)
            mv_names = [name for name in results[system][scenario]["voltage"].columns if name not in lv_names]
            make_cumulative_plot(results[system][scenario]["voltage"][mv_names],
                                 scenario_name=system,
                                 x_label=improve_label("mv_voltage"),
                                 xlims=(0.97, 1.01),
                                 legend=False)

            # all, hh, bess, hp, pv
            for type in ["hh", "bess", "hp", "pv"]:
                make_cumulative_plot(results[system][scenario][type],
                                     scenario_name=system,
                                     x_label=type + "_Active Power (kW)",
                                     xlims=None,
                                     legend=False)