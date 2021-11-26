import os
import json
import importlib

import numpy as np
import pandas as pd

import pflib.pf as pf  # We do not need to mess around with the path - PFLib does this for us
import pflib.object_frame as pof

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)

np.random.seed(1337)


def prepare_loads_df(grid):
    with open(os.path.join(os.path.dirname(__file__), "..", "config.json"), "r") as f:
        intelligent_loads = json.load(f)["intelligent_pf_loads"]

    #old
    '''project_folder = r"V:\flex_eve_lab\REACT"
    local_data_folder = os.path.join(os.path.dirname(__file__), "..", "input_react", "local_project_data")
    # filepath = os.path.join(project_folder, "Innis Mor Info", "Loads.xlsx")
    filepath = os.path.join(local_data_folder, "Loads.xlsx")
    print("load excel: ", filepath)
    loads_df_orig = pd.read_excel(filepath, sheet_name="Loads")'''

    #simbench
    local_data_folder = os.path.join(configuration.grid_data_folder, grid.split('.')[0])
    filepath = os.path.join(local_data_folder, "Load.csv")
    loads_df = pd.read_csv(filepath, delimiter=';')

    #old
    '''# loads_mapping_df = pd.read_excel(os.path.join(project_folder, "Daten_Grid", "node_line_mapping.xlsx"),
    loads_mapping_df = pd.read_excel(os.path.join(local_data_folder, "node_line_mapping.xlsx"),
                                     sheet_name="Load mapping")

    loads_df["NodeId"] = loads_mapping_df["NodeId"]
    loads_df["PF_Load"] = loads_mapping_df["PF_Load"]
    loads_df = loads_df.dropna(subset=["PF_Load"])
    loads_df["PF_Load"] = ["lod_" + str(round(el)) + "_1" for el in loads_df["PF_Load"]]
    loads_df["Total Customers"] = loads_df["Phase1Customers"] + loads_df["Phase2Customers"] + loads_df[
        "Phase3Customers"]
    loads_df["Total Customers"] = loads_df["Total Customers"].round().astype(int)
    loads_df.set_index("SectionId", inplace=True)
    loads_df = loads_df.loc[loads_df["PF_Load"].isin(intelligent_loads)]
    loads_df = loads_df.drop_duplicates(subset=["PF_Load"])'''

    return loads_df


def get_phase_config(n_p1, n_p2, n_p3):
    if (n_p1 == n_p2 == 0.) and (n_p3 != 0.):

        return "plinit"

    elif (n_p2 == n_p3 == 0.) and (n_p1 != 0.):

        return "plinir"

    elif (n_p1 == n_p3 == 0.) and (n_p2 != 0.):

        return "plinis"

    elif (n_p1 == n_p2 == n_p3 != 0.):

        return "plinir_plinis_plinit"

    else:
        return "invalid phase config"


def start_pf():
    pf.start()
    pf.app.Hide()
    with open(os.path.join(os.path.dirname(__file__), "..", "config.json"), "r") as f:
        grid_file = json.load(f)["pf_controller"]["grid_file"]
    err = pf.app.ActivateProject(grid_file.split(".")[0])
    pf.PFException.raise_on_error(err, "Unable to activate the test project")


def create_pf_load_elements(loads_df, mv_loads=True):
    start_pf()
    if mv_loads:
        for sec_id, load in loads_df.iterrows():
            node = load["PF_Load"].split("_")[1] + "(1)"
            elements = (pf.create_ElmLod("PV_" + load["PF_Load"].split("_")[1], term_obj_or_str=node),
                        pf.create_ElmLod("HP_" + load["PF_Load"].split("_")[1], term_obj_or_str=node),
                        pf.create_ElmLod("BESS_" + load["PF_Load"].split("_")[1], term_obj_or_str=node))
            for el in elements:
                el.i_sym = 1
    else:
        for sec_id, load in loads_df.iterrows():
            node = load["Terminal"]
            elements = (pf.create_ElmLod("PV_" + load["Load"].split(" ")[1], term_obj_or_str=node),
                        pf.create_ElmLod("HP_" + load["Load"].split(" ")[1], term_obj_or_str=node),
                        pf.create_ElmLod("BESS_" + load["Load"].split(" ")[1], term_obj_or_str=node))
            for el in elements:
                el.i_sym = 1


def extend_loads_df_lv(loads_df, lv_loads_df):
    row_792_id = 500065380
    row_758_id = 506184868

    row_792 = loads_df.loc[row_792_id]
    row_758 = loads_df.loc[row_758_id]

    for row in lv_loads_df.iterrows():

        pf_load_name = row[1]["Load"]

        if "Load 792" in pf_load_name:
            row_id = row_792_id
            new_row = row_792.copy(deep=True)
            ref_row = row_792.copy(deep=True)

        elif "Load 758" in pf_load_name:
            row_id = row_758_id
            new_row = row_758.copy(deep=True)
            ref_row = row_758.copy(deep=True)
        else:
            raise ValueError("Neither of the required loads that have LV connections are in the Load str. check input!")

        new_row["PF_Load"] = pf_load_name
        new_row["Total Customers"] = 1

        for phase in [1, 2, 3]:

            if ref_row[f"Phase{phase}Customers"] != 0:
                new_row[f"Phase{phase}Kwh"] = ref_row[f"Phase{phase}Kwh"] / ref_row[f"Phase{phase}Customers"]
                new_row[f"Phase{phase}Customers"] = ref_row[f"Phase{phase}Customers"] / ref_row[
                    f"Phase{phase}Customers"]

            else:
                new_row[f"Phase{phase}Kwh"] = 0.0
                new_row[f"Phase{phase}Customers"] = 0.0

        loads_df.loc[int(f"{row_id}{pf_load_name.split('_')[1]}")] = new_row

    loads_df = loads_df.drop([row_758_id, row_792_id])

    return loads_df


"""def distribute_intelligent_components(loads_df, scen, out_dir):
    n_PV = list()
    n_HP = list()
    n_BESS = list()
    phase_config = list()
    for sec_id, load in loads_df.iterrows():
        n_PV.append(np.random.binomial(load["Total Customers"], scen["share_PV"]))
        n_HP.append(np.random.binomial(load["Total Customers"], scen["share_HP"]))
        n_BESS.append(np.random.binomial(load["Total Customers"], scen["share_BESS"]))
        phase_config.append(get_phase_config(
            load["Phase1Customers"],
            load["Phase2Customers"],
            load["Phase3Customers"]))
    loads_df["n_PV"] = n_PV
    loads_df["n_HP"] = n_HP
    loads_df["n_BESS"] = n_BESS
    loads_df["phase_config"] = phase_config

    all_customers = sum([load["Total Customers"] for sec_id, load in loads_df.iterrows()])
    print("Distribution of intelligent components for", scen["scenario_name"])
    print("resulting share of PV:", round(sum(n_PV) / all_customers, 2))
    print("resulting share of HP:", round(sum(n_HP) / all_customers, 2))
    print("resulting share of BESS:", round(sum(n_BESS) / all_customers, 2))
    loads_df.to_csv(os.path.join(out_dir, scen["scenario_name"] + ".csv"))"""


# def _DRAFT_distribute_intelligent_components(loads_df, scen, out_dir):
#     all_customers = sum([load["Total Customers"] for sec_id, load in loads_df.iterrows()])
#     count_PV = round(scen["share_PV"] * all_customers)
#     count_HP = round(scen["share_HP"] * all_customers)
#     count_BESS = round(scen["share_BESS"] * all_customers)
#
#     customers = []
#     for sec_id, load in loads_df.iterrows():
#         customers += [load["PF_Load"] + "_" + str(i) for i in range(load["Total Customers"])]
#
#     np.random.shuffle(customers)
#     PV_customers = list()
#     for i in range(count_PV):
#         PV_customers.append(customers[i])
#         # unfinished
#     print("done")


def main(grid):
    loads_df = prepare_loads_df(grid)

    # create_pf_load_elements(loads_df)

    #old
    #lv_loads_df = pd.read_csv("../input_react/lv_load_map.csv")

    # create_pf_load_elements(lv_loads_df, mv_loads=False)

    #loads_df = extend_loads_df_lv(loads_df, lv_loads_df)

    '''out_dir = os.path.join(os.path.dirname(__file__), "..", "input_react", "scenarios")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filelist = os.listdir(out_dir)
    for item in filelist:
        if item.endswith(".csv"):
            os.remove(os.path.join(out_dir, item))'''

    """scenarios_df = pd.read_excel("../scenarios.xlsx")
    for idx, scen in scenarios_df.iterrows():
        distribute_intelligent_components(loads_df, scen, out_dir)

    print("Distribution of intelligent loads is finished")"""

    return loads_df


if __name__ == "__main__":
    main()
