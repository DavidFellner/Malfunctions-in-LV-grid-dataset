import os
import random

import pandas as pd


load_profile_loc = "C:\\Users\\ReihsD\\Documents\\Projekte\\REACT\\1-complete_data-mixed-all-2-sw\\"
load_profile_file = "LoadProfile.csv"

load_profiles_df = pd.read_csv(os.path.join(load_profile_loc, load_profile_file), sep=";")

# get phase load profiles

phase_load_profile = {}
phases = ["A", "B", "C"]

for phase in phases:

    phase_load_profile[phase] = load_profiles_df[f"H0-{phase}_pload"]
    phase_load_profile[f"{phase}_norm"] = load_profiles_df[f"H0-{phase}_pload"] / load_profiles_df[f"H0-{phase}_pload"].sum()

# normalize the phase load profiles

phase_load_profile["ABC"] = (phase_load_profile["A"] + phase_load_profile["B"] + phase_load_profile["C"])
phase_load_profile["ABC_norm"] = phase_load_profile["ABC"]/phase_load_profile["ABC"].sum()

# get db to pf load mapping

pf_db_mapping_loc = "C:\\Users\\ReihsD\\Documents\\Projekte\\REACT\\Daten_Grid"
pf_db_mapping = "db_pf_load_mapping.csv"

pf_db_mapping_df = pd.read_csv(os.path.join(pf_db_mapping_loc, pf_db_mapping))
pf_db_map = dict()

for row in pf_db_mapping_df.iterrows():
    pf_db_map[int(row[1]['SectionId_load'])] = int(row[1]['PF_Load'])
    # print(f"db load: {int(row[1]['SectionId_load'])}, pf load: {int(row[1]['PF_Load'])}")

# get db yearly consumptions to scale load profiles

db_load_loc = "C:\\Users\\ReihsD\\Documents\\Projekte\\REACT\\grid_data_aran\\Innis Mor Info"
db_load_file = "Loads.xlsx"

db_load_df = pd.read_excel(os.path.join(db_load_loc, db_load_file))

# iterate over loads and set result df columns

final_load_df = pd.DataFrame(index=load_profiles_df.index)

for row in db_load_df.iterrows():
    name = row[1]["SectionId"]
    phase_cons_year = {
        "plinir": row[1]["Phase1Kwh"],
        "plinis": row[1]["Phase2Kwh"],
        "plinit": row[1]["Phase3Kwh"]
    }
    # p1_yearly_cons = row[1]["Phase1Kwh"]
    # p2_yearly_cons = row[1]["Phase2Kwh"]
    # p3_yearly_cons = row[1]["Phase3Kwh"]

    phase_choice = ["A_norm", "B_norm", "C_norm"]
    random.shuffle(phase_choice)

    if name != 506288519: # not a real load - db value has all zeros seems to be dummy, remove it or next step breaks

        for index, phase in enumerate(["plinir", "plinis", "plinit"]):
            final_load_df[f"lod_{pf_db_map[name]}_1_{phase}"] = phase_load_profile["ABC_norm"].multiply(
                phase_cons_year[phase] * 4)
            # quarter hour values means sum over all the values should give 4 times the yearly consumption

print(final_load_df)

final_load_df.to_csv("input\\pf_load_profiles_unbalanced.csv")

# phase_choice = random.shuffle(["A_norm", "B_norm", "C_norm"])




print("debug")