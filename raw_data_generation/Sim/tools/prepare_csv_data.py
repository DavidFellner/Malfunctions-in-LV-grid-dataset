import pandas as pd
import os
import datetime


t_start = datetime.datetime(2017, 6, 5)
t_end = datetime.datetime(2017, 6, 6)

input_df = pd.read_csv(r"C:\Users\ReihsD\Documents\Projekte\REACT\P_HH5min.csv")
input_df.fillna(0.0)

input_df["datetime"] = pd.to_datetime(input_df["datetime"])
# input_df = input_df.set_index("datetime")

sim_df = input_df[(input_df.datetime >= t_start) & (input_df.datetime <= t_end)]
sim_df = sim_df.set_index("datetime")
sim_df = sim_df.resample("T").interpolate()

int_index = pd.RangeIndex(0, len(sim_df.index), step=1)
sim_df = sim_df.set_index(int_index)

profile_file_name = "load_profiles.csv"
dir_path = os.path.dirname(__file__)
path_to_save = os.path.join(dir_path, "..", "input", profile_file_name)

sim_df.to_csv(path_to_save)

