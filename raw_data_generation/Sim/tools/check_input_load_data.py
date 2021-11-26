import os
import pandas as pd

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", 'input/pf_load_profiles_unbalanced.csv'), index_col=0)

df["total_consumption"] = df.sum(axis=1)