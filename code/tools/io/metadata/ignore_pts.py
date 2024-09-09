#%%
"""
This script loads and returns the list of patients to ignore from the coherence analysis.
"""
import os
from os.path import join as ospj
import sys
import pandas as pd

sys.path.append(ospj("..", "..", ".."))
from tools.config import Paths

# %%
def ignore_pts():
    table = pd.read_excel(ospj(Paths.data_dir, "metadata", "nishant_ignore_pts.xlsx"))
    table = table[['record_id', 'ignore']]
    # drop rows with all NaNs
    table = table.dropna(how="any")
    # drop columns with all NaNs
    table = table.dropna(axis=1, how="all")

    table['record_id'] = table.record_id.apply(lambda x: f"sub-RID{str(int(x)).zfill(4)}")

    return table[table.ignore == 1].record_id.values

# %%
