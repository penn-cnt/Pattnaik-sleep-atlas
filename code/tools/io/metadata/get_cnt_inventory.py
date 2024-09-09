# %%
from os.path import join as ospj
import pandas as pd
import sys

sys.path.append(ospj("..", "..", ".."))
from tools.config import Paths

# %%
def get_cnt_inventory(bids_inventory=Paths.bids_inventory_file):
    inventory = pd.read_csv(bids_inventory, index_col=0)
    inventory = inventory == "yes"
    return inventory

# %%
