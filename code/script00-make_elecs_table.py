# %%
# imports
import os
from os.path import join as ospj
from glob import glob
import argparse
import sys

import numpy as np
# from utils import *
import nibabel as nib
import scipy.io as sio
from tqdm import tqdm
import pandas as pd

from tools.config import CONFIG

pd.set_option("display.max_rows", 1000)

# from config import CONFIG
# from utils import get_cnt_inventory
from tools.io.metadata import get_cnt_inventory
from tools.io.metadata import label_fix, clean_labels
from tools.io.metadata.get_metadata import get_outcome

inventory = get_cnt_inventory()
inventory = inventory[inventory.ieeg_recon]

# argparse
is_interactive = hasattr(sys, "ps1")
if not is_interactive:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

# %%
print("Getting all coords...")
all_data = None
for rid in tqdm(inventory.index, desc="rid", leave=True):
    ieeg_recon_path = ospj(
        CONFIG.paths.bids_dir,
        rid,
        "derivatives",
        "ieeg_recon",
    )
    if not os.path.exists(ieeg_recon_path):
        ieeg_recon_path = ieeg_recon_path.replace("ieeg_recon", "ieeg_recon_old")
        
    # get mni coordinates
    try:
        mni_path = glob(
            ospj(
                ieeg_recon_path,
                "module3",
                "MNI",
                "*vox_electrodes.txt",
            )
        )[0]
    except IndexError:
        mni_path = glob(
            ospj(
                ieeg_recon_path.replace("ieeg_recon", "ieeg_recon_old"),
                "module3",
                "MNI",
                "*vox_electrodes.txt",
            )
        )[0]
    mni_coords = np.loadtxt(mni_path)
    try:
        img = nib.load(
            glob(
                ospj(
                    ieeg_recon_path,
                    "module3",
                    "MNI",
                    "*T1w.nii.gz",
                )
            )[0]
        )
    except IndexError:
        img = nib.load(
            glob(
                ospj(
                    ieeg_recon_path.replace("ieeg_recon", "ieeg_recon_old"),
                    "module3",
                    "MNI",
                    "*T1w.nii.gz",
                )
            )[0]
        )
    affine_mat = img.header.get_sform()
    # apply affine matrix to mni
    # affine_mat = np.loadtxt(glob(ospj(mod3_dir, "MNI", "greedy_affine.mat"))[0])
    tmp = np.hstack((mni_coords, np.ones((mni_coords.shape[0], 1))))
    mni_coords = ((affine_mat @ tmp.T).T)[:, :3]


    # get mm space coords
    mm_coords = np.loadtxt(
        glob(
            ospj(
                ieeg_recon_path,
                "module2",
                "*mm_electrodes.txt",
            )
        )[0]
    )

    # get mm space coords
    vox_coords = np.loadtxt(
        glob(
            ospj(
                ieeg_recon_path,
                "module2",
                "*vox_electrodes.txt",
            )
        )[0]
    )

    # get electrode names
    elec_names = np.loadtxt(
        glob(
            ospj(
                ieeg_recon_path,
                "module2",
                "*names.txt",
            )
        )[0],
        dtype=str,
    )
    elec_names = np.array(clean_labels(elec_names, pt=rid))
    assert mni_coords.shape[0] == elec_names.shape[0]

    # get the final labels
    df = label_fix(rid)
    assert mni_coords.shape[0] == df.shape[0]
    # reg_labels = np.array([i.strip() for i in df["final_label"].values])
    reg_labels = np.array([i.strip() for i in df])

    # round mnicoords to 3 decimal places
    mni_coords = np.around(mni_coords, 3)

    # make a numpy array with rid, electrode names, mnicoords, final labels
    entries = np.hstack(
        (
            np.array([rid] * len(elec_names))[:, None],
            elec_names[:, None],
            mni_coords,
            mm_coords,
            vox_coords,
            reg_labels[:, None],
        )
    )

    if all_data is None:
        all_data = entries
    else:
        all_data = np.vstack((all_data, entries))

all_data = pd.DataFrame(all_data, columns=["rid", "name", "mni_x", "mni_y", "mni_z", "mm_x", "mm_y", "mm_z", "vox_x", "vox_y", "vox_z", "label"])

# %%
print("Getting all soz labels...")
# soz labels
all_data['soz'] = None
# insert SOZ labels
for ind, row in tqdm(CONFIG.metadata_io.soz_metadata.iterrows(), total=len(CONFIG.metadata_io.soz_metadata), desc="soz", leave=True):
    if "MP" in row["name"]:
        continue
    hupid = row["name"]
    rid = CONFIG.metadata_io.hup_to_rid[hupid]

    pt_elecs = all_data[all_data['rid'] == rid]

    if not isinstance(row["SOZ electrode"], str):
        continue

    soz_elecs = row["SOZ electrode"].split(",")
    soz_elecs = [i.strip() for i in soz_elecs]
    soz_elecs = clean_labels(soz_elecs, pt=hupid)
    is_soz = pt_elecs[pt_elecs.name.isin(soz_elecs)].index
    is_not_soz = pt_elecs[~pt_elecs.name.isin(soz_elecs)].index

    all_data.loc[is_soz, "soz"] = True
    all_data.loc[is_not_soz, "soz"] = False

# %%
print("Getting all resected labels...")

# # resected labels
all_data['resected'] = None

resected_channels = pd.read_pickle(ospj(CONFIG.paths.data_dir, "metadata", "resected_channels.pkl"))

for pt, group in tqdm(resected_channels.groupby("pt"), total=len(resected_channels.pt.unique()), desc="resected", leave=True):
    resected_idx = all_data[(all_data.rid == pt) & (all_data.name.isin(group.resected))].index
    not_resected_idx = all_data[(all_data.rid == pt) & (~all_data.name.isin(group.resected))].index

    all_data.loc[resected_idx, "resected"] = True
    all_data.loc[not_resected_idx, "resected"] = False

# patient_localization = sio.loadmat(ospj(CONFIG.data_dir, "metadata", "patient_localization_final.mat"))['patient_localization']
# patients = [i[0] for i in np.squeeze(patient_localization['patient'])]

# labels = []
# for row in tqdm(patient_localization['labels'][0, :], desc="resected", leave=True):
#     labels.append([i[0][0] for i in row])

# resect = np.squeeze(patient_localization['resect'])

# missings = []
# all_data['resected'] = np.nan
# for hupid, lab, res in tqdm(zip(patients, labels, resect), total=len(patients), desc="resected", leave=True):
#     if hupid == "HUP060":
#         continue
#     assert len(lab) == len(res)

#     rid = CONFIG.hup_to_rid[hupid]

#     if hupid in ("HUP086", "HUP089"):
#         lab = clean_labels(lab, pt=None)
#     else:
#         lab = clean_labels(lab, pt=hupid)
#     if hupid == "HUP086":
#         lab = [i[1:] for i in lab if (i[0] == "L") or (i[0] == "R")]

#     pt_norm = all_data.loc[all_data['rid'] == rid]

#     for l, r in zip(lab, res.squeeze()):
#         # insert resected into resected column if l is a substring of any of the normative_atlas names
#         if any((all_data['name'].str.contains(l))):
#             all_data.loc[(all_data['rid'] == rid) & (all_data['name'].str.contains(l)), 'resected'] = bool(r)
#         else:
#             # check if "EKG", "Rate", "FZ", "ROC" in l
#             if not any([i in l for i in ["EKG", "ECG", "Rate", "FZ", "ROC", "PZ"]]):
#                 missings.append((hupid, l, bool(r)))

# %%
print("Getting all spike rates...")
# spikes (takes some time)
spike_files = glob(ospj(CONFIG.paths.spikes_dir, "spike_rate_HUP*.mat"))
for f in tqdm(spike_files):
    hupid = int(f.split("/")[-1].split("_")[2].split(".")[0][3:])
    hupid = f"HUP{hupid:03d}"
    rid = CONFIG.metadata_io.hup_to_rid[hupid]

    # load spike file, mat format
    spike_data = sio.loadmat(f)
    spike_data = spike_data["perhour"][0, 0]

    spike_rate = np.array([float(i.squeeze()) for i in spike_data[:, 0]])
    elec_label = np.array([i.squeeze() for i in spike_data[:, 2]])

    elec_label = clean_labels(elec_label, pt=hupid)

    for l, r in zip(elec_label, spike_rate):
        all_data.loc[(all_data.rid == rid) & (all_data.name == l), "spike_rate"] = r

# %%
# outcome
print("Getting all outcomes...")
# # get the outcomes
# for ind, row in tqdm(CONFIG.patient_tab.iterrows(), total=len(CONFIG.patient_tab), desc="engel", leave=True):
#     rid = ind
#     engel = row["engel"]
#     if not isinstance(engel, float):
#         engel = None
#     elif np.isnan(engel):
#         engel = None
#     else:
#         engel = int(engel)
#     all_data.at[all_data["rid"] == rid, "engel"] = engel

for engel_time in [12, 24]:
    all_data[f"engel_{engel_time}"] = None
    outcomes = get_outcome(CONFIG.paths.metadata_file, engel_time)
    for rid, group in tqdm(all_data.groupby("rid"), total=len(all_data.rid.unique()), desc="engel", leave=True):
        if rid not in outcomes:
            continue
        for ind, row in group.iterrows():
            all_data.loc[ind, f"engel_{engel_time}"] = outcomes[rid]



# %%
# some qc
# missing_spikes = all_data[all_data.spike_rate.isna()]
# for rid, group in tqdm(missing_spikes.groupby('rid'), total=len(missing_spikes.rid.unique()), desc="missing spikes", leave=True):
#     hupid = CONFIG.rid_to_hup[rid]
#     f = glob(ospj(CONFIG.spikes_dir, f"spike_rate_{hupid}.mat"))
#     if len(f) == 0:
#         print(f"missing {hupid}")
#         continue

#     # load spike file, mat format
#     spike_data = sio.loadmat(f[0])
#     spike_data = spike_data["perhour"][0, 0]

#     spike_rate = np.array([float(i.squeeze()) for i in spike_data[:, 0]])
#     elec_label = np.array([i.squeeze() for i in spike_data[:, 2]])

#     new_elec_label = clean_labels(elec_label, pt=hupid)

    # print(rid, hupid)
    # print(np.array(new_elec_label))
    # print(group.name.values)

    # print()

# %%
print("Saving to data/metadata/master_elecs.csv...")
# save to data/metadata/master_elecs.csv
all_data.to_csv(ospj(CONFIG.paths.data_dir, "metadata", "master_elecs.csv"), index=False)

# %%
