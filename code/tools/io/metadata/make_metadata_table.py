# %%
# imports
from os.path import join as ospj
from glob import glob
import sys
import os
import re

import numpy as np
import nibabel as nib
import scipy.io as sio
from tqdm import tqdm
import pandas as pd

# append root directory to sys.path
sys.path.append(ospj("..", "..", ".."))

from tools.config import CONFIG
from tools.io.metadata import get_cnt_inventory, clean_labels, label_fix

# %%
inventory = get_cnt_inventory()
inventory = inventory[inventory.ieeg_recon]


apn_fname = "/mnt/leif/littlab/users/pattnaik/ictal_patterns/data/metadata/apn_dkt_labels.txt"
with open(apn_fname, "r") as f:
    lines = f.readlines()

apn_dkt = {}
for line in lines:
    if line.startswith("Label"):
        words = line.strip().split()
        reg_id = int(words[1][:-1])
        reg_name = " ".join(words[2:])
        apn_dkt[reg_id] = reg_name

# %%
def _get_mni_coordinates(ieeg_recon_path):
    """Function to get mni coordinates. This is a helper function for make_metadata_table.py.

    Args:
        ieeg_recon_path (str): path to ieeg_recon directory

    Returns:
        np.array: mni coordinates
    """
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
    tmp = np.hstack((mni_coords, np.ones((mni_coords.shape[0], 1))))
    mni_coords = ((affine_mat @ tmp.T).T)[:, :3]
    return mni_coords

def _get_mm_coordinates(ieeg_recon_path):
    """Function to get mm coordinates. This is a helper function for make_metadata_table.py.

    Args:
        ieeg_recon_path (str): path to ieeg_recon directory

    Returns:
        np.array: mm coordinates
    """
    return np.loadtxt(
        glob(
            ospj(
                ieeg_recon_path,
                "module2",
                "*mm_electrodes.txt",
            )
        )[0]
    )

def _get_vox_coordinates(ieeg_recon_path):
    """Function to get vox coordinates. This is a helper function for make_metadata_table.py.

    Args:
        ieeg_recon_path (str): path to ieeg_recon directory

    Returns:
        np.array: vox coordinates
    """
    return np.loadtxt(
        glob(
            ospj(
                ieeg_recon_path,
                "module2",
                "*vox_electrodes.txt",
            )
        )[0]
    )

def _get_elec_names(ieeg_recon_path, rid):
    """Function to get electrode names. This is a helper function for make_metadata_table.py.

    Args:
        ieeg_recon_path (str): path to ieeg_recon directory

    Returns:
        np.array: electrode names
    """
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
    return elec_names

def _make_bipolar_table(all_data):
    """Function to make bipolar table. This is a helper function for make_metadata_table.py.

    Args:
        all_data (pd.DataFrame): all_data dataframe
    
    Returns:
        pd.DataFrame: all_bipolars dataframe
    """
    all_bipolars = []
    for pt_rid, group in all_data.groupby(by='rid'):
        bipolar_pairs = []
        for ind, row in group.iterrows():
            ch1 = row['name']
            regex_match = re.match(r"(\D+)(\d+)", ch1)
            lead = regex_match.group(1)
            contact = int(regex_match.group(2))

            ch2 = f"{lead}{contact + 1:02d}"
            if ch2 in group['name'].values:
                # append indices
                bipolar_pairs.append((ind, group[group['name'] == ch2].index[0]))

        for ch1_ind, ch2_ind in bipolar_pairs:
            ch1 = group.loc[ch1_ind]
            ch2 = group.loc[ch2_ind]

            info = dict()
            info['rid'] = pt_rid
            info['name'] = f"{ch1['name']}-{ch2['name']}"
            info['mni_x'] = (ch1['mni_x'] + ch2['mni_x']) / 2
            info['mni_y'] = (ch1['mni_y'] + ch2['mni_y']) / 2
            info['mni_z'] = (ch1['mni_z'] + ch2['mni_z']) / 2
            info['mm_x'] = (ch1['mm_x'] + ch2['mm_x']) / 2
            info['mm_y'] = (ch1['mm_y'] + ch2['mm_y']) / 2
            info['mm_z'] = (ch1['mm_z'] + ch2['mm_z']) / 2
            info['vox_x'] = (ch1['vox_x'] + ch2['vox_x']) / 2
            info['vox_y'] = (ch1['vox_y'] + ch2['vox_y']) / 2
            info['vox_z'] = (ch1['vox_z'] + ch2['vox_z']) / 2
            info['ch1_label'] = ch1['label']
            info['ch2_label'] = ch2['label']
            info['ch1_soz'] = ch1['soz']
            info['ch2_soz'] = ch2['soz']
            info['ch1_spike_rate'] = ch1['spike_rate']
            info['ch2_spike_rate'] = ch2['spike_rate']
            info['ch1_resected'] = ch1['resected']
            info['ch2_resected'] = ch2['resected']
            info['engel_12'] = ch1['engel_12']
            info['engel_24'] = ch1['engel_24']

            all_bipolars.append(info)
    all_bipolars = pd.DataFrame(all_bipolars)
    bad_regions = [
        "cerebellum",
        "brain stem",
        "white matter",
    ]

    all_bipolars['bipolar_label'] = None
    for pt_rid, group in all_bipolars.groupby(by='rid'):
        # open aparc
        try:
            img_path = glob(ospj(
                CONFIG.paths.bids_dir,
                pt_rid,
                "derivatives/ieeg_recon/module3",
                "*DKTantspynet.nii.gz"
            ))[0]
        except IndexError:
            img_path = glob(ospj(
                CONFIG.paths.bids_dir,
                pt_rid,
                "derivatives/ieeg_recon_old/module3",
                "*DKTantspynet.nii.gz"
            ))[0]

        img = nib.load(img_path)
        img_data = img.get_fdata().astype(int)

        coords = group[['vox_x', 'vox_y', 'vox_z']].values.astype(int)

        for ind, coord in zip(group.index, coords):
            region_id = img_data[coord[0], coord[1], coord[2]]
            region_name = apn_dkt[region_id]
            if region_name == 'background':
                region_name = "EmptyLabel"
            all_bipolars.at[ind, 'bipolar_label'] = region_name

        ch1_label = all_bipolars.loc[group.index, 'ch1_label'].values
        ch2_label = all_bipolars.loc[group.index, 'ch2_label'].values
        bipolar_label = all_bipolars.loc[group.index, 'bipolar_label'].values

        final_labels = []
        for ind, ch1, ch2, bipolar in zip(group.index, ch1_label, ch2_label, bipolar_label):
            # final label is bipolar unless it is EmptyLabel or it contains "cerebellum"
            # if it is EmptyLabel, then final label is mode of ch1 and ch2
            # if it contains "cerebellum", then final label is mode of ch1 and ch2
            if bipolar == "EmptyLabel" or any([bad_region in bipolar for bad_region in bad_regions]):
                if ch1 == ch2:
                    final_label = ch1
                elif ch1 == "EmptyLabel" or any([bad_region in ch1 for bad_region in bad_regions]):
                    final_label = ch2
                elif ch2 == "EmptyLabel" or any([bad_region in ch2 for bad_region in bad_regions]):
                    final_label = ch1
                else:
                    final_label = ch1
            else:
                final_label = bipolar
            
            all_bipolars.at[ind, 'final_label'] = final_label
    # normative
    # all_bipolars['normative'] = None
    # # normative conditions: 
    # # 1. both ch1 and ch2 are not soz
    # # 2. both ch1 and ch2 are not resected
    # # 3. mean of ch1 and ch2 spike rate is less than CONFIG.spike_thresh
    # # write a cell to set normative to True if all of the above conditions are met
    # all_bipolars['normative'] = all_bipolars.apply(
    #     lambda row: (
    #         (row['ch1_soz'] == False) and
    #         (row['ch2_soz'] == False) and
    #         (row['ch1_resected'] == False) and
    #         (row['ch2_resected'] == False) and
    #         ((row['ch1_spike_rate'] + row['ch2_spike_rate']) / 2 < CONFIG.constants.spike_thresh) and
    #         (row['engel'] == 1)
    #     ),
    #     axis=1
    # )

    all_bipolars.to_csv(ospj(CONFIG.paths.data_dir, "metadata", "master_bipolars.csv"), index=False)
    return all_bipolars

def make_metadata_table(overwrite=False):
    """Function to make metadata table. This is a helper function for make_metadata_table.py.

    Args:
        overwrite (bool, optional): whether to overwrite existing metadata table. Defaults to False.
    """
    if os.path.exists(ospj(CONFIG.paths.data_dir, "metadata", "master_elecs.csv")) and not overwrite:
        print("Metadata table already exists. Skipping...")
        all_data = pd.read_csv(ospj(CONFIG.paths.data_dir, "metadata", "master_elecs.csv"))
    else:
        print("Getting all coords...")
        all_data = {
            "rid": [],
            "name": [],
            "mni_x": [],
            "mni_y": [],
            "mni_z": [],
            "mm_x": [],
            "mm_y": [],
            "mm_z": [],
            "vox_x": [],
            "vox_y": [],
            "vox_z": [],
            "label": [],

        }
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
            mni_coords = _get_mni_coordinates(ieeg_recon_path)

            # get mm space coords
            mm_coords = _get_mm_coordinates(ieeg_recon_path)

            # get vox space coords
            vox_coords = _get_vox_coordinates(ieeg_recon_path)

            # get electrode names
            elec_names = _get_elec_names(ieeg_recon_path, rid)
            assert mni_coords.shape[0] == elec_names.shape[0]
            
            regions = label_fix(rid)

            all_data['rid'].extend([rid] * len(elec_names))
            all_data['name'].extend(elec_names)
            all_data['mni_x'].extend(mni_coords[:, 0])
            all_data['mni_y'].extend(mni_coords[:, 1])
            all_data['mni_z'].extend(mni_coords[:, 2])
            all_data['mm_x'].extend(mm_coords[:, 0])
            all_data['mm_y'].extend(mm_coords[:, 1])
            all_data['mm_z'].extend(mm_coords[:, 2])
            all_data['vox_x'].extend(vox_coords[:, 0])
            all_data['vox_y'].extend(vox_coords[:, 1])
            all_data['vox_z'].extend(vox_coords[:, 2])
            all_data['label'].extend(regions)

        all_data = pd.DataFrame(all_data)

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

        print("Getting all resected labels...")
        # resected labels
        all_data['resected'] = None

        patient_localization = sio.loadmat(ospj(CONFIG.paths.data_dir, "metadata", "patient_localization_final.mat"))['patient_localization']
        patients = [i[0] for i in np.squeeze(patient_localization['patient'])]

        labels = []
        for row in tqdm(patient_localization['labels'][0, :], desc="resected", leave=True):
            labels.append([i[0][0] for i in row])

        resect = np.squeeze(patient_localization['resect'])

        missings = []
        all_data['resected'] = np.nan
        for hupid, lab, res in tqdm(zip(patients, labels, resect), total=len(patients), desc="resected", leave=True):
            if hupid == "HUP060":
                continue
            assert len(lab) == len(res)

            rid = CONFIG.metadata_io.hup_to_rid[hupid]

            if hupid in ("HUP086", "HUP089"):
                lab = clean_labels(lab, pt=None)
            else:
                lab = clean_labels(lab, pt=hupid)
            if hupid == "HUP086":
                lab = [i[1:] for i in lab if (i[0] == "L") or (i[0] == "R")]

            pt_norm = all_data.loc[all_data['rid'] == rid]

            for l, r in zip(lab, res.squeeze()):
                # insert resected into resected column if l is a substring of any of the normative_atlas names
                if any((all_data['name'].str.contains(l))):
                    all_data.loc[(all_data['rid'] == rid) & (all_data['name'].str.contains(l)), 'resected'] = bool(r)
                else:
                    # check if "EKG", "Rate", "FZ", "ROC" in l
                    if not any([i in l for i in ["EKG", "ECG", "Rate", "FZ", "ROC", "PZ"]]):
                        missings.append((hupid, l, bool(r)))

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

            # outcome
            print("Getting all outcomes...")
            # get the outcomes
            all_data['engel'] = None
            for ind, row in tqdm(CONFIG.metadata_io.patient_tab.iterrows(), total=len(CONFIG.metadata_io.patient_tab), desc="engel", leave=True):
                rid = ind
                engel = row["engel"]
                if not isinstance(engel, float):
                    engel = None
                elif np.isnan(engel):
                    engel = None
                else:
                    engel = float(engel)
                all_data.at[all_data["rid"] == rid, "engel"] = engel

        print("Saving to data/metadata/master_elecs.csv...")
        # save to data/metadata/master_elecs.csv
        all_data.to_csv(ospj(CONFIG.paths.data_dir, "metadata", "master_elecs.csv"), index=False)

    all_bipolars = _make_bipolar_table(all_data)
    # if os.path.exists(ospj(CONFIG.paths.data_dir, "metadata", "master_bipolars.csv")) and not overwrite:
    #     print("Bipolar table already exists. Skipping...")
    #     all_bipolars = pd.read_csv(ospj(CONFIG.paths.data_dir, "metadata", "master_bipolars.csv"))
    # else:
    #     print("Making bipolar table...")
    #     all_bipolars = _make_bipolar_table(all_data)
    
    return all_bipolars

# %%
