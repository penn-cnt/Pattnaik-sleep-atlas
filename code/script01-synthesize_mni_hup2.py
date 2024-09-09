"""
This script will make a table of all the MNI and HUP data as well as their signals so any features can be calculated.
"""
# %%
# imports
from os.path import exists as ospe
from os.path import join as ospj

# Third party imports
import pandas as pd
import numpy as np
import mne

# Local imports
from tools.config import CONFIG
from tools.io.metadata import clean_labels, make_metadata_table

clip_times = pd.read_csv(ospj(CONFIG.paths.data_dir, 'metadata', 'clip_times_combined.csv'), index_col=0)

overwrite = False
# %%
mni_stages = [
    ('N2', 'N', 'N2'),
    ('N3', 'D', 'N3'),
    ('REM', 'R', 'R'),
    ('Wakefulness', 'W', 'W')]

stages = [i[2] for i in mni_stages]

def _load_hup_atlas(overwrite=False):
    """Load the HUP atlas metadata.

    Returns:
        pd.DataFrame: hup_atlas
    """
    hup_atlas = make_metadata_table(overwrite=overwrite)
    hup_atlas['reg'] = hup_atlas.final_label.map(CONFIG.metadata_io.dkt_to_custom)
    # drop rows that don't have an mni_reg
    hup_atlas = hup_atlas.dropna(subset=['reg'])
    # rename rid to pt
    hup_atlas = hup_atlas.rename(columns={'rid': 'pt'})
    # set column hemi to be if mni_x < 0 then L else R
    hup_atlas['hemi'] = np.where(hup_atlas.mni_x < 0, 'L', 'R')

    # # rename hemi to the last character of reg
    hup_atlas['hemi'] = hup_atlas.reg.str[-1]

    return hup_atlas

def _load_mni_atlas():
    """Load the MNI atlas metadata. 

    Returns:
        pd.DataFrame: mni_atlas
    """
    mni_atlas = CONFIG.metadata_io.ch_info.copy(deep=True)
    # rename columns x, y, z to mni_x, mni_y, mni_z
    mni_atlas.rename(columns={'x': 'mni_x', 'y': 'mni_y', 'z': 'mni_z'}, inplace=True)
    # rename column Patient to pt
    mni_atlas.rename(columns={'Patient': 'pt'}, inplace=True)
    # rename column Channel name to name
    mni_atlas.rename(columns={'Channel name': 'name'}, inplace=True)
    # rename column Region to reg_id
    mni_atlas.rename(columns={'Region': 'reg_id'}, inplace=True)
    # rename column Hemisphere to hemi
    mni_atlas.rename(columns={'Hemisphere': 'hemi'}, inplace=True)
    # make a dict that converts between index and Region name
    reg_id_to_name = dict(zip(CONFIG.metadata_io.reg_info.index + 1, CONFIG.metadata_io.reg_info['Region name']))
    mni_atlas['mni_reg'] = mni_atlas.reg_id.map(reg_id_to_name)
    mni_atlas['reg'] = mni_atlas.mni_reg.map(CONFIG.metadata_io.mni_to_custom)
    # change the last character of 'reg' column to be 'hemi', ignore rows where 'reg' is nan
    mni_atlas['reg'] = np.where(mni_atlas.reg.isnull(), np.nan, mni_atlas.reg.str[:-1] + mni_atlas.hemi)

    mni_atlas.pt = mni_atlas.pt.astype(str)
    mni_atlas.pt = mni_atlas.pt.str.zfill(3)
    mni_atlas.pt = 'MNI' + mni_atlas.pt
    # mni_atlas['normative'] = True

    # drop Electrode type column
    mni_atlas.drop(columns=['Electrode type', 'reg_id'], inplace=True)
    return mni_atlas

# %%
# load or make combined atlas
if ospe(ospj(CONFIG.paths.data_dir, 'metadata', 'combined_atlas_metadata.csv')) and not overwrite:
    print("combined_atlas_metadata.csv already exists, skipping...")

    combined_atlas = pd.read_csv(ospj(CONFIG.paths.data_dir, 'metadata', 'combined_atlas_metadata.csv'), index_col=0)
else:
    print("combined_atlas_metadata.csv does not exist, creating...")
    hup_atlas = _load_hup_atlas(overwrite=overwrite)
    mni_atlas = _load_mni_atlas()

    # round all coordinates to 3 decimal places
    hup_atlas[['mni_x', 'mni_y', 'mni_z']] = hup_atlas[['mni_x', 'mni_y', 'mni_z']].round(3)
    hup_atlas[['mm_x', 'mm_y', 'mm_z']] = hup_atlas[['mm_x', 'mm_y', 'mm_z']].round(3)
    hup_atlas[['vox_x', 'vox_y', 'vox_z']] = hup_atlas[['vox_x', 'vox_y', 'vox_z']].round(3)
    hup_atlas[['ch1_spike_rate', 'ch2_spike_rate']] = hup_atlas[['ch1_spike_rate', 'ch2_spike_rate']].round(3)
    mni_atlas[['mni_x', 'mni_y', 'mni_z']] = mni_atlas[['mni_x', 'mni_y', 'mni_z']].round(3)

    combined_atlas = mni_atlas.append(hup_atlas, sort=False)
    combined_atlas = combined_atlas.reset_index(drop=True)
    combined_atlas['site'] = combined_atlas.pt.str.startswith('MNI').map({True: 'MNI', False: 'HUP'})

    combined_atlas.to_csv(ospj(CONFIG.paths.data_dir, 'metadata', 'combined_atlas_metadata.csv'))

# %%
# data structure for up to 10 clips per channel:
# np array shape (n_channels, n_stages [4])
# each element is a list of up to 10 signals
# each signal is the time series of the clip
mni_fs = 200
mni_ch_size = 30 * mni_fs

master_data = np.zeros((combined_atlas.shape[0], len(stages)), dtype=object)
fs_array = np.ones((len(combined_atlas),)) * np.nan

for stage_name, stage_char, stage_col_name in mni_stages:
    for ind, row in CONFIG.metadata_io.reg_info.iterrows():
        reg_name = row['Region name']

        reg_id = row['Region #']
        reg_path = ospj(CONFIG.paths.data_dir, "mni", f"{stage_name}_AllRegions", f"{reg_name}_{stage_char}.edf")
        # load edf
        raw = mne.io.read_raw_edf(reg_path, preload=True, verbose=False)

        # get channel names
        ch_names = raw.ch_names
        # remove last character from channel names
        ch_names = [ch_name[:-1] for ch_name in ch_names]

        # get the indices of the channels in the atlas for this pt in the order of the data
        atlas_ch_names = combined_atlas[(combined_atlas.mni_reg == reg_name) & (combined_atlas.site == 'MNI')].name
        for idx, name in atlas_ch_names.items():
            if name not in ch_names:
                fs_array[idx] = mni_fs
                continue
            # get the data for this channel
            ch_data = ch_names.index(name)
            ch_data = raw.get_data(picks=ch_data)
            fs = raw.info['sfreq']
            # get the signal for this channel
            signal = ch_data[0].astype(np.float16)

            # add it to the all_signals array
            clip_size = min(mni_ch_size, len(signal))
            # all_signals[stage_col_name][idx, :clip_size] = signal[:clip_size]
            if master_data[idx, stages.index(stage_col_name)] == 0:
                master_data[idx, stages.index(stage_col_name)] = [signal[:clip_size]]
            else:
                master_data[idx, stages.index(stage_col_name)].append(signal[:clip_size])
        
            fs_array[idx] = mni_fs

hup_pts = sorted(combined_atlas[combined_atlas.site == 'HUP'].pt.unique())
processed_dir = ospj(CONFIG.paths.data_dir, 'processed_eeg_final')
empty_files = pd.read_csv(ospj(processed_dir, 'empty_files.txt'), index_col=None, header=None)[0].values
# empty_files = np.hstack(
#     (empty_files,
#     [ospj(processed_dir, 'sub-RID0018_W_0.edf')],
#     [ospj(processed_dir, 'sub-RID0018_W_1.edf')],
#     )
# )

uncaught_errors = []

n_clips = 10

for pt in hup_pts:
    for stage in stages:
        for clip_num in range(n_clips):
            file_name = ospj(processed_dir, f"{pt}_{stage}_{clip_num}.edf")

            if not ospe(file_name):
                # print(f"{file_name} does not exist, skipping...")
                continue
            if file_name in empty_files:
                # print(f"{file_name} is empty, skipping...")
                continue
            
            # load the data
            try:
                clip = mne.io.read_raw_edf(
                    file_name,
                    preload=True,
                    verbose=False)
            except:
                uncaught_errors.append(file_name)
                continue
            
            clip_fs = clip.info['sfreq']
            clip_size = int(clip_fs * 30)

            # get the indices of the channels in the atlas for this pt in the order of the data
            atlas_ch_names = combined_atlas[combined_atlas.pt == pt].name
            atlas_idx = atlas_ch_names.index

            clip_ch_names = clip.ch_names

            for idx, name in atlas_ch_names.items():
                if name not in clip_ch_names:
                    fs_array[idx] = clip_fs
                    continue
                ch_data = clip.get_data(picks=clip_ch_names.index(name))
                signal = ch_data[0].astype(np.float16)
                signal = signal[:min(clip_size, len(signal))]

                if master_data[idx, stages.index(stage)] == 0:
                    master_data[idx, stages.index(stage)] = [signal]
                else:
                    master_data[idx, stages.index(stage)].append(signal)

                fs_array[idx] = clip_fs

# %%
# fill anywhere that is 0 with empty lists
for i in range(master_data.shape[0]):
    for j in range(master_data.shape[1]):
        if master_data[i, j] == 0:
            master_data[i, j] = []

# %%
# save the data
np.save(ospj(CONFIG.paths.data_dir, 'combined_atlas_signals_all_clips.npy'), master_data)
np.save(ospj(CONFIG.paths.data_dir, 'combined_atlas_fs.npy'), fs_array)

# %%
