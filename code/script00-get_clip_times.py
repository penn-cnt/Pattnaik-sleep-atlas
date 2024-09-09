"""
This script will pull up to 10 clip times for each stage to save them in a file. 
"""

# %%
# imports
from os.path import join as ospj
from glob import glob

import pandas as pd
import numpy as np

from tools.config import CONFIG

# %%
# paths
ii_time_path = "/mnt/leif/littlab/users/rsg20/IEEG_processing/data/{hupid}/interictal/interictal_clip_times.npy"
ii_file_path = "/mnt/leif/littlab/users/rsg20/IEEG_processing/data/{hupid}/interictal/interictal_clip_files.npy"

bids_path = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS/{rid}/ses-clinical01/ieeg/{rid}_ses-clinical01_task-interictal{start}_run-{file}_ieeg.edf"
ian_sleep = "/mnt/leif/littlab/users/ianzyong/sleep-atlas/data"

# %%
# metadata I/0
metadata = pd.read_csv(ospj(CONFIG.paths.data_dir, 'combined_atlas_metadata.csv'))

# %%
stage_to_int_map = {
    "R": 1, "W": 2, "N1": 3, "N2": 4, "N3": 5
}

int_to_stage_map = {v: k for k, v in stage_to_int_map.items()}

MIN_CLIPS = 10

 
# %%
master_clip_times = []

for pt, group in metadata[metadata.site == 'HUP'].groupby('pt'):
    hupid = CONFIG.metadata_io.rid_to_hup[pt]

    try:
        ii_times = np.load(ii_time_path.format(hupid=hupid))
        ii_files = np.load(ii_file_path.format(hupid=hupid))
    except FileNotFoundError as e:
        print(f"Skipping {pt} due to {e}")
        continue

    # keep the first 10 clips
    ii_times = ii_times[:min(10, len(ii_times))]
    ii_files = ii_files[:min(10, len(ii_files))]

    for i, (time, file) in enumerate(zip(ii_times, ii_files)):
        ieeg_fnames = CONFIG.metadata_io.patient_tab.ieeg_fname[pt]
        ieeg_fnames = ieeg_fnames.split(";")
        ieeg_fnames = [x.strip() for x in ieeg_fnames]
        ieeg_fname = ieeg_fnames[int(file) - 1]
        master_clip_times.append({
            'pt': pt,
            'hupid': hupid,
            'ieeg_fname': ieeg_fname,
            'time': int(time),
            'file': file,
            'stage': 'W',
            'clip_num': i,
        })

# master_clip_times = pd.DataFrame(master_clip_times)

# %%
night_classified = pd.read_csv(
    "/mnt/leif/littlab/users/ianzyong/sleep-atlas/util/nights_classified_final.csv",
    index_col=0,
)
night_classified['hupid'] = night_classified.index.str.split('_')
night_classified['hupid'] = night_classified['hupid'].apply(lambda x: x[0])
# make sure HUPXX has trailing 0s
night_classified['hupid'] = night_classified['hupid'].apply(lambda x: f"HUP{int(x[3:]):03d}")
night_classified['rid'] = night_classified['hupid'].apply(lambda x: CONFIG.metadata_io.hup_to_rid[x])

summary_files = []
for ind, row in night_classified.iterrows():
    summary_files.append(glob(ospj(ian_sleep, f"sub-{row['hupid']}", f"*night{row['night_number']}*sleepstage.csv")))
summary_files = [x for x in summary_files if len(x) > 0]
new_summary_files = []
for i, f in enumerate(summary_files):
    if len(f) > 1:
        new_summary_files.append([x for x in f if 'preictalhrs' in x or 'D01' in x][0])
    else:
        new_summary_files.append(f[0])
summary_files = sorted(new_summary_files)

# organize summary_files into a pd dataframe
file_table = []
for i, f in enumerate(summary_files):
    hupid = f.split("/")[-2].split("-")[-1]
    ieeg_fname = night_classified[night_classified.hupid == hupid].index[0]

    f_parts = f.split("_")
    if "D0" in f:
        start_time = f_parts[3]
    else:
        start_time = f_parts[2]

    file_table.append({
        'ieeg_fname': ieeg_fname,
        'file_path': f,
        'start_time': float(start_time),
    })
file_table = pd.DataFrame(file_table)
# sort by file_path
file_table = file_table.sort_values('file_path').reset_index(drop=True)

# for each ieeg_fname, only keep the row with the largest night_num
# file_table = file_table.sort_values(['ieeg_fname', 'night_num']).drop_duplicates(['ieeg_fname'], keep='last')
# reset index
file_table = file_table.reset_index(drop=True)

# %%
for ind, row in file_table.iterrows():
    data = pd.read_csv(row['file_path'])
    data = data[data.file_index == 1]
    data['time'] = data['time'] - data['time'].iloc[0]
    data['time'] = data['time'] * 24 * 60 * 60
    # round to one decimal place
    data['time'] = data['time'].apply(lambda x: int(x))
    data['sleep_stage'] = data['sleep_stage'].apply(lambda x: int_to_stage_map[x])

    hupid = row.file_path.split("/")[-2].split("-")[-1]

    for stage, group in data.groupby('sleep_stage'):
        if stage in ['N1', 'W']:
            continue
        # print(f"Stage {stage} has {len(group)} clips")

        group = group.sort_values('confidence', ascending=False)

        if len(group) < MIN_CLIPS:
            print(f"Skipping {hupid} {stage} due to insufficient clips")
            continue
        
        group = group.iloc[:10]

        for i, (time, confidence) in enumerate(zip(group.time, group.confidence)):
            if confidence < 0.5:
                continue

            master_clip_times.append({
                'pt': CONFIG.metadata_io.hup_to_rid[hupid],
                'hupid': hupid,
                'ieeg_fname': row['ieeg_fname'],
                'time': int(time + (row['start_time'] / 1e6)),
                'file': 1,
                'stage': stage,
                'clip_num': i,
            })

# %%
master_clip_times = pd.DataFrame(master_clip_times)
master_clip_times.sort_values(['pt', 'stage', 'clip_num'], inplace=True)

master_clip_times.to_csv(ospj(CONFIG.paths.data_dir, 'metadata/final_clip_times.csv'), index=False)

# %%
