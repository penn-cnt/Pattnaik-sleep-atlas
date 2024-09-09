# %%
# imports

import os
from os.path import join as ospj
import time
from fractions import Fraction
import warnings

import pandas as pd
from scipy.signal import resample_poly, firwin, filtfilt, iirfilter, iirnotch, decimate, resample, lfilter
import numpy as np


import mne
from tqdm import tqdm

from tools.config import CONFIG
# from utils import clean_labels, artifact_removal, get_iEEG_data, check_channel_types, bipolar_montage
from tools.preprocessing import clean_labels, artifact_removal, check_channel_types, bipolar_montage
from tools.io.ieeg import get_iEEG_data

from mne_bids import BIDSPath, write_raw_bids

OVERWRITE = False

# get clip_table
###### OLD CODE (ONE PER STAGE) ######
# clip_table = pd.read_csv(ospj(CONFIG.paths.data_dir, 'metadata', 'clip_times_combined.csv'), index_col=0)

# which_file = clip_table.ieeg_fname.str.extract(r'D(\d+)')
# which_file = which_file.fillna(1)
# which_file = which_file.astype(int)
# clip_table['which_file'] = which_file

# # remove N1
# clip_table = clip_table[clip_table.stage_name != 'N1']
# # reset index
# clip_table.reset_index(drop=True, inplace=True)
clip_table = pd.read_csv(ospj(CONFIG.paths.data_dir, 'metadata', 'final_clip_times.csv'))

# %%
# set up bids path
bids_path_kwargs = {
    "root": CONFIG.paths.bids_dir,
    "datatype": "ieeg",
    "extension": ".edf",
    "suffix": "ieeg",
    "task": "ictal",
    "session": "clinical01",
}
bids_path = BIDSPath(**bids_path_kwargs)

# set up processed data path
save_dir = "/mnt/leif/littlab/users/pattnaik/ieeg_sleep_atlas/data/processed_eeg_final"
os.makedirs(save_dir, exist_ok=True)

# %%
# test_rids = ['sub-RID0146', 'sub-RID0186', 'sub-RID0190', 'sub-RID0250', 'sub-RID0272', 'sub-RID0296', 'sub-RID0329', 'sub-RID0330', 'sub-RID0338', 'sub-RID0356', 'sub-RID0394', 'sub-RID0420', 'sub-RID0502', 'sub-RID0508', 'sub-RID0530', 'sub-RID0560', 'sub-RID0582', 'sub-RID0588', 'sub-RID0589', 'sub-RID0646', 'sub-RID0649']
# clip_table = clip_table[clip_table.rid.isin(test_rids) & clip_table.stage_name.isin(['W'])]
empty_files = []

for ind, row in tqdm(clip_table.iterrows(), total=len(clip_table), desc="Saving clips", leave=True, position=0):
    # for debugging and continuing the loop where it left off
    # if ind < 264:
    #     continue
    # for i_clip in [1, 2]:
    #     if pd.isnull(row[f'start_time{i_clip}']):
    #         continue
        # start_time = row[f'start_time{i_clip}']
    start_time = row.time

    fname = f"{row.pt}_{row.stage}_{row.clip_num}.edf"
    if os.path.exists(ospj(save_dir, fname)) and not OVERWRITE:
        continue

    end_time = start_time + 30

    # get bids path
    clip_bids_path = bids_path.copy().update(
        subject=row.pt.split('-')[1],
        # run=int(row["which_file"]),
        run=int(row.file),
        task=f"interictal{int(start_time)}",
    )

    # check if the file already exists, if so, load the data from here
    if clip_bids_path.fpath.exists():
        try:
            data = mne.io.read_raw_edf(clip_bids_path.fpath, verbose=False)
        except ValueError as e:
            tqdm.write(f"ValueError: {e}")
            # delete the file
            os.remove(clip_bids_path.fpath)
            # write empty file
            with open(ospj(save_dir, fname), 'w') as f:
                f.write('')
            empty_files.append(ospj(save_dir, fname))
            continue
        fs = data.info['sfreq']

        # convert to pd dataframe
        data = pd.DataFrame(data.get_data().T, columns=data.ch_names)

        # if it's longer than 30 seconds, clip it to middle 30 seconds
        if data.shape[0] > 30 * fs:
            data = data.iloc[0:int(30 * fs)]

    else:
        # get data
        data, fs = get_iEEG_data(
            CONFIG.ieeg_login.usr,
            CONFIG.ieeg_login.pwd,
            row.ieeg_fname,
            start_time_usec=start_time * 1e6,
            stop_time_usec=end_time * 1e6,
        )
        # convert nan to 0
        data.fillna(0, inplace=True)

        # write pulled data to bids
        # channels with flat line may not save proprely, so we'll drop them
        data = data[data.columns[data.min(axis=0) != data.max(axis=0)]]

        # if there aren't any channels, move on to the next row
        if data.shape[1] == 0:
            tqdm.write(f"{row.hupid} {row.pt} {row.stage} {row.clip_num} has no channels, moving on to next stage")
            # write empty file
            with open(ospj(save_dir, fname), 'w') as f:
                f.write('')
            empty_files.append(ospj(save_dir, fname))
            continue

        # clean the labels
        data.columns = clean_labels(data.columns, pt=row.ieeg_fname)

        # if there are duplicate labels, keep the first one in the table
        data = data.loc[:, ~data.columns.duplicated()]
        # get the channel types
        ch_types = check_channel_types(list(data.columns))
        ch_types.set_index("name", inplace=True, drop=True)

        # save the data
        # run is the iEEG file number
        # task is ictal with the start time in seconds appended
        data_info = mne.create_info(
            ch_names=list(data.columns), sfreq=fs, ch_types="eeg", verbose=False
        )
        data = data / 1e6  # mne needs data in volts
        raw = mne.io.RawArray(
            data.to_numpy().T,
            data_info,
            verbose=False,
        )

        raw.set_channel_types(ch_types.type)
        annots = mne.Annotations(
            onset=[0],
            duration=[CONFIG.constants.clip_size],
            description=["interictal"],
        )
        if not os.path.exists(clip_bids_path.directory):
            os.makedirs(clip_bids_path.directory)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_annotations(annots)

            write_raw_bids(
                raw,
                clip_bids_path,
                overwrite=OVERWRITE,
                verbose=False,
                allow_preload=True,
                format="EDF",
            )



    # if the data is not clean, then move on to the next row
    is_artifact = artifact_removal(data.to_numpy() * 1e6, fs).sum(axis=0) / data.shape[1] > 0.1
    if all(is_artifact):
        tqdm.write(f"{row.hupid} {row.pt} {row.stage} {row.clip_num} has artifact, moving on to next stage")
        # write empty file
        with open(ospj(save_dir, fname), 'w') as f:
            f.write('')
        empty_files.append(ospj(save_dir, fname))
        continue

    # convert to volts
    # data = data * 1e-6

    # remove artifact channels
    data = data.iloc[:, ~is_artifact]

    ch_types = check_channel_types(data.columns)

    bipolar_data, bipolar_ch_types = bipolar_montage(data.to_numpy().T, ch_types)

    if bipolar_data.shape[0] == 0:
        tqdm.write(f"{row.hupid} {row.pt} {row.stage} {row.clip_num} has no channels, moving on to next stage")
        # if the file exists, remove it
        if os.path.exists(ospj(save_dir, fname)):
            os.remove(ospj(save_dir, fname))
        # write empty file
        with open(ospj(save_dir, fname), 'w') as f:
            f.write('')
        empty_files.append(ospj(save_dir, fname))
        continue

    # downsample
    target_fs = 200
    frac = Fraction(target_fs, int(fs))

    anti_aliasing_cutoff = 80
    numtaps = 101
    fir_coeff = firwin(numtaps, anti_aliasing_cutoff, fs=fs)
    bipolar_data = resample_poly(
        x=bipolar_data,
        # up=frac.numerator,
        # down=frac.denominator,
        up=1,
        # down=int(fs / target_fs),
        down=np.floor(fs / target_fs).astype(int),
        axis=1,
        window=fir_coeff,
    )

    target_fs = int(fs / np.floor(fs / target_fs))

    # powerline noise
    # high pass at 48 Hz with (FIR filter, order 100, 38-48 Hz transition band)
    # powerline_subtract_signal = bipolar_data.copy()
    # nyquist = fs / 2
    # numtaps = 100
    # fir_coeff = firwin(numtaps, [38, 48], fs = target_fs, pass_zero=False)
    # powerline_subtract_signal = filtfilt(fir_coeff, 1, powerline_subtract_signal, axis=1)

    power_line_freq = 60

    # Function to estimate and remove harmonics for multiple channels without a for loop
    def remove_harmonic(signals, freq, fs, order=1):
        t = np.arange(signals.shape[1]) / fs
        harmonic_signal = np.sin(2 * np.pi * freq * t)
        harmonic_signal = harmonic_signal[np.newaxis, :]  # Shape to (1, samples)

        # Estimate amplitude and phase
        estimated_amplitude = np.sum(signals * harmonic_signal, axis=1) / signals.shape[1]
        estimated_phase = np.arctan2(
            np.sum(signals * np.cos(2 * np.pi * freq * t), axis=1),
            np.sum(signals * np.sin(2 * np.pi * freq * t), axis=1)
        )
        
        # Broadcast estimated amplitude and phase for each channel
        estimated_harmonic = estimated_amplitude[:, np.newaxis] * np.sin(2 * np.pi * freq * t + estimated_phase[:, np.newaxis])
        
        # Smooth the estimated harmonic
        smoothed_estimate = lfilter([1 - 1/order], [1, -1/order], estimated_harmonic, axis=1)
        
        # Subtract the smoothed estimate from the signals
        return signals - smoothed_estimate

    # Estimate and subtract the third harmonic
    bipolar_data = remove_harmonic(bipolar_data, 3 * power_line_freq, target_fs)

    # Estimate and subtract the second harmonic
    bipolar_data = remove_harmonic(bipolar_data, 2 * power_line_freq, target_fs)

    # Estimate and subtract the fundamental power-line frequency
    bipolar_data = remove_harmonic(bipolar_data, power_line_freq, target_fs)

    # apply a low-pass antialiasing filter at 80 Hz using scipy.signal at 180, 120, 60
    # if fs > 360:
    #     notches = [180, 120, 60]
    # else:
    #     notches = [120, 60]

    # for notch in notches:
    #     # estimate phase and amplitude at notch
    #     b, a = iirnotch(notch, 100, fs)
    #     bipolar_data = filtfilt(b, a, bipolar_data, axis=1)

    # # resample to 200 Hz
    # new_fs = 200
    # frac = Fraction(new_fs, int(fs))
    # bipolar_data = resample_poly(
    #     bipolar_data.T, up=frac.numerator, down=frac.denominator
    # ).T  # (n_samples, n_channels)
    # fs = new_fs

    # subtract mean of each channel axis=1
    bipolar_data = bipolar_data - bipolar_data.mean(axis=1, keepdims=True)

    # fill up to 6000 on axis 1 with np.infs
    if bipolar_data.shape[1] < 6000:
        bipolar_data = np.pad(bipolar_data, ((0, 0), (0, 6000 - bipolar_data.shape[1])), constant_values=0)
    else:
        # clip
        bipolar_data = bipolar_data[:, :6000]
    bipolar_data = bipolar_data.astype(np.float16)

    # save as mne raw object
    raw = mne.io.RawArray(
        bipolar_data,
        mne.create_info(
            list(bipolar_ch_types.name),
            target_fs,
            ch_types='eeg'),
        verbose=False)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mne.export.export_raw(
                ospj(save_dir, fname),
                raw,
                fmt='edf',
                overwrite=OVERWRITE,
                verbose=False,
            )
    except RuntimeError as e:
        tqdm.write(f"RuntimeError: {e}")
        time.sleep(1)
        mne.export.export_raw(
            ospj(save_dir, fname),
            raw,
            fmt='edf',
            overwrite=OVERWRITE,
            verbose=False,
        )

# save empty files
with open(ospj(save_dir, 'empty_files.txt'), 'w') as f:
    for empty_file in empty_files:
        f.write(empty_file + '\n')

# %%
# make empty files list
empty_file_inds = []
for ind, row in tqdm(clip_table.iterrows(), total=len(clip_table), desc="Saving clips", leave=True, position=0):
    start_time = row.time

    fname = f"{row.pt}_{row.stage}_{row.clip_num}.edf"
    try:
        data = mne.io.read_raw_edf(ospj(save_dir, fname), verbose=False)
    except ValueError as e:
        empty_file_inds.append(ind)
        continue


# %%
empty_files = pd.DataFrame(clip_table.loc[empty_file_inds])

# which patients have the most empty files
display(empty_files.pt.value_counts())

# which stages have the most empty files
display(empty_files.stage.value_counts())

# how many have all clips empty filter so that they equal 10, get indices
all_empty = empty_files.groupby(['pt', 'stage']).size().apply(lambda x: x == 10)
all_empty = all_empty[all_empty].index
display(all_empty)
# %%
existing_files = pd.DataFrame(clip_table.loc[~clip_table.index.isin(empty_file_inds)])
existing_file_names = existing_files.apply(lambda x: ospj(save_dir, f"{x.pt}_{x.stage}_{x.clip_num}.edf"), axis=1)

empty_file_names = empty_files.apply(lambda x: ospj(save_dir, f"{x.pt}_{x.stage}_{x.clip_num}.edf"), axis=1)

existing_file_names.to_csv(ospj(save_dir, 'existing_files.csv'), index=False)

# %%
