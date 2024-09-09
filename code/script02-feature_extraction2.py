"""
This script will extract the features from the data and save them to a file.
"""
#%%
# imports
from os.path import join as ospj
import itertools
from tqdm import tqdm

import pandas as pd
import numpy as np

from tools.config import CONFIG
from tools.features import bandpower
from tools.io.ieeg import get_iEEG_data
from scipy.signal import coherence
import pickle

# %%
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']
stages = ['N2', 'N3', 'R', 'W']
fs = 200

band_ranges = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 80),
    "broad": (1, 80)
}
N_BANDS = len(bands)


# %%
def _pxx_to_coher_bands(freq, pxx):
    """Convert power spectral density to coherence bands.

    Args:
        freq (np.ndarray): Frequencies.
        pxx (np.ndarray): Power spectral density.

    Returns:
        np.ndarray: Coherence bands.
    """
    coher_bands = np.zeros((N_BANDS, len(pxx)))
    for i_band, (band_name, band_range) in enumerate(band_ranges.items()):
        band_idx = np.logical_and(freq >= band_range[0], freq < band_range[1])
        coher_bands[i_band] = np.mean(pxx[band_idx], axis=0)

    return coher_bands


# %%
# data loading
metadata = pd.read_csv(ospj(CONFIG.paths.data_dir, 'metadata' , 'combined_atlas_metadata.csv'), index_col=0)
# signals = np.load(ospj(CONFIG.paths.data_dir, 'combined_atlas_signals.npz'))
signals = np.load(ospj(CONFIG.paths.data_dir, 'combined_atlas_signals_all_clips.npy'), allow_pickle=True)
# fs_array = np.load(ospj(CONFIG.paths.data_dir, 'combined_atlas_fs.npz'))['fs_array']
fs_array = np.load(ospj(CONFIG.paths.data_dir, 'combined_atlas_fs.npy'), allow_pickle=True)

# convert signals to dict for easier access
# signals = {pt: signals[pt] for pt in signals}

# assert that signals is same shape as metadata
# for s in stages:
    # assert len(signals[s]) == len(metadata)

# %%
### BANDPOWER ###
# calculate bandpower for each signal
bandpowers = np.zeros_like(signals, dtype=object)
for sig in range(len(signals)):
    for s in range(len(signals[sig])):
        bandpowers[sig][s] = []

for i_ch, ch in tqdm(enumerate(signals), total=len(signals), desc='Channels'):
    fs = fs_array[i_ch]
    if np.isnan(fs):
        pt_fs = fs_array[metadata[metadata.pt == metadata.loc[i_ch, 'pt']].index]
        pt_fs = pt_fs[~np.isnan(pt_fs)]
        fs = np.mean(pt_fs)

    if np.isnan(fs):
        rid = metadata.loc[i_ch, 'pt']
        ieeg_fnames = CONFIG.metadata_io.patient_tab.loc[rid].ieeg_fname
        ieeg_fname = ieeg_fnames.split(';')[0].strip()
        _, fs = get_iEEG_data(
            CONFIG.ieeg_login.usr,
            CONFIG.ieeg_login.pwd,
            ieeg_fname,
            start_time_usec=10000 * 1e6,
            stop_time_usec=10001 * 1e6,
        )
        fs = fs / np.floor(fs / 200)
        fs_array[i_ch] = fs
        fs_array[metadata[metadata.pt == metadata.loc[i_ch, 'pt']].index] = fs
    
    fs = int(fs)
    for i_stage, stage in enumerate(ch):
        ch_stage_signals = np.array(signals[i_ch][i_stage])
        if len(ch_stage_signals) == 0:
            continue
        ch_stage_signals = ch_stage_signals[:, :-int(fs)]

        if len(ch_stage_signals) == 0:
            continue

        bandpowers[i_ch, i_stage] = bandpower.bandpower(
            ch_stage_signals,
            fs, method='welch', relative=True, log_and_offset=False).astype(np.float32)

# %%
# save bandpowers
np.save(
    ospj(CONFIG.paths.data_dir, 'features', 'all_bandpowers.npy'),
    bandpowers
)

# %%
# quality check
# bandpowers = np.load(ospj(CONFIG.paths.data_dir, 'features', 'all_bandpowers.npy'), allow_pickle=True)
len_bandpowers = np.zeros_like(bandpowers, dtype=int)
for i_ch, ch in enumerate(bandpowers):
    for i_stage, stage in enumerate(ch):
        len_bandpowers[i_ch, i_stage] = len(bandpowers[i_ch, i_stage])
    
len_signals = np.zeros_like(signals, dtype=int)
for i_ch, ch in enumerate(signals):
    for i_stage, stage in enumerate(ch):
        len_signals[i_ch, i_stage] = len(signals[i_ch][i_stage])

assert np.all(len_bandpowers == len_signals)

# %%
### COHERENCE ###
all_coherences = {}

for pt, group in tqdm(metadata.groupby('pt'), total=metadata.pt.nunique()):
    pt_signals = signals[group.index]

    fs = fs_array[group.index[0]]
    if np.isnan(fs):
        pt_fs = fs_array[metadata[metadata.pt == metadata.loc[group.index[0], 'pt']].index]
        pt_fs = pt_fs[~np.isnan(pt_fs)]
        fs = np.mean(pt_fs)

    if np.isnan(fs):
        rid = metadata.loc[group.index[0], 'pt']
        ieeg_fnames = CONFIG.metadata_io.patient_tab.loc[rid].ieeg_fname
        ieeg_fname = ieeg_fnames.split(';')[0].strip()
        _, fs = get_iEEG_data(
            CONFIG.ieeg_login.usr,
            CONFIG.ieeg_login.pwd,
            ieeg_fname,
            start_time_usec=10000 * 1e6,
            stop_time_usec=10001 * 1e6,
        )
        fs = fs / np.floor(fs / 200)
        fs_array[group.index] = fs
    
    fs = int(fs)

    for i_stage, stage in enumerate(stages):
        pt_stage_metadata = group.copy()
        pt_stage_signals = pt_signals[:, i_stage]

   
        if np.all([len(sig) == 0 for sig in pt_stage_signals]):
            continue

        # remove any empty signals
        empty_idx = [i for i, sig in enumerate(pt_stage_signals) if len(sig) == 0]

        # if there is a variable number of clip, choose the most number of common clips
        if len(np.unique([len(i) for i in pt_stage_signals])) != 1:
            n_clips_to_choose = np.max([len(i) for i in pt_stage_signals])
            empty_idx.extend([i for i, sig in enumerate(pt_stage_signals) if len(sig) != n_clips_to_choose])

        # remove channels that aren't samples for all clips
        pt_stage_signals = np.delete(pt_stage_signals, empty_idx, axis=0)
        pt_stage_metadata = pt_stage_metadata.drop(pt_stage_metadata.index[empty_idx])

        edge_pair_idx = list(itertools.combinations(pt_stage_metadata.index, 2))

        to_drop = []
        for i_edge, (ch1_idx, ch2_idx) in enumerate(edge_pair_idx):
            if pt_stage_metadata.loc[ch1_idx, 'reg'] == pt_stage_metadata.loc[ch2_idx, 'reg']:
                to_drop.append(i_edge)

        edge_pair_idx = [edge_pair_idx[i] for i in range(len(edge_pair_idx)) if i not in to_drop]
        n_edges = len(edge_pair_idx)
        if n_edges == 0:
            continue

        n_clips = len(pt_stage_signals[0])

        coherences = np.ones((n_edges, n_clips, len(bands)), dtype=np.float16) * np.nan

        for i_clip in range(n_clips):
            pt_stage_clip_signals = np.array([pt_stage_signals[i][i_clip] for i in range(len(pt_stage_signals))])
            pt_stage_clip_signals = pt_stage_clip_signals[:, :-int(fs)]

            cohers = np.zeros((fs + 1, n_edges))
            for i_edge, (ch1_idx, ch2_idx) in enumerate(edge_pair_idx):
                i_ch1 = pt_stage_metadata.index.get_loc(ch1_idx)
                i_ch2 = pt_stage_metadata.index.get_loc(ch2_idx)

                freq, pair_coher = coherence(
                    pt_stage_clip_signals[i_ch1],
                    pt_stage_clip_signals[i_ch2],
                    fs=fs,
                    window="hamming",
                    nperseg=int(fs * 2),
                    noverlap=int(fs * 1),
                )

                cohers[:, i_edge] = pair_coher

            filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
            freq = freq[filter_idx]
            cohers = cohers[filter_idx]

            coher_bands = np.empty((N_BANDS, n_edges))
            coher_bands[-1] = np.mean(cohers, axis=0)

            # format all frequency bands
            for i_band, (_, (lower, upper)) in enumerate(list(band_ranges.items())[:-1]):
                filter_idx = np.logical_and(freq >= lower, freq <= upper)
                coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

            coher_bands = coher_bands.T
            coherences[:, i_clip] = coher_bands

        all_coherences[pt, stage] = {
            'edge_pair_idx': np.array(edge_pair_idx),
            'coherences': coherences
        }

# save as pickle
with open(ospj(CONFIG.paths.data_dir, 'features', 'all_coherences.pkl'), 'wb') as f:
    pickle.dump(all_coherences, f)
