# imports
import itertools
from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import coherence

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 80),
    "broad": (1, 80)
}
N_BANDS = len(bands)

def coherence_bands(
    data: Union[pd.DataFrame, np.ndarray], fs: float, win_size=2, win_stride=1
) -> np.ndarray:
    """_summary_

    Args:
        data (Union[pd.DataFrame, np.ndarray]): _description_
        fs (float): _description_

    Returns:
        np.ndarray: _description_
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    _, n_channels = data.shape
    n_edges = sum(1 for _ in itertools.combinations(range(n_channels), 2))
    if n_edges == 0:
        return np.zeros((N_BANDS, 0))

    n_freq = int(fs) + 1

    cohers = np.zeros((n_freq, n_edges))
    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        freq, pair_coher = coherence(
            data[:, ch1],
            data[:, ch2],
            fs=fs,
            window="hamming",
            nperseg=int(fs * win_size),
            noverlap=int(fs * win_stride),
        )

        cohers[:, i_pair] = pair_coher

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    cohers = cohers[filter_idx]

    coher_bands = np.empty((N_BANDS, n_edges))
    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (_, (lower, upper)) in enumerate(list(bands.items())[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)
        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    return coher_bands
