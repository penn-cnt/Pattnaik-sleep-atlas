# imports
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
from mne.time_frequency import psd_array_multitaper


def bandpower(x: np.ndarray, fs: float, lo=1, hi=120, relative=True, win_size=2, win_stride=1, log_and_offset=True, method='multitaper') -> np.array:
    """
    Calculates the relative bandpower of a signal x, using a butterworth filter of order 'order'
    and bandpass filter between lo and hi Hz.

    Use scipy.signal.welch and scipy.signal.simpson
    """
    # clip the data to remove the first and last 2.5 seconds
    x = x[:, int(2.5*fs):-int(2.5*fs)]

    all_nans = np.all(np.isnan(x), axis=1)

    # fill nans with 0
    x = np.nan_to_num(x)

    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}

    nperseg = int(win_size * fs)
    noverlap = int(win_stride * fs)

    if method == 'welch':
        freq, pxx = welch(x=x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1, window='hamming', average='mean')
    elif method == 'multitaper':
        pxx, freq = psd_array_multitaper(x, fs, adaptive=False, normalization='full', verbose=0, fmin=lo, fmax=hi, n_jobs=1)
    else:
        raise ValueError(f"Method {method} not recognized")
    
    if log_and_offset:
        # log transform the power spectrum
        pxx = 10 * np.log10(pxx, where=(pxx > 0))
        # offset so it's greater than 0
        pxx = pxx + np.abs(np.min(pxx, axis=1))[:, None]

    all_bands = np.zeros((pxx.shape[0], len(bands)+1))
    for i, (band, (lo, hi)) in enumerate(bands.items()):
        idx_band = np.logical_and(freq >= lo, freq <= hi)
        bp = simpson(pxx[:, idx_band], dx=freq[1] - freq[0])

        # from nishant
        # bp = np.log10(bp + 1)

        # relative
        # if relative:
        #     lo, hi = bands["delta"][0], bands["gamma"][1]
        #     idx_band = np.logical_and(freq >= lo, freq <= hi)
        #     bp /= simpson(pxx[:, idx_band], dx=freq[1] - freq[0])
        all_bands[:, i] = bp

    # relative (second method)
    if relative:
        with np.errstate(divide='ignore', invalid='ignore'):
            all_bands = all_bands / all_bands.sum(axis=1)[:, None]

    # Broadband
    lo, hi = bands["delta"][0], bands["gamma"][1]
    idx_band = np.logical_and(freq >= lo, freq <= hi)
    all_bands[:, -1] = np.log10(
        simpson(pxx[:, idx_band], dx=freq[1] - freq[0]),
        where=(simpson(pxx[:, idx_band], dx=freq[1] - freq[0]) > 0)
        )

    all_bands[all_nans] = np.nan

    return all_bands
    # return bp
    # return data_filt
