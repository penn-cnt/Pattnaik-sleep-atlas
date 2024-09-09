# imports
import time
import logging
import numpy as np
import pandas as pd
import pickle
from numbers import Number
from ieeg.auth import Session
from ..metadata.clean_labels import clean_labels

def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    """
    Pull data while handling iEEGConnectionError
    """
    i = 0
    while True:
        if i == 50:
            logger = logging.getLogger()
            logger.error(
                f"failed to pull data for {ds.name}, {start_usec / 1e6}, {duration_usec / 1e6}, {len(channel_ids)} channels"
            )
            return None
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as _:
            time.sleep(1)
            i += 1


def get_iEEG_data(
    username: str,
    password_bin_file: str,
    iEEG_filename: str,
    start_time_usec: float,
    stop_time_usec: float,
    select_electrodes=None,
    ignore_electrodes=None,
    outputfile=None,
):
    """_summary_

    Args:
        username (str): _description_
        password_bin_file (str): _description_
        iEEG_filename (str): _description_
        start_time_usec (float): _description_
        stop_time_usec (float): _description_
        select_electrodes (_type_, optional): _description_. Defaults to None.
        ignore_electrodes (_type_, optional): _description_. Defaults to None.
        outputfile (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        pwd = f.read()

    iter = 0
    while True:
        try:
            if iter == 50:
                raise ValueError("Failed to open dataset")
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
            
        except Exception as e:
            time.sleep(1)
            iter += 1

            if iter == 50:
                raise ValueError("Failed to open dataset")

    all_channel_labels = clean_labels(all_channel_labels, iEEG_filename)

    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, iEEG_filename)
            if any([i not in all_channel_labels for i in select_electrodes]):
                raise ValueError("Channel not in iEEG")

            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [
                i
                for i in np.arange(len(all_channel_labels))
                if i not in ignore_electrodes
            ]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes, iEEG_filename)
            channel_ids = [
                i
                for i, e in enumerate(all_channel_labels)
                if e not in ignore_electrodes
            ]
            channel_names = [
                e for e in all_channel_labels if e not in ignore_electrodes
            ]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    # if clip is small enough, pull all at once, otherwise pull in chunks
    if (duration < 120 * 1e6) and (len(channel_ids) < 100):
        data = _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    elif (duration > 120 * 1e6) and (len(channel_ids) < 100):
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                new_data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
                data = np.concatenate((data, new_data), axis=0)
            clip_start = clip_start + clip_size

        last_clip_size = stop_time_usec - clip_start
        new_data = _pull_iEEG(ds, clip_start, last_clip_size, channel_ids)
        data = np.concatenate((data, new_data), axis=0)
    else:
        # there are too many channels, pull chunks and concatenate
        channel_size = 20
        channel_start = 0
        data = None
        while channel_start + channel_size < len(channel_ids):
            if data is None:
                data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
            else:
                new_data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
                data = np.concatenate((data, new_data), axis=1)
            channel_start = channel_start + channel_size

        last_channel_size = len(channel_ids) - channel_start
        new_data = _pull_iEEG(
            ds,
            start_time_usec,
            duration,
            channel_ids[channel_start : channel_start + last_channel_size],
        )
        data = np.concatenate((data, new_data), axis=1)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate

    if outputfile:
        with open(outputfile, "wb") as f:
            pickle.dump([df, fs], f)
    else:
        return df, fs

