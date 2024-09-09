import pandas as pd

def _rid_converted(x):
    """
    Converts a list of integers to a list of strings with a specific format.

    Args:
        x (list): A list of integers.

    Returns:
        list: A list of strings with the format "sub-RIDxxxx", where xxxx is the integer value with leading zeros.

    Example:
        >>> _rid_converted([1, 10, 100])
        ['sub-RID0001', 'sub-RID0010', 'sub-RID0100']
    """
    return [f"sub-RID{str(int(i)).zfill(4)}" for i in x]

def _hup_converted(x):
    """
    Converts a list of numbers to a list of strings with a specific format.

    Args:
        x (list): A list of numbers.

    Returns:
        list: A list of strings with the format "HUPXXX", where XXX is the number from the input list, zero-padded to 3 digits.

    Example:
        >>> _hup_converted([1, 2, 10])
        ['HUP001', 'HUP002', 'HUP010']
    """
    converted_li = []
    for i in x:
        if not pd.isna(i):
            converted_li.append(f"HUP{str(int(i)).zfill(3)}")
        else:
            converted_li.append("")
    return converted_li

def get_ignore(metadata_path):
    """
    Reads the metadata from an Excel file and returns a dictionary mapping record IDs to ignore flags.

    Parameters:
    metadata_path (str): The path to the Excel file containing the metadata.

    Returns:
    dict: A dictionary mapping record IDs to ignore flags. The ignore flag is a boolean value indicating whether the record should be ignored or not.
    """
    metadata = pd.read_excel(metadata_path)
    metadata['record_id'] = _rid_converted(metadata['record_id'])

    # fill nan with 0
    metadata['ignore'] = metadata['ignore'].fillna(0)
    metadata['ignore'] = metadata['ignore'].astype(bool)

    return metadata[['record_id', 'ignore']].set_index('record_id').to_dict()['ignore']

def get_rid_to_hup(metadata_path):
    """
    Reads metadata from an Excel file and returns a dictionary mapping record IDs to HUP IDs.

    Args:
        metadata_path (str): The path to the Excel file containing the metadata.

    Returns:
        dict: A dictionary mapping record IDs to HUP IDs.
    """
    metadata = pd.read_excel(metadata_path)
    metadata['record_id'] = _rid_converted(metadata['record_id'])
    metadata['hup_id'] = _hup_converted(metadata['hup_id'])

    # drop na
    metadata = metadata.dropna(subset=['hup_id'])
    metadata = metadata[metadata['hup_id'] != ""]

    return metadata[['record_id', 'hup_id']].set_index('record_id').to_dict()['hup_id']

def get_hup_to_rid(metadata_path):
    """
    Reads metadata from an Excel file and returns a dictionary mapping HUP IDs to record IDs.

    Args:
        metadata_path (str): The path to the Excel file containing the metadata.

    Returns:
        dict: A dictionary mapping HUP IDs to record IDs.
    """
    metadata = pd.read_excel(metadata_path)
    metadata['record_id'] = _rid_converted(metadata['record_id'])
    metadata['hup_id'] = _hup_converted(metadata['hup_id'])

    # drop na
    metadata = metadata.dropna(subset=['hup_id'])
    metadata = metadata[metadata['hup_id'] != ""]

    return metadata[['hup_id', 'record_id']].set_index('hup_id').to_dict()['record_id']


def get_outcome(metadata_path, time=12):
    """
    Retrieves the outcome data from the metadata file.

    Args:
        metadata_path (str): The path to the metadata file.
        time (int, optional): The time in hours (12 or 24). Defaults to 12.

    Returns:
        dict: A dictionary containing the outcome data, with record IDs as keys and outcome values as values.
    """
    if time not in [12, 24]:
        raise ValueError("time must be either 12 or 24")
    
    metadata = pd.read_excel(metadata_path)
    metadata['record_id'] = _rid_converted(metadata['record_id'])

    outcome_col = f"seizure_Engel{time}m"
    metadata = metadata.dropna(subset=[outcome_col])
    metadata = metadata[metadata[outcome_col] != "Unknown"]
    metadata = metadata[metadata[outcome_col] != "NA?"]

    # convert any strings with commas to lists of floats
    if time == 12:
        metadata[outcome_col] = metadata[outcome_col].apply(lambda x: x.split(",") if (type(x) == str) else x)
    if time == 24:
        metadata[outcome_col] = metadata[outcome_col].apply(lambda x: x.split("or") if type(x) == str else x)

    metadata[outcome_col] = metadata[outcome_col].apply(lambda x: [float(i.strip()) for i in x] if type(x) == list else x)

    return metadata[['record_id', outcome_col]].set_index('record_id').to_dict()[outcome_col]
