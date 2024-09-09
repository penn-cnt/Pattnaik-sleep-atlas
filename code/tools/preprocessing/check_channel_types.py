import re
import pandas as pd

def check_channel_types(ch_list, threshold=15):
    """Function to check channel types

    Args:
        ch_list (_type_): _description_
        threshold (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    ch_df = []
    for i in ch_list:
        regex_match = re.match(r"(\D+)(\d+)", i)
        if regex_match is None:
            ch_df.append({"name": i, "lead": i, "contact": 0, "type": "misc"})
            continue
        lead = regex_match.group(1)
        contact = int(regex_match.group(2))
        ch_df.append({"name": i, "lead": lead, "contact": contact})
    ch_df = pd.DataFrame(ch_df)
    for lead, group in ch_df.groupby("lead"):
        if lead in ["ECG", "EKG"]:
            ch_df.at[group.index, "type"] = "ecg"
            continue
        if lead in [
            "C",
            "Cz",
            "CZ",
            "F",
            "Fp",
            "FP",
            "Fz",
            "FZ",
            "O",
            "P",
            "Pz",
            "PZ",
            "T",
        ]:
            ch_df.at[group.index, "type"] = "eeg"
            continue
        if len(group) > threshold:
            ch_df.at[group.index, "type"] = "ecog"
        else:
            ch_df.at[group.index, "type"] = "seeg"
    return ch_df
