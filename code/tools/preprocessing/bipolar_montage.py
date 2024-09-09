import numpy as np
import pandas as pd

def bipolar_montage(data: np.ndarray, ch_types: pd.DataFrame) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): _description_
        ch_types (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """
    n_ch = len(ch_types)
    new_ch_types = []
    for ind, row in ch_types.iterrows():
        # do only if type is ecog or seeg
        if row["type"] not in ["ecog", "seeg"]:
            continue

        ch1 = row["name"]

        ch2 = ch_types.loc[
            (ch_types["lead"] == row["lead"])
            & (ch_types["contact"] == row["contact"] + 1),
            "name",
        ]
        if len(ch2) > 0:
            ch2 = ch2.iloc[0]
            entry = {
                "name": ch1 + "-" + ch2,
                "type": row["type"],
                "idx1": ind,
                "idx2": ch_types.loc[ch_types["name"] == ch2].index[0],
            }
            new_ch_types.append(entry)

    new_ch_types = pd.DataFrame(new_ch_types)
    # apply montage to data
    new_data = np.empty((len(new_ch_types), data.shape[1]))
    for ind, row in new_ch_types.iterrows():
        new_data[ind, :] = data[row["idx1"], :] - data[row["idx2"], :]

    return new_data, new_ch_types
