import re
import pandas as pd

def clean_labels(channel_li: list, pt: str, ch_harmon_pt="/mnt/leif/littlab/users/pattnaik/ictal_patterns/data/metadata/channel_harmonize.xlsx") -> list:
    """This function cleans a list of channels and returns the new channels

    Args:
        channel_li (list): _description_

    Returns:
        list: _description_
    """
    channel_harm = pd.read_excel(
        ch_harmon_pt
    )

    new_channels = []
    for i in channel_li:
        i = i.replace("-", "")
        i = i.replace("GRID", "G")  # mne has limits on channel name size
        # sub-RID0042 has a ridiculous corner case
        if pt in ("HUP112_phaseII", "HUP112", "sub-RID0042"):
            i = i.replace("LAF14", "LAF")
            i = i.replace("LAF15", "LPF")
            i = i.replace("LPF15", "LPF")

        # standardizes channel names
        regex_match = re.match(r"(\D+)(\d+)", i)
        if regex_match is None:
            new_channels.append(i)
            continue
        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))

        if pt in channel_harm[['rid', 'hup', 'ieeg_fname']].values:
            # find the rows that match the pt
            pt_rows = channel_harm[
                (channel_harm.rid == pt)
                | (channel_harm.hup == pt)
                | (channel_harm.ieeg_fname == pt)
            ]

            old_to_new = pt_rows[['old', 'new']]
            # dict
            old_to_new = dict(zip(old_to_new.old, old_to_new.new))
            if lead in old_to_new.keys():
                lead = old_to_new[lead]

        if (pt in ("HUP112_phaseII", "HUP112", "sub-RID0042")) and ("-" in i):
                new_channels.append(f"{lead}{contact:02d}-{i.strip().split('-')[-1]}")
                continue
        
        if pt in ("HUP116_phaseII", "HUP116", "sub-RID0175"):
            new_channels.append(f"{lead}{contact:02d}".replace("-", ""))
            continue

        new_channels.append(f"{lead}{contact:02d}")

    return new_channels
