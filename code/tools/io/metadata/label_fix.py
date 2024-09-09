from glob import glob
import pandas as pd
import numpy as np

SKIP_REGS = [
    "EmptyLabel",
    "cerebellum"
]

def label_fix(rid, threshold=0.3):
    """
    label_fix reassigns labels overlapping brain regions to "empty labels" in our DKTantspynet output from IEEG_recon
    input:  rid - name of patient. example: 'sub-RID0031'
            data_directory - directory containing CNT_iEEG_BIGS folder. (must end in '/')
            threshold - arbitrary threshold that r=2mm surround of electrode must overlap with a brain region. default: threshold = 25%, Brain region has a 25% or more overlap.
    output: relabeled_df - a list that contains the most overlapping brain region.
    """
    try:
        json_labels = glob(
            f"/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS/{rid}/derivatives/ieeg_recon/module3/{rid}_ses-*_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json"
        )[0]
    except IndexError:
        json_labels = glob(
            f"/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS/{rid}/derivatives/ieeg_recon_old/module3/{rid}_ses-*_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json"
        )[0]
    try:
        json_labels = pd.read_json(json_labels, lines=True)
    except ValueError:
        json_labels = pd.read_json(json_labels)

    labels_sorted = json_labels.labels_sorted.values
    percent_assigned = json_labels.percent_assigned.values

    final_labels = []

    for lab, pct in zip(labels_sorted, percent_assigned):
        candidate_labs = np.array(lab)[np.array(pct) > threshold]

        # remove to_skip labels from candidate_labs
        candidate_labs_new = [l for l in candidate_labs if l not in SKIP_REGS]
        if len(candidate_labs_new) > 0:
            candidate_labs = candidate_labs_new
        else:
            candidate_labs = candidate_labs

        final_labels.append(candidate_labs[0])

    final_labels = np.array([i.strip() for i in final_labels])

    return final_labels