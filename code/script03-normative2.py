# %%
# standard library imports
import os
from os.path import join as ospj
from glob import glob
import itertools
import json
import multiprocessing as mp
import pickle
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tools.config import CONFIG
from tools.io.metadata import get_ignore

USE_PARAM_JSON = True
RUN_BANDPOWER = True
RUN_COHERENCE = True

# %%
# constants
stages = ['N2', 'N3', 'R', 'W']
feature_types = ['bandpower', 'coherence']
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']

pts_to_ignore = pd.Series(get_ignore(CONFIG.paths.metadata_file))
pts_to_ignore = pts_to_ignore[pts_to_ignore].index
bad_sleepers = np.load(ospj(CONFIG.paths.data_dir, 'metadata', 'bad_sleep_stage_patients.npy'), allow_pickle=True)
features_dir = ospj(CONFIG.paths.data_dir, 'features')

ignore_pts = np.union1d(pts_to_ignore, bad_sleepers)

spike_threshold = 1
min_normative_bandpower = 10
min_normative_coherence = 4 # 10
engel_time = 12
engel_threshold = 1.1

if USE_PARAM_JSON:
    with open(ospj(os.getcwd(), 'normative_parameters.json'), 'r') as f:
        params_dict = json.load(f)

else:
    params_dict = {"": {
        "spike_threshold": 1,
        "min_normative_bandpower": 10,
        "min_normative_coherence": 4,
        "engel_time": 12,
        "engel_threshold": 1.1
    }}

metadata = pd.read_csv(ospj(CONFIG.paths.data_dir, 'metadata/combined_atlas_metadata.csv'), index_col=0)
n_channels = len(metadata)

# turn params dict of dicts into a list of dicts
params_li = [{k: v} for k, v in params_dict.items()]

# %%
# load in bandpower and coherence features
bandpower = np.load(ospj(CONFIG.paths.data_dir, "features", "all_bandpowers.npy"), allow_pickle=True)
with open(ospj(CONFIG.paths.data_dir, "features", "all_coherences.pkl"), "rb") as f:
    coherences = pickle.load(f)

# %%
if RUN_BANDPOWER:
    for params in params_li:
        i_params, params = list(params.items())[0]

        normative_bandpower = np.zeros_like(bandpower)
        for i in range(len(bandpower)):
            for j in range(len(bandpower[i])):
                normative_bandpower[i, j] = []

        ## Get normative
        normative_rows = metadata[
            ((metadata.site == 'HUP') &
            (metadata.pt.isin(ignore_pts) == False) &
            (metadata.ch1_spike_rate < params["spike_threshold"]) & (metadata.ch2_spike_rate < params["spike_threshold"]) &
            (metadata.ch1_resected == False) & (metadata.ch2_resected == False) &
            (metadata.ch1_soz == False) & (metadata.ch2_soz == False) &
            (metadata[f'engel_{params["engel_time"]}'] <= params["engel_threshold"])) |
            (metadata.site == 'MNI')
        ]
        normative_hup_rows = normative_rows[normative_rows.site == 'HUP']
        normative_mni_rows = normative_rows[normative_rows.site == 'MNI']

        irritative_rows = metadata[
            (metadata.site == 'HUP') &
            ((metadata.ch1_spike_rate >= params["spike_threshold"]) | (metadata.ch2_spike_rate >= params["spike_threshold"])) &
            (metadata.ch1_resected == False) & (metadata.ch2_resected == False) &
            (metadata.ch1_soz == False) & (metadata.ch2_soz == False) &
            (metadata[f'engel_{params["engel_time"]}'] <= params["engel_threshold"])
        ]

        resected_soz_rows = metadata[
            (metadata.site == 'HUP') &
            (metadata.ch1_resected == True) | (metadata.ch2_resected == True) &
            (metadata.ch1_soz == True) | (metadata.ch2_soz == True) &
            (metadata[f'engel_{params["engel_time"]}'] <= params["engel_threshold"])
        ]


        for reg, group in metadata.groupby('reg'):
            reg_bandpower = bandpower[group.index]

            is_normative = group.index.isin(normative_rows.index)
            pt_names = group.pt

            for pt in np.unique(pt_names):
                # leave one patient out
                train_idx = np.where(pt_names != pt)[0]
                test_idx = np.where(pt_names == pt)[0]

                for i_stage, stage in enumerate(stages):
                    # get all clip bandpowers for this stage
                    stage_bandpower = reg_bandpower[:, i_stage]

                    # expand the row numbers to account for multiple clips per channel
                    group_idx = np.array(list(itertools.chain(*[[i] * len(b) for i, b in zip(group.index, stage_bandpower)])))
                    # expand the bandpowers to account for multiple clips per channel
                    stage_bandpower = np.array(list(itertools.chain(*stage_bandpower)))

                    # compute means and stds for each band and z score the bandpowers
                    normative_train_indices = np.isin(group_idx, group.index[train_idx]) & np.isin(group_idx, normative_rows.index)
                    if np.sum(normative_train_indices) < params["min_normative_bandpower"]:
                        continue

                    means = np.mean(stage_bandpower[normative_train_indices], axis=0)
                    stds = np.std(stage_bandpower[normative_train_indices], axis=0)

                    z_scores = (stage_bandpower[np.isin(group_idx, group.index[test_idx])] - means) / stds

                    # combine z_scores for multiple clips per channel
                    z_scores = pd.DataFrame(
                        z_scores,
                        index=group_idx[np.isin(group_idx, group.index[test_idx])],
                        columns=bands
                    )
                    # add z_scores to normative_bandpower
                    for ind, row in z_scores.iterrows():
                        normative_bandpower[ind, i_stage].append(row.values)

        np.save(
            ospj(CONFIG.paths.data_dir, 'features', 'multi_clip_features', f"bandpower_z_{i_params}.npy"),
            normative_bandpower,
            allow_pickle=True
        )

# %%
# takes about 15 minutes
if not os.path.exists(ospj(CONFIG.paths.data_dir, 'features', 'multi_clip_features', 'coherence_table.pkl')):
    coherence_df = []
    for (pt, stage), value in tqdm(coherences.items(), desc='coherence', total=len(coherences)):
        edges = value['edge_pair_idx']
        coherence_vals = value['coherences']

        for i_band, band in enumerate(bands):
            for i_edge, edge in enumerate(edges):
                reg_pair = metadata.loc[edge, 'reg'].values
                reg_pair = sorted(reg_pair)
                reg_pair = '-'.join(reg_pair)

                coherence_df.append({
                    'pt': pt,
                    'stage': stage,
                    'band': band,
                    'edge': reg_pair,
                    'coherence': list(coherence_vals[i_edge, :, i_band]),
                    'ch1': edge[0],
                    'ch2': edge[1]
                })

    coherence_df = pd.DataFrame(coherence_df)
    # save
    coherence_df.to_pickle(ospj(CONFIG.paths.data_dir, 'features', 'multi_clip_features', 'coherence_table.pkl'))
else:
    coherence_df = pd.read_pickle(ospj(CONFIG.paths.data_dir, 'features', 'multi_clip_features', 'coherence_table.pkl'))

# %%
def normative_coh_calculator(coherence_table, pt, stage, band, i_params):
    # save channel_abnormalities
    fname = ospj(
        CONFIG.paths.data_dir,
        'features',
        'multi_clip_features',
        f"coherence_z_{i_params}",
        f"{pt}_{stage}_{band}.pkl"
    )
    # if os.path.exists(fname):
    #     return

    target_table = coherence_table[
        (coherence_table.pt == pt) &
        (coherence_table.stage == stage) &
        (coherence_table.band == band)
    ]

    for edge, edge_group in target_table.groupby('edge'):
        normative_distribution = coherence_table[
            (coherence_table.pt != pt) &
            (coherence_table.stage == stage) &
            (coherence_table.band == band) &
            (coherence_table.edge == edge) &
            (coherence_table.normative)
        ].coherence.values
        normative_distribution = np.array(list(itertools.chain(*normative_distribution)))

        if len(normative_distribution) < min_normative_coherence:
            continue

        mean = np.mean(normative_distribution, axis=0)
        std = np.std(normative_distribution, axis=0)

        target_table.loc[edge_group.index, 'z_score'] = edge_group.coherence.apply(lambda x: [(i - mean) / std for i in x])

    # channel_abnormalities = {}
    # for ch in np.unique(np.concatenate([target_table.ch1, target_table.ch2])):
    #     z_scores = target_table[np.isin(target_table.ch1, ch) | np.isin(target_table.ch2, ch)].z_score.values

    #     # remove empty lists
    #     z_scores = [val for val in z_scores if len(val) > 0]
    #     z_scores = np.array(z_scores)

    #     if z_scores.ndim < 1:
    #         channel_abnormalities[ch] = np.ones(len(bands)) * np.nan
    #     else:
    #         abnormality = np.percentile(z_scores, 75, axis=0)
    #         channel_abnormalities[ch] = abnormality

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as f:
        # pickle.dump(channel_abnormalities, f)
        pickle.dump(target_table, f)

## Multiprocessing
if RUN_COHERENCE:
    to_fix = pd.read_csv(ospj(CONFIG.paths.data_dir, 'features', 'multi_clip_features', 'coherence_to_fix.csv'))

    for params in tqdm(params_li, desc='coherence (params iteration)', total=len(params_li)):
        i_params, params = list(params.items())[0]
        coherence_table = pd.read_pickle(ospj(CONFIG.paths.data_dir, 'features', 'multi_clip_features', 'coherence_table.pkl'))

        ## Get normative
        normative_rows = metadata[
            ((metadata.site == 'HUP') &
            (metadata.pt.isin(ignore_pts) == False) &
            (metadata.ch1_spike_rate < params["spike_threshold"]) & (metadata.ch2_spike_rate < params["spike_threshold"]) &
            (metadata.ch1_resected == False) & (metadata.ch2_resected == False) &
            (metadata.ch1_soz == False) & (metadata.ch2_soz == False) &
            (metadata[f'engel_{params["engel_time"]}'] <= params["engel_threshold"])) |
            (metadata.site == 'MNI')
        ]

        coherence_table['normative'] = coherence_table.ch1.isin(normative_rows.index) & coherence_table.ch2.isin(normative_rows.index)
        coherence_table['z_score'] = [[]] * len(coherence_table)
        
        args = itertools.product(
            [coherence_table],
            coherence_table.pt.unique(),
            stages,
            bands,
            [i_params]
        )
        args = to_fix[to_fix.params == int(i_params)][['pt', 'stage', 'band']].values
        # prepend coherence_table to each row
        args = [(coherence_table,) + tuple(row) + (i_params,) for row in args]

        with mp.Pool(16) as pool:
            pool.starmap(normative_coh_calculator, args)

# %%
# Example plots
# target_reg = "Temporal_Mid_L"
# target_bandpower = bandpower[metadata.reg == target_reg]
# labels = np.empty((len(target_bandpower)), dtype=object)

# labels[metadata[metadata.reg == target_reg].index.isin(normative_hup_rows.index)] = "norm HUP"
# labels[metadata[metadata.reg == target_reg].index.isin(normative_mni_rows.index)] = "norm MNI"
# labels[metadata[metadata.reg == target_reg].index.isin(irritative_rows.index)] = "irritative"
# labels[metadata[metadata.reg == target_reg].index.isin(resected_soz_rows.index)] = "soz"


# for i_stage, stage in enumerate(stages):
#     target_bandpower_stage = target_bandpower[:, i_stage]
#     stage_labels = labels.copy()

#     # flatten target_bandpower_stage (which is array of lists) and extend stage_labels to match size
#     stage_labels = np.array(list(itertools.chain(*[[l] * len(b) for l, b in zip(stage_labels, target_bandpower_stage)])))
#     target_bandpower_stage = np.array(list(itertools.chain(*target_bandpower_stage)))

#     df = {'label': stage_labels}
#     for i_band, band in enumerate(bands):
#         df[band] = target_bandpower_stage[:, i_band]
#     df = pd.DataFrame(df)

#     fig, axes = plt.subplots(1, len(bands), figsize=(len(bands) * 3, 3))
#     for i_band, band in enumerate(bands):
#         sns.violinplot(x='label', y=band, data=df, ax=axes[i_band], palette=CONFIG.plotting.elec_type_colors)
#         axes[i_band].set_title(band)
#         if i_band == 0:
#             axes[i_band].set_ylabel('Bandpower')
#         else:
#             axes[i_band].set_ylabel('')
#         axes[i_band].set_xlabel('')
#         axes[i_band].set_xticklabels(axes[i_band].get_xticklabels(), rotation=45, horizontalalignment='right')
        
    
#     sns.despine()
    