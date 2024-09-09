import sys
from os.path import join as ospj
import pandas as pd

# suppress setting with copy warning
pd.options.mode.chained_assignment = None

from tools.features import bandpower, coherence_bands

class DeviceConfig:
    """Device configuration class."""
    is_interactive = hasattr(sys, "ps1")

class Paths:
    """Paths class."""
    bids_dir = ""
    spikes_dir = ""
    data_dir = ""
    fig_dir = ""
    log_dir = ""
    channel_harmonization_file = ""
    bids_inventory_file = ""
    metadata_file = ""

class Plotting:
    """Plotting class."""
    kwargs = dict(
        scalings=dict(eeg=2e-4),
        show_scrollbars=False,
        show=True,
        block=True,
        verbose=False,
        n_channels=20,
        duration=30,
    )
    stage_colors = {
        "N2": "#6cedf9",
        "N3": "#45768e",
        "R": "#0a293a",
        "W": "#e37720",
    }
    elec_type_colors = {
        'norm MNI': 'limegreen',
        'norm HUP': 'darkgreen',
        'irritative': 'darkorange',
        'soz': 'firebrick',
    }
    outcome_colors = {
        'good': '#9575cd',
        # 'poor': '#FFA732',
        'poor': '#d23600'
    }


class IeegLogin:
    """ieeg.org login class."""
    usr = ""
    pwd = ""

class Constants:
    """Constants class."""
    clip_size = 60  # seconds
    spike_thresh = 2

class MetadataIO:
    """Metadata I/O class."""
    rid_hup_musc_table = pd.read_csv(
        ospj(Paths.data_dir, "metadata/rid_hup_table.csv"), index_col=0
    )
    rid_hup_table = rid_hup_musc_table.dropna(subset=["hupsubjno"])
    for ind, row in rid_hup_table.iterrows():
        rid_hup_table.loc[ind, "hupsubjno"] = int(row["hupsubjno"][:3])
    del ind, row
    rid_hup_table.index = [f"sub-RID{x:04d}" for x in rid_hup_table.index]
    rid_hup_table['hupsubjno'] = [f"HUP{x:03d}" for x in rid_hup_table['hupsubjno']]

    rid_to_hup = rid_hup_table['hupsubjno'].to_dict()
    hup_to_rid = {v: k for k, v in rid_to_hup.items()}

    rid_musc_table = rid_hup_musc_table.dropna(subset=["muscsubjno"])
    rid_musc_table.index = [f"sub-RID{x:04d}" for x in rid_musc_table.index]

    rid_to_musc = rid_musc_table['muscsubjno'].to_dict()
    musc_to_rid = {v: k for k, v in rid_to_musc.items()}

    del rid_hup_table

    interictal_metadata = pd.read_excel(
        ospj(Paths.data_dir, "metadata/atlas_metadata_final_updated_interictal.xlsx")
    )

    sz_times = pd.read_excel(
        ospj(Paths.data_dir, "metadata/Manual validation.xlsx"),
        sheet_name="AllSeizureTimes",
        index_col=0,
    )
    sz_times.dropna(inplace=True, subset=["IEEGname"])

    soz_metadata = pd.read_excel(
        ospj(Paths.data_dir, "metadata/Manual validation.xlsx"), sheet_name="SOZ"
    )

    patient_tab = pd.read_excel(
        ospj(Paths.data_dir, "metadata/master_pt_table_manual.xlsx"),
        index_col=0
    )

    ch_info = pd.read_csv(ospj(Paths.data_dir, 'mni', 'Information', 'ChannelInformation.csv'))
    ch_info['Channel name'] = ch_info['Channel name'].str[1:-1]
    ch_info['Electrode type'] = ch_info['Electrode type'].str[1:-1]
    ch_info['Hemisphere'] = ch_info['Hemisphere'].str[1:-1]

    reg_info = pd.read_csv(ospj(Paths.data_dir, 'mni', 'Information', 'RegionInformation.csv'))
    reg_info['Region name'] = reg_info['Region name'].str[1:-1]

    pt_info = pd.read_csv(ospj(Paths.data_dir, 'mni', 'Information', 'PatientInformation.csv'))

    dkt_mni_parcs = pd.read_excel(ospj(Paths.data_dir, 'metadata', 'dkt_mni_parcs_RG.xlsx'), header=None, sheet_name='Sheet1')
    dkt_custom = dkt_mni_parcs.iloc[:, [0, 2]]
    mni_custom = dkt_mni_parcs.iloc[:, [7, 9]]

    dkt_custom.columns = ['dkt', 'custom']
    mni_custom.columns = ['mni', 'custom']

    dkt_custom = dkt_custom[dkt_custom['dkt'].str.startswith('Label')]
    dkt_custom['dkt_id'] = dkt_custom['dkt'].str.split(':').str[0].str.strip()
    dkt_custom['dkt_id'] = dkt_custom['dkt_id'].str.split(' ').str[1].str.strip().astype(int)
    dkt_custom['dkt'] = dkt_custom['dkt'].str.split(':').str[1].str.strip()

    mni_custom['mni'] = mni_custom['mni'].str.strip()
    mni_custom['mni'] = mni_custom['mni'].str.strip("'")

    dkt_custom.dropna(inplace=True)
    mni_custom.dropna(inplace=True)

    dkt_to_custom = dict(zip(dkt_custom['dkt'], dkt_custom['custom']))
    mni_to_custom = dict(zip(mni_custom['mni'], mni_custom['custom']))

    del dkt_mni_parcs

class Bands:
    """Bands class."""
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 80),
        "broad": (1, 80)
    }

class FeatureOptions:
    """Feature options class."""
    options = {
        'bandpower': {
            'func': bandpower,
            'kwargs': {'fs': 200},
            'name': 'bandpower',
            'ft_names': list(Bands.bands.keys()),
            'ft_units': 'uV^2',
            'ft_type': 'spectral',
            'n_features': len(Bands.bands)
        },
        'coherence': {
            'func': coherence_bands,
            'kwargs': {'fs': 200},
            'name': 'coherence',
            'ft_names': list(Bands.bands.keys()),
            'ft_units': 'uV^2',
            'ft_type': 'spectral',
            'n_features': len(Bands.bands)
        },
    }

class CONFIG:
    """Configuration class for the project."""
    device = DeviceConfig()
    paths = Paths()
    plotting = Plotting()
    ieeg_login = IeegLogin()
    constants = Constants()
    metadata_io = MetadataIO()
    bands = Bands()
    feature_options = FeatureOptions()
