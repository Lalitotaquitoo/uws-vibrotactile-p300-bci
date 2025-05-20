import os
import numpy as np
from scipy.io import loadmat
import mne


def create_epochs(mat_path, channels=None, sfreq=256, l_freq=1, h_freq=12,
                  tmin=-0.1, tmax=0.6):
    """
    Load .mat EEG data and return MNE Epochs object.
    """
    if channels is None:
        channels = ['Fz', 'C3', 'Cz', 'C4', 'CP1', 'CPz', 'CP2', 'Pz']

    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types='eeg')
    info.set_montage('standard_1020')

    data = loadmat(mat_path)
    eeg_data = data['y']  # shape: (n_channels, n_times)
    raw = mne.io.RawArray(eeg_data.T, info)

    # Add stim channel
    stim = data['trig']  # shape: (1, n_times)
    stim_raw = mne.io.RawArray(stim.T, mne.create_info(['STI'], sfreq, ['stim']))
    raw.add_channels([stim_raw], force_update_info=True)

    # Filtering
    raw.notch_filter(freqs=50)
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Events and epochs
    events = mne.find_events(raw, stim_channel='STI')
    event_id = {'nontarget': 1, 'target': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, preload=True)
    return epochs