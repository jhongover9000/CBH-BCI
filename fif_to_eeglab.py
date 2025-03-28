import mne
import numpy as np
from scipy.io import savemat
import eeglabio
import mne

# Directory containing .fif files
input_dir = './lsl_data/combined/'
filename = 'MI_ME_20_1'

# Load FIF file
raw = mne.io.read_raw_fif(f'{input_dir}{filename}.fif', preload=True)

# Export to EEGLAB format
raw.export("MI_ME_10_3_eeglab.set", fmt="eeglab")