# BCI FUNCTIONS
# Joseph Hong
# Description: Supplementary code for functions. Assuming that 
# the data has channels dropped and is at the desired sfreq.

# =============================================================
# =============================================================
# INCLUDES
from datetime import datetime
import numpy as np
import mne

# =============================================================
# =============================================================
# FUNCTIONS
def preprocess_raw(data_buffer, f_list, info, new_freq, is_lsl, is_verbose):

    # Copy & Convert Data Buffer into MNE array
    mne_array = mne.io.RawArray(data_buffer, info, verbose = is_verbose)

    # Add Reference Channel (Livestream & Emulator)
    if not (is_lsl):
        mne_array = mne_array.add_reference_channels(ref_channels=["FCz"], verbose = is_verbose)
        mne_array = mne_array.set_montage('standard_1020')
        if is_verbose:
            print("Finished Adding Reference Channel:", datetime.now())
            print("")

    # Bandpass
    mne_array = mne_array.filter(l_freq=f_list[0], h_freq=f_list[1], fir_design='firwin', verbose = is_verbose)
    if is_verbose:
        print("Finished Bandpass Filtering:", datetime.now())
        print("")

    # Re-Reference
    mne_array = mne_array.set_eeg_reference(ref_channels='average', projection=False, verbose = is_verbose)
    if is_verbose:
        print("Finished re-referencing:", datetime.now())
        print("")

    # Downsample if LSL to match 7-electrode data (THIS IS HARDCODED)
    if is_lsl:
        mne_array = mne_array.resample(new_freq, verbose = is_verbose)
        if is_verbose:
            print(f"Finished resampling to {new_freq}Hz:", datetime.now())
            print("")

    # Baseline Correction
    # mneDataFiltered.apply_function(lambda x: x - np.median(x), picks='eeg')

    # NEED TO CHECK THE VOLTAGE OF THE SIGNALS
    mne_array= mne_array.get_data()

    #add dimensions to match the input layer of the model
    mne_array = np.expand_dims(mne_array, axis=0)
    mne_array = np.expand_dims(mne_array, axis=0)
    
    return mne_array
