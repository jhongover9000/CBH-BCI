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
import warnings

# =============================================================
# =============================================================
# FUNCTIONS
def preprocess_raw(data_buffer, f_list, info, new_freq, is_lsl, is_verbose):



    # Copy & Convert Data Buffer into MNE array
    mne_array = mne.io.RawArray(data_buffer, info, verbose = is_verbose)

    montage = mne.channels.make_standard_montage('standard_1020')

    # Retain only channels that are part of the standard montage
    # valid_ch_names = montage.ch_names
    # common_ch_names = list(set(mne_array.ch_names) & set(valid_ch_names))
    mne_array.set_montage(montage)

    # Downsample if LSL to match 7-electrode data (THIS IS HARDCODED)
    if is_lsl:
        mne_array = mne_array.resample(new_freq, verbose = is_verbose)
        if is_verbose:
            print(f"Finished resampling to {new_freq}Hz:", datetime.now())
            print("")
        

    # Bandpass
    mne_array = mne_array.filter(l_freq=f_list[0], h_freq=f_list[1], fir_design='firwin', verbose = is_verbose)
    if is_verbose:
        print("Finished Bandpass Filtering:", datetime.now())
        print("")

    if is_lsl:
        # Optionally, mark 'Cz' as bad and interpolate it if present
        if 'Cz' in mne_array.ch_names:
            mne_array.info['bads'] = ['Cz']     
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mne_array.interpolate_bads(reset_bads=True,  verbose = is_verbose)
            # mne_array.drop_channels('Cz')

    # Set common average reference if reference channels exist
    ref_channels = ["A1", "A2"]
    available_ref = [ch for ch in ref_channels if ch in mne_array.ch_names]
    if available_ref:
        mne_array.set_eeg_reference(ref_channels=available_ref)
    else:
        mne_array.set_eeg_reference('average', verbose = is_verbose)
        if is_verbose:
            print("Finished re-referencing:", datetime.now())
            print("")
        

    

    # Baseline Correction
    # mneDataFiltered.apply_function(lambda x: x - np.median(x), picks='eeg')

    # NEED TO CHECK THE VOLTAGE OF THE SIGNALS
    mne_array= 1e6 * mne_array.get_data()

    #add dimensions to match the input layer of the model
    mne_array = np.expand_dims(mne_array, axis=0)
    mne_array = np.expand_dims(mne_array, axis=0)
    
    return mne_array
