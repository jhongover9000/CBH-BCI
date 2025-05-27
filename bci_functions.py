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

    # Remove Specific Channels (Set one to False if using the other)
    drop_chan = True
    chan2drop = ['A1', 'A2', 'X5', 'TP10', 'TP9', 'FT10', 'FT9']

    # Copy & Convert Data Buffer into MNE array
    mne_array = mne.io.RawArray(data_buffer, info, verbose = is_verbose)

    montage = mne.channels.make_standard_montage('standard_1020')

    # Retain only channels that are part of the standard montage
    # valid_ch_names = montage.ch_names
    # common_ch_names = list(set(mne_array.ch_names) & set(valid_ch_names))
    mne_array.set_montage(montage)

    original_ch_names = mne_array.ch_names

    # Drop channels
    if drop_chan and chan2drop:
        ch_to_drop_present = [ch for ch in chan2drop if ch in original_ch_names]
        if len(ch_to_drop_present) < len(chan2drop):
            missing = set(chan2drop) - set(ch_to_drop_present)
            print(f"  Warning: Channels requested for dropping not found: {missing}. Dropping available: {ch_to_drop_present}")
        if ch_to_drop_present:
            print(f"  Dropping channels: {ch_to_drop_present}")
            mne_array.drop_channels(ch_to_drop_present)
        else:
            print(f"  No channels to drop from list {chan2drop} were found.")

    # Downsample if LSL to match 7-electrode data (THIS IS HARDCODED)
    mne_array = mne_array.resample(new_freq, verbose = is_verbose)
    if is_verbose:
        print(f"Finished resampling to {new_freq}Hz:", datetime.now())
        print("")
        

    # Bandpass
    mne_array = mne_array.filter(l_freq=f_list[0], h_freq=f_list[1], fir_design='firwin', verbose = is_verbose)
    if is_verbose:
        print("Finished Bandpass Filtering:", datetime.now())
        print("")

    # mne_array.notch_filter(55)

    if is_lsl:
        # Optionally, mark 'Cz' as bad and interpolate it if present
        if 'Cz' in mne_array.ch_names:
            mne_array.info['bads'] = ['Cz']     
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mne_array.interpolate_bads(reset_bads=True,  verbose = is_verbose)
            # mne_array.drop_channels('Cz')

    # Add FCz reference channel back as zeros (similar to EEGLAB's pop_reref)
    if 'FCz' not in mne_array.ch_names:
        # Create FCz channel with zeros (since data is already referenced to FCz)
        n_samples = mne_array.n_times
        fcz_data = np.zeros((1, n_samples))
        
        # Create info for FCz channel
        fcz_info = mne.create_info(['FCz'], mne_array.info['sfreq'], ch_types='eeg')
        fcz_info.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Create a Raw object for FCz
        fcz_raw = mne.io.RawArray(fcz_data, fcz_info, verbose=is_verbose)
        
        # Add FCz to the main data
        mne_array.add_channels([fcz_raw], force_update_info=True)
        
        if is_verbose:
            print("Added FCz reference channel back as zeros:", datetime.now())
            print("")

    # Set reference - Priority: FCz > A1/A2 > Common Average Reference
    if 'FCz' in mne_array.ch_names:
        # Use FCz as reference (similar to EEGLAB's pop_reref)
        mne_array.set_eeg_reference(ref_channels=['FCz'], verbose=is_verbose)
        if is_verbose:
            print("Finished re-referencing to FCz:", datetime.now())
            print("")
    else:
        # Fallback to original referencing logic
        ref_channels = ["A1", "A2"]
        available_ref = [ch for ch in ref_channels if ch in mne_array.ch_names]
        if available_ref:
            mne_array.set_eeg_reference(ref_channels=available_ref, verbose=is_verbose)
            if is_verbose:
                print(f"Finished re-referencing to {available_ref}:", datetime.now())
                print("")
        else:
            mne_array.set_eeg_reference('average', verbose=is_verbose)
            if is_verbose:
                print("Finished re-referencing to average:", datetime.now())
                print("")

    # Baseline Correction
    # mne_array.apply_function(lambda x: x - np.median(x), picks='eeg')

    # NEED TO CHECK THE VOLTAGE OF THE SIGNALS
    mne_array= 1e6 * mne_array.get_data()

    #add dimensions to match the input layer of the model
    mne_array = np.expand_dims(mne_array, axis=0)
    mne_array = np.expand_dims(mne_array, axis=0)
    
    return mne_array