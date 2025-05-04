"""
EEG DATA PREPROCESSING & LABELING (Epoched .set Files)

Description:
- Processes .set files in a specified directory matching the pattern "MIT<subject number>_INT.set".
- Assumes that these .set files already contain epoched data, likely time-locked to the MI event ('S  9').
- Identifies Motor Imagery (MI) epochs based on the 'S  9' event marker.
- Derives Rest epochs by cropping the time window immediately preceding the MI event from the MI epochs.
- Applies bandpass filtering, optional channel selection/dropping, resampling, and *correct* baseline correction separately for MI and derived Rest epochs.
- Saves the processed data in .npz format, organized by subject.

Joseph Hong (Adapted for pre-epoched .set files, Baseline Correction Fixed)
"""

# ==================================================================================================
# IMPORTS
import numpy as np
import os
import mne
import warnings

# ==================================================================================================
# DIRECTORIES & VARIABLES

# Define the directory containing the .set files
data_dir = "./data/rawdata/epoched_no_asr/"  # Update this with the correct directory path
save_dir = "./data/"
os.makedirs(save_dir, exist_ok=True)

# File naming pattern
file_pattern = "MIT{}_INT.set"
num_subjects = 33 # Set to a smaller number (e.g., 1) for testing

# Data Version
data_version = 'v5' # Updated version

# Bandpass Filter Range (Hz)
f_low, f_high = 2, 49

# Remove Specific Channels (Set one to False if using the other)
drop_chan = True
chan2drop = ['A1', 'A2', 'X5', 'TP10', 'TP9', 'FT10', 'FT9']

# Use Specific Channels Only (Set one to False if using the other)
select_chan = False
chan2use = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4'] # Example relevant channels

# Target Sampling Frequency (Hz)
new_freq = 250

# Event marker for MI (Adjust based on your data inspection)
# Common variations include 'S 9', '9', 'MI_Start', etc.
# Check epochs.event_id after loading one file if unsure.
mi_event_marker = 'S  9' # ****** IMPORTANT: VERIFY THIS MARKER NAME ******

# Time window for derived Rest epoch relative to MI onset (seconds)
mi_tmin, mi_tmax = 0.25, 1.25
rest_tmin, rest_tmax = -1.25, -0.25 # E.g., 1 second ending at MI onset

# Baseline correction periods relative to MI onset (seconds)
# Assumes original epochs cover these time ranges relative to the MI onset
baseline_mi = (-0.25, 0.0)      # Baseline for MI: -0.5 seconds to MI onset (0s)
baseline_rest = (-1.5, -1.25) # Baseline for Rest: -1.5 seconds to -1.25 seconds relative to MI onset

# ==================================================================================================
# FUNCTIONS

import warnings # Ensure warnings is imported at the top

def process_subject_data(set_file_path, subject_id, new_freq, f_low, f_high,
                        select_chan, chan2use, drop_chan, chan2drop,
                        mi_event_marker, rest_tmin, rest_tmax, baseline_mi, baseline_rest):
    """Processes epoched EEG data for a single subject from a .set file,
    deriving Rest epochs, applying filtering, resampling, and correct baseline correction."""
    print(f"Processing subject: {subject_id}")
    try:
        # Load epochs WITHOUT the baseline argument
        epochs = mne.read_epochs_eeglab(set_file_path, verbose='warning')
        print(f"  Loaded {set_file_path}. Found event IDs: {epochs.event_id}")
        print(f"  Original epoch time range: [{epochs.tmin:.3f}, {epochs.tmax:.3f}]s") # Check loaded time range

    except FileNotFoundError:
        print(f"Warning: File not found: {set_file_path}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading {set_file_path}: {e}")
        return None, None, None, None, None, None

    # --- Channel Selection / Dropping ---
    original_ch_names = epochs.ch_names
    # (Keep channel selection/dropping logic as before)
    if select_chan and chan2use:
        ch_to_use_present = [ch for ch in chan2use if ch in original_ch_names]
        if len(ch_to_use_present) < len(chan2use):
            missing = set(chan2use) - set(ch_to_use_present)
            print(f"  Warning: Requested channels not found: {missing}. Using available: {ch_to_use_present}")
        if not ch_to_use_present:
            print(f"  Error: None of the requested channels {chan2use} found in data.")
            return None, None, None, None, None, None # Adjusted return
        print(f"  Selecting channels: {ch_to_use_present}")
        epochs.pick_channels(ch_to_use_present, ordered=False)
    elif drop_chan and chan2drop:
        ch_to_drop_present = [ch for ch in chan2drop if ch in original_ch_names]
        if len(ch_to_drop_present) < len(chan2drop):
            missing = set(chan2drop) - set(ch_to_drop_present)
            print(f"  Warning: Channels requested for dropping not found: {missing}. Dropping available: {ch_to_drop_present}")
        if ch_to_drop_present:
            print(f"  Dropping channels: {ch_to_drop_present}")
            epochs.drop_channels(ch_to_drop_present)
        else:
            print(f"  No channels to drop from list {chan2drop} were found.")

    # --- Identify MI Epochs ---
    try:
        if mi_event_marker not in epochs.event_id:
            print(f"  Warning: MI event marker '{mi_event_marker}' not found in event IDs: {epochs.event_id}. Skipping.")
            return None, None, None, None, None, None
        mi_epochs_orig = epochs[mi_event_marker].copy() # Get original MI epochs
        print(f"  Found {len(mi_epochs_orig)} epochs for marker '{mi_event_marker}'. Epoch time range: [{mi_epochs_orig.tmin:.3f}, {mi_epochs_orig.tmax:.3f}]s")
        if len(mi_epochs_orig) == 0:
            print(f"  Warning: No epochs found for marker '{mi_event_marker}'. Skipping.")
            return None, None, None, None, None, None
    except KeyError:
        print(f"  Warning: Event marker '{mi_event_marker}' not found. Available: {epochs.event_id}. Skipping.")
        return None, None, None, None, None, None

    # --- Apply Common Processing (Filtering, Resampling) BEFORE Splitting Paths ---
    print(f"  Applying common processing (Filter, Resample)...")
    sfreq_original = mi_epochs_orig.info['sfreq']
    print(f"    Original sfreq: {sfreq_original} Hz")

    # Create a working copy to modify
    mi_epochs = mi_epochs_orig.copy()

    # 1. Filter
    # mi_epochs.filter(l_freq=f_low, h_freq=f_high, method='fir', fir_design='firwin', phase='zero', verbose='warning')
    # 2. Resample (if necessary)
    if sfreq_original != new_freq:
        print(f"    Resampling from {sfreq_original} Hz to {new_freq} Hz")
        mi_epochs.resample(new_freq, npad='auto', verbose='warning')
        print(f"    New epoch time range after resampling: [{mi_epochs.tmin:.3f}, {mi_epochs.tmax:.3f}]s")
    else:
        print(f"    Resampling not needed (already at {new_freq} Hz).")


    # --- MI Path: Apply MI Baseline ---
    print(f"  Processing MI path...")
    mi_processed = mi_epochs.copy() # Work on a copy for MI
    print(f"    Applying MI baseline correction: {baseline_mi}")
    # Check if baseline period is within the *current* epoch bounds
    if baseline_mi[0] < mi_processed.tmin or baseline_mi[1] > mi_processed.tmax:
        print(f"    Warning: MI baseline period {baseline_mi} is outside epoch limits [{mi_processed.tmin:.3f}, {mi_processed.tmax:.3f}]. Skipping MI baseline.")
        # Decide how to handle this - skip baseline or error out? Skipping for now.
    else:
        try:
            mi_processed.apply_baseline(baseline=baseline_mi, verbose='warning')
            # 2. Crop AFTER baseline correction
            print(f"    Cropping to MI interval: [{mi_tmin}, {mi_tmax}]")
            # Check if crop interval is valid within the current times
            if mi_tmin < mi_processed.tmin or mi_tmax > mi_processed.tmax:
                    print(f"    Warning: Requested Rest crop interval [{mi_tmin}, {mi_tmax}] is outside current epoch limits [{rest_processed.tmin:.3f}, {rest_processed.tmax:.3f}]. Cannot crop.")
                    mi_processed_derived = None
            else:
                    # Perform the crop on the baselined data
                    mi_processed_derived = mi_processed.crop(tmin=mi_tmin, tmax=mi_tmax, include_tmax=False)
                    print(f"    Derived MI epoch time range: [{mi_processed_derived.tmin:.3f}, {mi_processed_derived.tmax:.3f}]s")
        except ValueError as e:
            print(f"    Error during MI baseline application or cropping: {e}")
    
            mi_processed_derived = None # Ensure it's None if error occurs


    # 3. Get Data if cropping was successful
    if mi_processed_derived is not None and len(mi_processed_derived) > 0:
        if len(mi_processed_derived) != len(mi_epochs): # Sanity check
            print(f"    Warning: Number of derived Rest epochs ({len(mi_processed_derived)}) doesn't match original MI count ({len(mi_epochs)}).")
        mi_data_processed = mi_processed_derived.get_data(copy=True)
        mi_labels = np.ones(len(mi_data_processed))
        mi_subject_ids = np.full(len(mi_data_processed), subject_id)
    else:
        print(f"  No valid derived Rest epochs generated for subject {subject_id}.")
        # Ensure empty arrays are returned
        mi_data_processed = np.array([])
        mi_labels = np.array([])
        mi_subject_ids = np.array([])

    



    # --- Rest Path: Apply Rest Baseline THEN Crop ---
    print(f"  Processing Rest path...")
    rest_processed = mi_epochs.copy() # Work on another copy for Rest
    rest_data_processed = np.array([]) # Initialize empty
    rest_labels = np.array([])
    rest_subject_ids = np.array([])

    # 1. Apply Rest Baseline (to the uncropped, filtered, resampled data)
    print(f"    Applying Rest baseline correction: {baseline_rest}")
    # Check if baseline period is within the *current* epoch bounds
    if baseline_rest[0] < rest_processed.tmin or baseline_rest[1] > rest_processed.tmax:
        print(f"    Warning: Rest baseline period {baseline_rest} is outside epoch limits [{rest_processed.tmin:.3f}, {rest_processed.tmax:.3f}]. Cannot derive Rest epochs this way.")
        # If the baseline period isn't available, we cannot proceed with this Rest definition
        rest_epochs_derived = None
    else:
        try:
            rest_processed.apply_baseline(baseline=baseline_rest, verbose='warning')
            # 2. Crop AFTER baseline correction
            print(f"    Cropping to Rest interval: [{rest_tmin}, {rest_tmax}]")
            # Check if crop interval is valid within the current times
            if rest_tmin < rest_processed.tmin or rest_tmax > rest_processed.tmax:
                print(f"    Warning: Requested Rest crop interval [{rest_tmin}, {rest_tmax}] is outside current epoch limits [{rest_processed.tmin:.3f}, {rest_processed.tmax:.3f}]. Cannot crop.")
                rest_epochs_derived = None
            else:
                # Perform the crop on the baselined data
                rest_epochs_derived = rest_processed.crop(tmin=rest_tmin, tmax=rest_tmax, include_tmax=False)
                print(f"    Derived Rest epoch time range: [{rest_epochs_derived.tmin:.3f}, {rest_epochs_derived.tmax:.3f}]s")

        except ValueError as e:
            print(f"    Error during Rest baseline application or cropping: {e}")
            rest_epochs_derived = None # Ensure it's None if error occurs


    # 3. Get Data if cropping was successful
    if rest_epochs_derived is not None and len(rest_epochs_derived) > 0:
        if len(rest_epochs_derived) != len(mi_epochs): # Sanity check
            print(f"    Warning: Number of derived Rest epochs ({len(rest_epochs_derived)}) doesn't match original MI count ({len(mi_epochs)}).")
        rest_data_processed = rest_epochs_derived.get_data(copy=True)
        rest_labels = np.zeros(len(rest_data_processed))
        rest_subject_ids = np.full(len(rest_data_processed), subject_id)
    else:
        print(f"  No valid derived Rest epochs generated for subject {subject_id}.")
        # Ensure empty arrays are returned
        rest_data_processed = np.array([])
        rest_labels = np.array([])
        rest_subject_ids = np.array([])


    print(f"  Finished processing subject {subject_id}.")
    return mi_data_processed, mi_labels, mi_subject_ids, rest_data_processed, rest_labels, rest_subject_ids

# --- Main loop ---
# (Keep the main loop structure as it was, it should correctly handle
# the potentially empty arrays returned by the modified function)
# ... (rest of your script) ...

# ==================================================================================================
# PROCESS DATA FOR ALL SUBJECTS

all_mi_data = []
all_mi_labels = []
all_mi_subject_ids = []
all_rest_data = []
all_rest_labels = []
all_rest_subject_ids = []

for subject_num in range(1, num_subjects + 1):
    filename = file_pattern.format(subject_num)
    set_file_path = os.path.join(data_dir, filename)

    # Check if file exists before attempting to process
    if not os.path.exists(set_file_path):
        print(f"Skipping subject {subject_num}: File not found at {set_file_path}")
        continue

    mi_data, mi_labels, mi_subject_ids, \
    rest_data, rest_labels, rest_subject_ids = process_subject_data(
        set_file_path, subject_num, new_freq, f_low, f_high,
        select_chan, chan2use, drop_chan, chan2drop,
        mi_event_marker, rest_tmin, rest_tmax, baseline_mi, baseline_rest
    )

    if mi_data is not None and mi_data.size > 0 :
        # Use extend for lists if data is returned per subject,
        # or append if data is already aggregated within the function
        # Assuming process_subject_data returns numpy arrays for one subject:
        all_mi_data.append(mi_data) # Append arrays first
        all_mi_labels.append(mi_labels)
        all_mi_subject_ids.append(mi_subject_ids)
    else:
        print(f"Warning: No MI data processed for subject {subject_num}.")


    # Check rest_data specifically for None or empty array
    if rest_data is not None and rest_data.size > 0:
        all_rest_data.append(rest_data) # Append arrays first
        all_rest_labels.append(rest_labels)
        all_rest_subject_ids.append(rest_subject_ids)
    else:
        # This is expected if rest derivation failed or wasn't possible
        print(f"Note: No derived Rest data generated for subject {subject_num}.")

## --- Sanity check: Ensure all MI data have the same shape ---
for idx, data in enumerate(all_mi_data):
    print(f"MI Subject index {idx}: data shape {data.shape}")
for idx, data in enumerate(all_rest_data):
    print(f"Rest Subject index {idx}: data shape {data.shape}")

# Combine MI and Rest data (if both exist)
if all_mi_data and all_rest_data:
    X_mi_final = np.vstack(all_mi_data)
    y_mi_final = np.concatenate(all_mi_labels)
    subject_mi_final = np.concatenate(all_mi_subject_ids)

    X_rest_final = np.vstack(all_rest_data)
    y_rest_final = np.concatenate(all_rest_labels)
    subject_rest_final = np.concatenate(all_rest_subject_ids)

    # Combine MI and Rest together
    X_final = np.vstack([X_mi_final, X_rest_final])
    y_final = np.concatenate([y_mi_final, y_rest_final])
    subject_final = np.concatenate([subject_mi_final, subject_rest_final])

    print("Final Combined Data Shape:", X_final.shape)
    print("Final Combined Labels Shape:", y_final.shape)
    print("Final Combined Subject IDs Shape:", subject_final.shape)

    # Save everything together into one file
    save_filename_combined = f"bci_subject_data_{data_version}.npz"
    np.savez(os.path.join(save_dir, save_filename_combined), X=X_final, y=y_final, subject_ids=subject_final)
    print(f"Combined MI+Rest epoched data saved to {os.path.join(save_dir, save_filename_combined)}")

else:
    print("Insufficient data collected. Either MI or Rest data is missing.")
