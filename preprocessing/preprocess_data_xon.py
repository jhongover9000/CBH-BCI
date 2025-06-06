import mne
import numpy as np
import os
from glob import glob

# -------------------------------
# PARAMETERS & FILE PATHS
# -------------------------------
# Directory containing .fif files
input_dir = './lsl_data/combined/'
# Glob pattern to get all .fif files
fif_files = glob(os.path.join(input_dir, '*.fif'))
if not fif_files:
    raise ValueError("No .fif files found in the directory: " + input_dir)

save_dir = "./data/"
os.makedirs(save_dir, exist_ok=True)

marker_names = ["imagery", "execution"]  # List of marker names to process

# Data Version
data_version = 'v6'
filename = f"xon_subject_data_{data_version}.npz"  # Specify your desired output file

output_filename = os.path.join(save_dir, filename)

target_sfreq = 200  # Desired sampling frequency (Hz)
l_freq, h_freq = 2, 49  # Bandpass filter limits (Hz)

# Channel configuration
drop_chan = True
chan2drop = ['A1', 'A2', 'X5', 'BIP']
select_chan = False
chan2use = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4']

# Lists to accumulate processed data
X_list = []
y_list = []
subject_list = []

montage = mne.channels.make_standard_montage('standard_1020')

# -------------------------------
# PROCESS EACH .fif FILE
# -------------------------------
for fif_file in fif_files:
    print("Processing file:", fif_file)

    # Load the raw data from the .fif file
    try:
        raw = mne.io.read_raw_fif(fif_file, preload=True)
    except Exception as e:
        print(f"Error loading {fif_file}: {e}")
        continue

    # raw._data /= 1e6

    print(raw._data)

    # Downsample to target frequency
    raw.resample(target_sfreq)
    # After resampling
    print("Data range after resampling:", raw._data.min(), raw._data.max())

    raw.notch_filter(50)

    # Apply bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin')

    # After filtering
    print("Data range after filtering:", raw._data.min(), raw._data.max())

    # Drop specified channels if they exist
    available_drop = [ch for ch in chan2drop if ch in raw.ch_names]
    if drop_chan and available_drop:
        raw.drop_channels(available_drop)
        print("Channels after dropping:", raw.info['ch_names'])

    # Retain only channels that are part of the standard montage
    valid_ch_names = montage.ch_names
    common_ch_names = list(set(raw.ch_names) & set(valid_ch_names))
    raw.set_montage(montage)

    # Optionally, mark 'Cz' as bad and interpolate it if present
    if 'Cz' in raw.ch_names:
        raw.info['bads'] = ['Cz']
        # raw.interpolate_bads(reset_bads=True)
        raw.drop_channels('Cz')

    # Set common average reference if reference channels exist
    ref_channels = ["A1", "A2"]
    available_ref = [ch for ch in ref_channels if ch in raw.ch_names]
    if available_ref:
        raw.set_eeg_reference(ref_channels=available_ref)
    else:
        raw.set_eeg_reference('average')

    # After referencing
    print("Data range after referencing:", raw._data.min(), raw._data.max())

    # Optionally, select a specific subset of channels
    if select_chan:
        raw.pick_channels(chan2use)
        print("Selected channels:", raw.info['ch_names'])

    print(raw._data)

    # -------------------------------
    # EXTRACT EVENTS AND EPOCH DATA FOR MULTIPLE MARKERS
    # -------------------------------
    # Ensure the raw data contains annotations
    if raw.annotations is None or len(raw.annotations) == 0:
        print(f"No annotations found in {fif_file}. Skipping file.")
        continue

    events, event_id = mne.events_from_annotations(raw)
    print("Found event IDs:", event_id)

    for marker in marker_names:
        if marker in event_id:
            marker_event_code = event_id[marker]
            marker_events = events[events[:, 2] == marker_event_code]

            if marker_events.shape[0] == 0:
                print(f"No {marker} events found in {fif_file}.")
                continue

            print(f"Creating epochs for '{marker}' events.")
            epochs = mne.Epochs(raw, marker_events, event_id={marker: marker_event_code},
                                tmin=-2, tmax=2, baseline=None, preload=True)
            print(f"Number of '{marker}' epochs in {fif_file}: {len(epochs)}")

            # -------------------------------
            # SPLIT EPOCHS INTO 1-SECOND SEGMENTS AND LABEL
            # -------------------------------
            n_samples = target_sfreq

            for i, epoch in enumerate(epochs.get_data()):
                if epoch.shape[1] < 4 * n_samples:  # Epochs are 4 seconds long
                    print(f"Epoch {i} ({marker}) in {fif_file} does not have enough samples. Skipping epoch.")
                    continue

                # Pre-event: -1 to 0 sec relative to the event
                pre_event = epoch[:, n_samples:2 * n_samples]
                X_list.append(pre_event)
                y_list.append(0)  # Label 0 for all pre-event segments
                subject_list.append(os.path.basename(fif_file))

                # Post-event: 0 to +1 sec relative to the event
                post_event = epoch[:, 2 * n_samples:3 * n_samples]
                X_list.append(post_event)
                y_list.append(1)  # Label 1 for all post-event segments
                subject_list.append(os.path.basename(fif_file))
        else:
            print(f"No '{marker}' marker found in {fif_file}.")

# -------------------------------
# FINALIZE AND SAVE PROCESSED DATA
# -------------------------------
X_final = np.array(X_list)  # Shape: (total_segments, channels, 200)
y_final = np.array(y_list)  # Shape: (total_segments,)
subject_final = np.array(subject_list)  # Shape: (total_segments,)

print("Final Combined Data Shape:", X_final.shape)
print("Final Labels Shape:", y_final.shape)
print("Final Subject Labels Shape:", subject_final.shape)

# Save the processed data to an .npz file
np.savez(output_filename, X=X_final, y=y_final, subject_ids=subject_final)
print(f"Data saved to {output_filename}.")