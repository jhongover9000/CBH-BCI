# Preprocess Script
# Desc: Preprocesses EEG data per trial using MNE (bandpass filtering & CAR), then saves it.
#
# Joseph Hong
# ==================================================================================================
import scipy.io
import numpy as np
import os
import re
from glob import glob
import mne

# Define Data Directory
ref_dir = "./reference/"
epoch_folder = "./epoched/"  # Update this with the correct folder path
save_dir = "./data/"

# Data Version
data_version = 'v4'
filename = f"subject_data_{data_version}.npz"  # Specify your desired output file

# BPF Variables
f_list = [0.1, 49]

drop_chan = False
chan2drop = ["T7", "T8", 'FT7', 'FT8']
select_chan = True
chan2use = ['F3','F4','C3','Cz','C4','P3','P4']

# Get List of All .mat Files
mat_files = glob(os.path.join(epoch_folder, "*.mat"))

# Initialize Lists to Store Data, Labels, and Subject IDs
X_list, y_list, subject_list = [], [], []

# Load Channel Names
channels_data = scipy.io.loadmat(os.path.join(ref_dir,"channels.mat"))
channel_names = [ch[0] for ch in channels_data['channels'].flatten()]  # Extract names properly

# Define Sampling Frequency (Adjust if known)
sfreq = 100  # Update if dataset uses a different rate

# Iterate Over Each Subject's File
for file in mat_files:
    # Extract Subject ID from Filename (e.g., "SubA-160405_index.mat" → "SubA")
    match = re.match(r"(Sub[A-Z]+)-", os.path.basename(file))
    if not match:
        print(f"Skipping {file}, could not extract subject ID.")
        continue
    subject_id = match.group(1)  # "SubA"

    print(f"Processing: {file} (Subject: {subject_id})")
    
    # Load EEG Data
    eeg_data = scipy.io.loadmat(file)
    
    if 'X_bs' not in eeg_data:
        print(f"Skipping {file}, 'X_bs' key not found.")
        continue

    X = eeg_data['X_bs']  # Shape: (trials, channels, timepoints)

    # Iterate over each trial
    for trial_idx in range(X.shape[0]):
        trial_data = X[trial_idx, :, :]  # Shape: (channels, timepoints)

        # Convert trial to MNE RawArray
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(trial_data, info)

        # Drop additional channels to fit BFN
        if(drop_chan):
            raw = raw.drop_channels(chan2drop)
            print("Channels after dropping:", raw.info['ch_names'])

        # Pick specific channels: C3, O1, O2 (if needed)
        if(select_chan):
            raw = raw.pick_channels(chan2use)
            print("Channels after selecting:", raw.info['ch_names'])

        # Apply Bandpass Filter (7–30 Hz)
        raw.filter(l_freq=f_list[0], h_freq=f_list[1], fir_design='firwin')

        # Apply Common Average Referencing (CAR)
        raw.set_eeg_reference(ref_channels="average")

        # Get preprocessed data as NumPy array
        processed_data = raw.get_data()  # Shape: (channels, timepoints)

        # Split into Rest (first 200 timepoints) & Motor Imagery (last 200 timepoints)
        X_rest = processed_data[:, :200]  # First 200 timepoints (Rest)
        X_mi = processed_data[:, 200:]    # Last 200 timepoints (Motor Imagery)

        # Create Labels
        y_rest = 0  # Rest = 0
        y_mi = 1    # Motor Imagery = 1

        # Store Processed Trials
        X_list.append(X_rest)
        y_list.append(y_rest)
        subject_list.append(subject_id)

        X_list.append(X_mi)
        y_list.append(y_mi)
        subject_list.append(subject_id)

# Convert to NumPy Arrays
X_final = np.array(X_list)  # Shape: (total_trials, channels, 200)
y_final = np.array(y_list)  # Shape: (total_trials,)
subject_final = np.array(subject_list)  # Shape: (total_trials,)

# Print Shape
print("Final Combined Data Shape:", X_final.shape)  # (total_trials, channels, 200)
print("Final Labels Shape:", y_final.shape)  # (total_trials,)
print("Final Subject Labels Shape:", subject_final.shape)  # (total_trials,)

# Save Data to .npz File
np.savez(os.path.join(save_dir, filename), X=X_final, y=y_final, subject_ids=subject_final)

print(f"Data saved to {os.path.join(save_dir, filename)}.")