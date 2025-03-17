"""
EEG DATA PREPROCESSING & LABELING (Variable Subjects, 2 Columns)

Description:
- Processes .mat files where each row represents a different subject.
- Each subject has Rest and MI trials in separate columns.
- Applies bandpass filtering, referencing, and channel selection.
- Saves the processed data in .npz format for later use in classification.

Joseph Hong

"""

# ==================================================================================================
# IMPORTS
import scipy.io
import numpy as np
import os
import mne

# ==================================================================================================
# DIRECTORIES & VARIABLES

# Define Data Directory
ref_dir = "./reference/"
epoch_folder = "./epoched/"  # Update this with the correct folder path
save_dir = "./data/"
mat_file_path = f"{epoch_folder}STvsRest.mat"  # Update with actual path

# Data Version
data_version = 'v4'
save_filename = f"mit_subject_data_{data_version}.npz"  # Specify your desired output file

# Bandpass Filter Range (Hz)
f_low, f_high = 2, 49  

# Remove Specific Channels, Retain Others
drop_chan = False
# chan2drop = ["T7", "T8", 'FT7', 'FT8']
chan2drop = ['A1', 'A2', 'X5']

# Use Specific Channels Only
select_chan = False
chan2use = ['F3','F4','C3','Cz','C4','P3','P4']

# 60 Channels
channels_data = scipy.io.loadmat(os.path.join(ref_dir,"channels_60.mat"))
channel_names = [ch[0] for ch in channels_data['channels'].flatten()]  # Extract names properly

# Define Sampling Frequency (adjust if needed)
sfreq = 1000
new_freq = 200

original_timepoints = 2000
original_freq = sfreq  # Hz
target_freq = new_freq  # Hz

def compute_new_timepoints(original_timepoints, original_freq, target_freq):
    return int(original_timepoints * (target_freq / original_freq))

new_timepoints = compute_new_timepoints(original_timepoints, original_freq, target_freq)
print("New Timepoints after Downsampling:", new_timepoints)

# ==================================================================================================
# LOAD & PROCESS DATA

# Load .mat file
data = scipy.io.loadmat(mat_file_path)

print(data.keys())

# Extract EEG data (assuming a known variable name in the .mat file)
eeg_data = data['data']  # Replace with actual variable name

# Determine number of subjects dynamically
num_subjects = eeg_data.shape[0]  # Get the number of rows

# Ensure correct structure: (num_subjects, 2 conditions)
assert eeg_data.shape[1] == 2, "Unexpected data structure. Expected (num_subjects, 2)."

# Initialize lists
X_list, y_list, subject_list = [], [], []

# Iterate through all subjects dynamically
for subject_id in range(num_subjects):
    rest_data = eeg_data[subject_id, 0]  # (channels, timepoints, trials)
    mi_data = eeg_data[subject_id, 1]  # (channels, timepoints, trials)

    # Ensure correct shape
    assert rest_data.ndim == 3, f"Rest data for subject {subject_id} has incorrect dimensions"
    assert mi_data.ndim == 3, f"MI data for subject {subject_id} has incorrect dimensions"

    # Get number of trials
    rest_trials = rest_data.shape[2]
    mi_trials = mi_data.shape[2]

    print(rest_trials)
    print(mi_trials)

    # Process each trial
    for trial_idx in range(rest_trials):
        trial = rest_data[:, :, trial_idx]  # (channels, timepoints)

        # Convert to MNE Raw object
        info = mne.create_info(ch_names=[f"Ch{i}" for i in range(trial.shape[0])], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(trial, info)

        # Select relevant channels
        if select_chan:
            raw = raw.pick_channels(chan2use)

        # Apply bandpass filter
        # raw.filter(l_freq=2, h_freq=49, method='fir', fir_design='firwin', phase='zero')

        # Downsample (if needed)
        if(sfreq != new_freq):
            raw.resample(new_freq)

        # Store processed rest data
        X_list.append(raw.get_data())
        y_list.append(0)  # Rest Label
        subject_list.append(subject_id)

    
    # MI Trials
    for trial_idx in range(mi_trials):
        trial = mi_data[:, :, trial_idx]  # (channels, timepoints)

        # Convert to MNE Raw object
        info = mne.create_info(ch_names=[f"Ch{i}" for i in range(trial.shape[0])], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(trial, info)

        # Select relevant channels
        if select_chan:
            raw = raw.pick_channels(chan2use)

        # Apply bandpass filter
        # raw.filter(l_freq=2, h_freq=49, method='fir', fir_design='firwin', phase='zero')

        # Downsample (if needed)
        if(sfreq != new_freq):
            raw.resample(new_freq)

        # Store processed MI data
        X_list.append(raw.get_data())
        y_list.append(1)  # MI Label
        subject_list.append(subject_id)

fixed_length = new_timepoints # Set timepoint limit based on Rest dataset

for i in range(len(X_list)):
    num_timepoints = X_list[i].shape[1]

    # Truncate MI trials if they exceed the fixed length
    if num_timepoints > fixed_length:
        X_list[i] = X_list[i][:, :fixed_length]  # Keep only the first 2000 timepoints


# Convert to NumPy arrays
X_final = np.array(X_list)  # (total_trials, channels, timepoints)
y_final = np.array(y_list)  # (total_trials,)
subject_final = np.array(subject_list)  # (total_trials,)

# Save to .npz file
np.savez(os.path.join(save_dir, save_filename), X=X_final, y=y_final, subject_ids=subject_final)

# Print final shape info
print("Final Data Shape:", X_final.shape)  # (trials, channels, timepoints)
print("Final Labels Shape:", y_final.shape)  # (trials,)
print("Final Subject IDs Shape:", subject_final.shape)  # (trials,)
print(f"Data saved to {os.path.join(save_dir, save_filename)}")
