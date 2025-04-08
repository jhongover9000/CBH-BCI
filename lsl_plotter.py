"""
LSL PLOTTER

Description:
- Preprocesses .fif raw files recorded using the LSL Recorder script
- Reads all files in a given DIRECTORY and searches using an specific event marker
- Plots topographies of alpha and beta band frequency powers over time
- DOES NOT SAVE THE DATA, REFER TO LSL EPOCHER FOR CONVERTING AND SAVING EPOCHS

Joseph Hong

"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# ----- 1. Load and Combine All FIF Files -----
data_dir = "./lsl_data/combined/"
file_ext = '.fif'
raw_list = []
event = "imagery"
montage = mne.channels.make_standard_montage('standard_1020')

for fname in os.listdir(data_dir):
    if fname.endswith(file_ext):
        file_path = os.path.join(data_dir, fname)
        print(f"Loading {fname}")
        temp = mne.io.read_raw_fif(file_path, preload=True)
        
        # Retain only channels that are part of the standard montage
        valid_ch_names = montage.ch_names
        common_ch_names = list(set(temp.ch_names) & set(valid_ch_names))
        temp.pick_channels(common_ch_names)
        temp.set_montage(montage)
        
        # Optionally, mark 'Cz' as bad and interpolate it if present
        if 'Cz' in temp.ch_names:
            temp.info['bads'] = ['Cz']
            temp.interpolate_bads(reset_bads=True)
            # temp.drop_channels('Cz')
        
        # Apply filtering and set common average reference
        temp.filter(1, 40.0, fir_design='firwin')
        temp.set_eeg_reference('average', projection=False)
        
        # Check if the file contains "event" events in the annotations
        annotations = temp.annotations
        event_onsets = [onset for onset, desc in zip(annotations.onset, annotations.description)
                          if event in desc.lower()]
        if not event_onsets:
            print(f"No {event} events found in {fname}. Skipping this file.")
            continue
        
        raw_list.append(temp)

if not raw_list:
    raise ValueError("No FIF files with event events were found in the directory.")

# Combine all the Raw objects into one continuous Raw object
raw_combined = raw_list[0]
if len(raw_list) > 1:
    for raw_obj in raw_list[1:]:
        raw_combined.append(raw_obj)
raw = raw_combined

# ----- 2. Extract event Events -----
annotations = raw.annotations
event_indices = [i for i, desc in enumerate(annotations.description) if event in desc.lower()]
if not event_indices:
    raise ValueError(f"No '{event}' events found in the combined raw data.")
sfreq = raw.info['sfreq']

# ----- 3. Compute Morlet Wavelet Power for Alpha & Beta Bands -----
n_cycles = 5
frequencies_alpha = np.arange(8, 14, 1)     # Alpha: 8-13 Hz
frequencies_beta = np.arange(13, 31, 2)       # Beta: 13-31 Hz

# Get raw data (channels x time)
data = raw.get_data()

# Compute power using Morlet wavelets (output shape: channels x n_freqs x time)
power_alpha = mne.time_frequency.tfr_array_morlet(data[np.newaxis, :],
                                                  sfreq=sfreq,
                                                  freqs=frequencies_alpha,
                                                  n_cycles=n_cycles,
                                                  output='power')[0]
power_beta = mne.time_frequency.tfr_array_morlet(data[np.newaxis, :],
                                                 sfreq=sfreq,
                                                 freqs=frequencies_beta,
                                                 n_cycles=n_cycles,
                                                 output='power')[0]

# Average power over the frequency range and convert to decibels (dB)
mean_power_alpha = power_alpha.mean(axis=1)  # shape: (n_channels, n_times)
mean_power_beta = power_beta.mean(axis=1)
mean_power_alpha_db = 10 * np.log10(mean_power_alpha + 1e-10)
mean_power_beta_db = 10 * np.log10(mean_power_beta + 1e-10)

# ----- 4. Baseline Correction & Grand-Average Topographies (Post-Event: 0–1 s) -----
alpha_event_powers = []
beta_event_powers = []

for idx in event_indices:
    onset = annotations.onset[idx]
    onset_sample = int(onset * sfreq)
    
    # Define baseline window: from -0.5 to 0 s relative to event onset
    bl_start = int((onset - 0.5) * sfreq)
    bl_end = onset_sample
    # Define post-event window: from 0 to 1 s after event onset
    post_start = onset_sample
    post_end = int((onset + 1.0) * sfreq)
    
    if bl_start < 0 or post_end > mean_power_alpha_db.shape[1]:
        continue
    
    # Compute the average baseline power (in dB) per channel
    bl_alpha = mean_power_alpha_db[:, bl_start:bl_end].mean(axis=1)
    bl_beta = mean_power_beta_db[:, bl_start:bl_end].mean(axis=1)
    
    # Compute the average post-event power (in dB) per channel
    post_alpha = mean_power_alpha_db[:, post_start:post_end].mean(axis=1)
    post_beta = mean_power_beta_db[:, post_start:post_end].mean(axis=1)
    
    # Baseline-corrected power (dB)
    alpha_event_powers.append(post_alpha - bl_alpha)
    beta_event_powers.append(post_beta - bl_beta)

# Grand-average across event events (post-event)
alpha_avg = np.mean(alpha_event_powers, axis=0)
beta_avg = np.mean(beta_event_powers, axis=0)

# ----- 5. Plot Grand-Average Topographies (Post-Event) -----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
mne.viz.plot_topomap(alpha_avg, raw.info, axes=axes[0], show=False,
                     cmap='viridis', contours=6)
axes[0].set_title("Alpha Band (dB, BL-corrected)")
mne.viz.plot_topomap(beta_avg, raw.info, axes=axes[1], show=False,
                     cmap='viridis', contours=6)
axes[1].set_title("Beta Band (dB, BL-corrected)")
plt.suptitle("Grand Average Topographies (0–1 s post-event)")
plt.show()

# ----- 6. Compute Baseline Topography -----
# For each event event, compute the baseline (–0.5 to 0 s) power per channel (in dB)
baseline_topo_alpha_events = []
baseline_topo_beta_events = []
for idx in event_indices:
    onset = annotations.onset[idx]
    onset_sample = int(onset * sfreq)
    bl_start = int((onset - 0.5) * sfreq)
    bl_end = onset_sample
    if bl_start < 0 or onset_sample > mean_power_alpha_db.shape[1]:
        continue
    baseline_alpha = mean_power_alpha_db[:, bl_start:bl_end].mean(axis=1)
    baseline_beta = mean_power_beta_db[:, bl_start:bl_end].mean(axis=1)
    baseline_topo_alpha_events.append(baseline_alpha)
    baseline_topo_beta_events.append(baseline_beta)

baseline_topo_alpha_avg = np.mean(baseline_topo_alpha_events, axis=0)
baseline_topo_beta_avg = np.mean(baseline_topo_beta_events, axis=0)

# ----- 7. Time-Binned Topographies for Post-Event (0–1 s) -----
# Define 100 ms bins within the post-event window (0–1 s)
time_bin_sec = 0.1
bin_samples = int(time_bin_sec * sfreq)
n_bins = int(1.0 / time_bin_sec)
time_bins = [(i * bin_samples, (i+1) * bin_samples) for i in range(n_bins)]

alpha_binned_topos = []
beta_binned_topos = []

# For each time bin, compute the baseline-corrected power for every event and then average
for bin_start, bin_end in time_bins:
    alpha_bin_events = []
    beta_bin_events = []
    for idx in event_indices:
        onset = annotations.onset[idx]
        onset_sample = int(onset * sfreq)
        bl_start = int((onset - 0.5) * sfreq)
        bl_end = onset_sample
        post_start = onset_sample + bin_start
        post_end = onset_sample + bin_end
        
        if bl_start < 0 or post_end > mean_power_alpha_db.shape[1]:
            continue
        
        bl_alpha = mean_power_alpha_db[:, bl_start:bl_end].mean(axis=1)
        bl_beta = mean_power_beta_db[:, bl_start:bl_end].mean(axis=1)
        post_alpha = mean_power_alpha_db[:, post_start:post_end].mean(axis=1)
        post_beta = mean_power_beta_db[:, post_start:post_end].mean(axis=1)
        
        alpha_bin_events.append(post_alpha - bl_alpha)
        beta_bin_events.append(post_beta - bl_beta)
    
    if alpha_bin_events:
        alpha_binned_topos.append(np.mean(alpha_bin_events, axis=0))
        beta_binned_topos.append(np.mean(beta_bin_events, axis=0))
    else:
        alpha_binned_topos.append(np.full(len(raw.ch_names), np.nan))
        beta_binned_topos.append(np.full(len(raw.ch_names), np.nan))

# Combine baseline topography with post-event binned topographies.
# The first topoplot will represent the entire baseline period.
all_alpha_topos = [baseline_topo_alpha_avg] + alpha_binned_topos
all_beta_topos = [baseline_topo_beta_avg] + beta_binned_topos

# ----- 8. Plot Time-Binned Topographies with Baseline as First Plot -----
# Total number of plots = 1 (baseline) + n_bins (post-event)
total_bins = n_bins + 1

# Plot Alpha band topographies
fig_alpha, axes_alpha = plt.subplots(1, total_bins, figsize=(3 * total_bins, 4))
axes_alpha[0].set_title("Baseline\n(-0.5 to 0 s)")
mne.viz.plot_topomap(all_alpha_topos[0], raw.info, axes=axes_alpha[0],
                     show=False, cmap='viridis', contours=6)
for i in range(1, total_bins):
    mne.viz.plot_topomap(all_alpha_topos[i], raw.info, axes=axes_alpha[i],
                         show=False, cmap='viridis', contours=6)
    start_t = (i-1) * time_bin_sec
    end_t = i * time_bin_sec
    axes_alpha[i].set_title(f"{start_t:.1f}-{end_t:.1f} s")
fig_alpha.suptitle("Alpha Band Topographies", fontsize=16)
plt.show()

# Plot Beta band topographies
fig_beta, axes_beta = plt.subplots(1, total_bins, figsize=(3 * total_bins, 4))
axes_beta[0].set_title("Baseline\n(-0.5 to 0 s)")
mne.viz.plot_topomap(all_beta_topos[0], raw.info, axes=axes_beta[0],
                     show=False, cmap='viridis', contours=6)
for i in range(1, total_bins):
    mne.viz.plot_topomap(all_beta_topos[i], raw.info, axes=axes_beta[i],
                         show=False, cmap='viridis', contours=6)
    start_t = (i-1) * time_bin_sec
    end_t = i * time_bin_sec
    axes_beta[i].set_title(f"{start_t:.1f}-{end_t:.1f} s")
fig_beta.suptitle("Beta Band Topographies", fontsize=16)
plt.show()
