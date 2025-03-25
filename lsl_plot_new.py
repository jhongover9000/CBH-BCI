import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from autoreject import AutoReject

# ===============================================================
# 1. Load and Combine Raw Data from Directory
# ===============================================================
data_dir = "./lsl_data/mi/"
file_ext = '.fif'
raw_list = []

for fname in os.listdir(data_dir):
    if fname.endswith(file_ext):
        file_path = os.path.join(data_dir, fname)
        print(f"Loading {fname}")
        temp = mne.io.read_raw_fif(file_path, preload=True)

        # Set montage and pick only valid channels
        montage = mne.channels.make_standard_montage('standard_1020')
        valid_ch_names = montage.ch_names
        common_ch_names = list(set(temp.ch_names) & set(valid_ch_names))
        temp.pick_channels(common_ch_names)
        temp.set_montage(montage)

        if 'Cz' in temp.ch_names:
            temp.info['bads'] = ['Cz']
            temp.interpolate_bads(reset_bads=True)

        # Preprocessing: bandpass (0.5–40 Hz) and average reference
        temp.filter(1, 40.0, fir_design='firwin')
        temp.set_eeg_reference('average', projection=False)

        # Check for "Imagery" events in annotations
        annotations = temp.annotations
        imagery_onsets = [onset for onset, desc in zip(annotations.onset, annotations.description)
                          if 'imagery' in desc.lower()]
        if not imagery_onsets:
            print(f"No imagery events in {fname}, skipping.")
            continue

        raw_list.append(temp)

if not raw_list:
    raise ValueError("No files with 'Imagery' events were found.")

# Combine raw files (if more than one)
raw = raw_list[0]
if len(raw_list) > 1:
    raw.append(raw_list[1:])

raw.plot(title="Combined Raw Data", show=True)

# ===============================================================
# 2. Extract 'Imagery' Events and Create Epochs
# ===============================================================
# Convert annotations to events using a mapping (here all 'imagery' events get code 1)
events, event_id = mne.events_from_annotations(raw, event_id={'Imagery': 1})
print(f"Found {len(events)} imagery events.")

# Define epoch time window (here –5 to 5 s relative to event onset)
tmin, tmax = -5.0, 5.0
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    preload=True, baseline=(-0.5,0))
print(f"Created {len(epochs)} epochs.")
# epochs.plot(title="Epochs from Imagery Events", show=True)

# ===============================================================
# 3. Clean Epochs using AutoReject
# ===============================================================
# ar = AutoReject()
# epochs_clean = ar.fit_transform(epochs)
# epochs_clean.plot(title="Cleaned Epochs", show=True)

ica = mne.preprocessing.ICA(n_components=7, max_iter="auto", random_state=97)
ica.fit(epochs)

epochs_clean = epochs

# ===============================================================
# 4. Time–Frequency Analysis on Cleaned Epochs
# ===============================================================
sfreq = epochs_clean.info['sfreq']
n_cycles = 5

# Define frequency ranges
frequencies_alpha = np.arange(8, 14, 1)
frequencies_beta  = np.arange(13, 31, 2)

# Compute TFR using Morlet wavelets (keep individual epochs for later binning)
power_alpha = mne.time_frequency.tfr_morlet(epochs_clean, freqs=frequencies_alpha,
                                            n_cycles=n_cycles, return_itc=False, average=False)
power_beta  = mne.time_frequency.tfr_morlet(epochs_clean, freqs=frequencies_beta,
                                            n_cycles=n_cycles, return_itc=False, average=False)

# ===============================================================
# 5. Baseline Correction (Using a Pre-Event Window)
# ===============================================================
# Define baseline period (e.g., –2 to –1 s relative to event onset)
baseline = (-0.5, 0.0)

power_alpha.apply_baseline(baseline=baseline, mode='logratio')
power_beta.apply_baseline(baseline=baseline, mode='logratio')

# Grand average across epochs (for topographic plotting)
power_alpha_avg = power_alpha.average()
power_beta_avg  = power_beta.average()

mean_power_alpha_db = 10 * np.log10(power_alpha_avg.data + 1e-10)
mean_power_beta_db = 10 * np.log10(power_beta_avg.data + 1e-10)

# 6. Topoplot
# ===============================================================

topo_data_alpha = mean_power_alpha_db.mean(axis=(1, 2))
topo_data_beta  = mean_power_beta_db.mean(axis=(1, 2))


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
mne.viz.plot_topomap(topo_data_alpha, raw.info, axes=axes[0], show=False, cmap='viridis', contours=6)
axes[0].set_title("Alpha Avg (dB, BL-removed)")

mne.viz.plot_topomap(topo_data_beta, raw.info, axes=axes[1], show=False, cmap='viridis', contours=6)
axes[1].set_title("Beta Avg (dB, BL-removed)")
plt.suptitle("Grand Avg Topos (0–1s post-imagery)", fontsize=14)
plt.show()

# ===============================================================
# 7. Compute and Plot
# ===============================================================

# Define time bins (100 ms each) from 0 to 1 second post-imagery
time_bin_ms = 100
bin_samples = int((time_bin_ms / 1000) * sfreq)
post_start_offset = 0  # relative to imagery onset
post_end_offset = int(1.0 * sfreq)
time_bins = list(range(post_start_offset, post_end_offset, bin_samples))

# Use the grand average TFRs to extract power for C3
times = power_alpha_avg.times  # common time vector
bin_width = 0.1  # seconds (100 ms bins)
time_bins = np.arange(0, 1.0, bin_width)  # from 0 to 1 s post-event
c3_alpha_power = []
c3_beta_power  = []

try:
    c3_idx = epochs_clean.ch_names.index('C3')
except ValueError:
    raise ValueError("Channel C3 not found in data.")

for t_start in time_bins:
    # Create a mask for the time bin
    t_mask = (times >= t_start) & (times < t_start + bin_width)
    # For alpha: average over frequencies and time for channel C3
    alpha_val = power_alpha_avg.data[c3_idx, :, t_mask].mean()
    beta_val  = power_beta_avg.data[c3_idx, :, t_mask].mean()
    c3_alpha_power.append(alpha_val)
    c3_beta_power.append(beta_val)

if c3_idx is not None:

    for bin_start in time_bins:
        bin_end = bin_start + bin_samples

        alpha_bin_vals = []
        beta_bin_vals = []
        c3_alpha_bin_vals = []
        c3_beta_bin_vals = []

        for epoch in epochs_clean:
            onset_sample = int(annotations.onset[idx] * sfreq)

            # Baseline window (-2 to -1.8s)
            bl_start = int((annotations.onset[idx] - 2.0) * sfreq)
            bl_end = int((annotations.onset[idx] - 1.8) * sfreq)

            # Validate baseline and post windows
            if bl_start < 0 or (onset_sample + bin_end) > mean_power_alpha.shape[1]:
                continue

            # Compute baseline per event (in dB)
            bl_alpha = mean_power_alpha[:, bl_start:bl_end].mean(axis=1)
            bl_beta = mean_power_beta[:, bl_start:bl_end].mean(axis=1)
            bl_alpha_db = 10 * np.log10(bl_alpha + 1e-10)
            bl_beta_db = 10 * np.log10(bl_beta + 1e-10)

            # Extract post-onset bin, convert to dB, subtract baseline
            alpha_bin = mean_power_alpha[:, onset_sample + bin_start:onset_sample + bin_end]
            beta_bin = mean_power_beta[:, onset_sample + bin_start:onset_sample + bin_end]

            alpha_bin_db = 10 * np.log10(alpha_bin.mean(axis=1) + 1e-10) - bl_alpha_db
            beta_bin_db = 10 * np.log10(beta_bin.mean(axis=1) + 1e-10) - bl_beta_db

            alpha_bin_vals.append(alpha_bin_db)
            beta_bin_vals.append(beta_bin_db)
            c3_alpha_bin_vals.append(alpha_bin_db[c3_idx])
            c3_beta_bin_vals.append(beta_bin_db[c3_idx])

        # Compute average across events for this bin
        alpha_binned_topos.append(np.mean(alpha_bin_vals, axis=0))
        beta_binned_topos.append(np.mean(beta_bin_vals, axis=0))
        c3_alpha_power.append(np.mean(c3_alpha_bin_vals))
        c3_beta_power.append(np.mean(c3_beta_bin_vals))

    # Plot topographies over time (binned)
    fig_alpha_bin, axes_alpha_bin = plt.subplots(1, len(alpha_binned_topos), figsize=(4 * len(alpha_binned_topos), 4))
    fig_beta_bin, axes_beta_bin = plt.subplots(1, len(beta_binned_topos), figsize=(4 * len(beta_binned_topos), 4))

    for i, (alpha_topo, beta_topo) in enumerate(zip(alpha_binned_topos, beta_binned_topos)):
        mne.viz.plot_topomap(alpha_topo, raw.info, axes=axes_alpha_bin[i], show=False, cmap='viridis', contours=6)
        axes_alpha_bin[i].set_title(f'{times_sec[i]:.1f}s')

        mne.viz.plot_topomap(beta_topo, raw.info, axes=axes_beta_bin[i], show=False, cmap='viridis', contours=6)
        axes_beta_bin[i].set_title(f'{times_sec[i]:.1f}s')

    fig_alpha_bin.suptitle('Alpha Band Topography (100 ms bins)', fontsize=16)
    fig_beta_bin.suptitle('Beta Band Topography (100 ms bins)', fontsize=16)
    plt.show()


plt.figure(figsize=(8, 4))
plt.plot(time_bins + bin_width/2, c3_alpha_power, label='Alpha (C3)', marker='o')
plt.plot(time_bins + bin_width/2, c3_beta_power, label='Beta (C3)', marker='s')
plt.xlabel('Time (s) post-event')
plt.ylabel('Power (logratio baseline corrected)')
plt.title('C3 Channel Power Over Time')
plt.legend()
plt.grid(True)
plt.show()

if c3_idx is not None:
    for bin_start in time_bins:
        bin_end = bin_start + bin_samples

        alpha_bin_vals = []
        beta_bin_vals = []
        c3_alpha_bin_vals = []
        c3_beta_bin_vals = []



        for idx in imagery_indices:
            onset_sample = int(annotations.onset[idx] * sfreq)

            # Baseline window (-2 to -1.8s)
            bl_start = int((annotations.onset[idx] - 2.0) * sfreq)
            bl_end = int((annotations.onset[idx] - 1.8) * sfreq)

            # Validate baseline and post windows
            if bl_start < 0 or (onset_sample + bin_end) > mean_power_alpha.shape[1]:
                continue

            # Compute baseline per event (in dB)
            bl_alpha = mean_power_alpha[:, bl_start:bl_end].mean(axis=1)
            bl_beta = mean_power_beta[:, bl_start:bl_end].mean(axis=1)
            bl_alpha_db = 10 * np.log10(bl_alpha + 1e-10)
            bl_beta_db = 10 * np.log10(bl_beta + 1e-10)

            # Extract post-onset bin, convert to dB, subtract baseline
            alpha_bin = mean_power_alpha[:, onset_sample + bin_start:onset_sample + bin_end]
            beta_bin = mean_power_beta[:, onset_sample + bin_start:onset_sample + bin_end]

            alpha_bin_db = 10 * np.log10(alpha_bin.mean(axis=1) + 1e-10) - bl_alpha_db
            beta_bin_db = 10 * np.log10(beta_bin.mean(axis=1) + 1e-10) - bl_beta_db

            alpha_bin_vals.append(alpha_bin_db)
            beta_bin_vals.append(beta_bin_db)
            c3_alpha_bin_vals.append(alpha_bin_db[c3_idx])
            c3_beta_bin_vals.append(beta_bin_db[c3_idx])

        # Compute average across events for this bin
        alpha_binned_topos.append(np.mean(alpha_bin_vals, axis=0))
        beta_binned_topos.append(np.mean(beta_bin_vals, axis=0))
        c3_alpha_power.append(np.mean(c3_alpha_bin_vals))
        c3_beta_power.append(np.mean(c3_beta_bin_vals))

    # Plot C3 power over time
    times_sec = np.array(time_bins) / sfreq
    plt.figure(figsize=(8, 4))
    plt.plot(times_sec, c3_alpha_power, label='Alpha (C3)', marker='o')
    plt.plot(times_sec, c3_beta_power, label='Beta (C3)', marker='s')
    plt.xlabel('Time (s) post-onset')
    plt.ylabel('Power (dB, BL-removed)')
    plt.title('C3 Channel Power Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot topographies over time (binned)
    fig_alpha_bin, axes_alpha_bin = plt.subplots(1, len(alpha_binned_topos), figsize=(4 * len(alpha_binned_topos), 4))
    fig_beta_bin, axes_beta_bin = plt.subplots(1, len(beta_binned_topos), figsize=(4 * len(beta_binned_topos), 4))

    for i, (alpha_topo, beta_topo) in enumerate(zip(alpha_binned_topos, beta_binned_topos)):
        mne.viz.plot_topomap(alpha_topo, raw.info, axes=axes_alpha_bin[i], show=False, cmap='viridis', contours=6)
        axes_alpha_bin[i].set_title(f'{times_sec[i]:.1f}s')

        mne.viz.plot_topomap(beta_topo, raw.info, axes=axes_beta_bin[i], show=False, cmap='viridis', contours=6)
        axes_beta_bin[i].set_title(f'{times_sec[i]:.1f}s')

    fig_alpha_bin.suptitle('Alpha Band Topography (100 ms bins)', fontsize=16)
    fig_beta_bin.suptitle('Beta Band Topography (100 ms bins)', fontsize=16)
    plt.show()

    result = "Averaged power over time bins and plotted C3 power contour."
    
result

# ===============================================================
# 8. Time–Frequency Contour for Channel C3
# ===============================================================
# Compute an evoked response from cleaned epochs for channel C3
c3_evoked = epochs_clean.average(picks=['C3'])

# Define optimized frequencies for contour (e.g., 8–30 Hz in steps of 2)
optimized_freqs = np.arange(8, 31, 2)
# Compute TFR on the evoked data (converting the Evoked to an array with shape [1, n_times])
power_c3 = mne.time_frequency.tfr_array_morlet(c3_evoked.data[np.newaxis, :],
                                               sfreq=sfreq,
                                               freqs=optimized_freqs,
                                               n_cycles=n_cycles,
                                               output='power')[0]
# Now power_c3 has shape (1, n_freqs, n_times); remove channel dimension:
power_c3 = power_c3[0]

# Determine baseline indices for c3_evoked.times (using same baseline period as before)
baseline_mask = (c3_evoked.times >= baseline[0]) & (c3_evoked.times < baseline[1])
if baseline_mask.sum() == 0:
    raise ValueError("No baseline samples found for C3 evoked data.")
baseline_power = power_c3[:, baseline_mask].mean(axis=1, keepdims=True)
# Baseline-corrected power in dB
power_c3_db = 10 * np.log10((power_c3 + 1e-10) / (baseline_power + 1e-10))

# Scale the dB values to the range [-5, 5] via min–max scaling
orig_min = power_c3_db.min()
orig_max = power_c3_db.max()
if orig_max == orig_min:
    power_c3_db_scaled = np.zeros_like(power_c3_db)
else:
    power_norm = (power_c3_db - orig_min) / (orig_max - orig_min)
    power_c3_db_scaled = power_norm * 10 - 5

# Plot the contour
times_sec = c3_evoked.times  # x-axis time vector
plt.figure(figsize=(10, 6))
contour = plt.contourf(times_sec, optimized_freqs, power_c3_db_scaled,
                       levels=40, cmap='viridis', vmin=-5, vmax=5)
plt.colorbar(contour, label='Scaled Power (dB)')
plt.xlabel('Time (s) relative to Imagery Onset')
plt.ylabel('Frequency (Hz)')
plt.title('C3 Time–Frequency Power (Scaled to [-5, 5] dB)')
plt.axvline(x=0, color='r', linestyle='--', label='Imagery Onset')
plt.legend()
plt.show()

# ===============================================================
# (Optional) Export the C3 Time–Frequency Data to CSV
# ===============================================================
df_c3 = pd.DataFrame(power_c3_db_scaled, index=optimized_freqs, columns=np.round(times_sec, 3))
df_c3.index.name = 'Frequency (Hz)'
df_c3.to_csv('c3_time_frequency_power_db_scaled.csv')
print("C3 power saved to: c3_time_frequency_power_db_scaled.csv")
