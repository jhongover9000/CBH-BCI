import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from pyprep.prep_pipeline import PrepPipeline

# ---------- 1. Load .fif file ----------
data_dir = "./lsl_data/mi/"
file_ext = '.fif'
raw_list = []

# Find all FIF files in the directory
for fname in os.listdir(data_dir):
    if fname.endswith(file_ext):
        file_path = os.path.join(data_dir, fname)
        print(f"Loading {fname}")
        temp = mne.io.read_raw_fif(file_path, preload=True)

        montage = mne.channels.make_standard_montage('standard_1020')
        valid_ch_names = montage.ch_names
        common_ch_names = list(set(temp.ch_names) & set(valid_ch_names))
        temp.pick_channels(common_ch_names)
        temp.set_montage(montage)

        # if 'Cz' in temp.ch_names:
        #     temp.info['bads'] = ['Cz']
        #     temp.interpolate_bads(reset_bads=True)

        # Apply Bandpass Filter (0.5 - 40 Hz)
        temp.filter(0.5, 40.0, fir_design='firwin')

        # Apply Common Average Referencing (CAR)
        temp.set_eeg_reference('average', projection=False)

        annotations = temp.annotations
        imagery_onsets = [onset for onset, desc in zip(annotations.onset, annotations.description) if 'imagery' in desc.lower()]
        if not imagery_onsets:
            continue

        events = np.array([[int(onset * temp.info['sfreq']), 0, 1] for onset in imagery_onsets])
        raw_list.append(temp)

print(raw_list)
# Load and combine
raw = raw_list[0]
raw.append(raw_list[1:])

raw.plot()

from autoreject import AutoReject

# Create epochs to apply autoreject
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)

# AutoReject: detects bad segments + interpolates bad channels per epoch
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)

# Convert back to Raw if needed
# Get cleaned data and metadata
data = epochs_clean.get_data()  # shape: (n_epochs, n_channels, n_times)
n_epochs, n_channels, n_times = data.shape

# Reshape: concatenate epochs along time axis
data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)

# Build new RawArray
info = epochs_clean.info.copy()  # Keep same channel info
raw_clean = mne.io.RawArray(data_reshaped, info)

# Set original montage again
montage = mne.channels.make_standard_montage('standard_1020')
raw_clean.set_montage(montage)

# Now raw_clean is your cleaned Raw object!
raw_clean.plot(n_channels=32, title='Cleaned Raw (from epochs)')

# ---------- 3. Extract annotations ----------
annotations = raw.annotations
imagery_indices = [i for i, desc in enumerate(annotations.description) if 'imagery' in desc.lower()]
assert imagery_indices, "No 'Imagery' events found in annotations."
first_imagery_time = annotations.onset[imagery_indices[0]]

# ---------- 4. Morlet Wavelet Power (Alpha & Beta bands) ----------
sfreq = raw_clean.info['sfreq']
n_cycles = 5

frequencies_alpha = np.arange(8, 14, 1)
frequencies_beta = np.arange(13, 31, 2)

power_alpha = mne.time_frequency.tfr_array_morlet(
    raw_clean.get_data()[np.newaxis, :], sfreq=sfreq, freqs=frequencies_alpha, n_cycles=n_cycles, output='power'
)[0]  # shape: (n_channels, n_freqs, n_times)

power_beta = mne.time_frequency.tfr_array_morlet(
    raw_clean.get_data()[np.newaxis, :], sfreq=sfreq, freqs=frequencies_beta, n_cycles=n_cycles, output='power'
)[0]

mean_power_alpha = power_alpha.mean(axis=1)
mean_power_beta = power_beta.mean(axis=1)

mean_power_alpha_db = 10 * np.log10(mean_power_alpha + 1e-10)
mean_power_beta_db = 10 * np.log10(mean_power_beta + 1e-10)

# ---------- 5. Baseline Correction (-2s to -1.8s relative to each imagery onset) ----------
# Parameters
baseline_start = -1.0  # seconds
baseline_end = -0.5
post_start = -1.0
post_end = 2.0

# Collect power for each imagery event
alpha_event_powers = []
beta_event_powers = []

for idx in imagery_indices:
    onset_time = annotations.onset[idx]
    onset_sample = int(onset_time * sfreq)

    # Baseline sample indices
    bl_start_sample = int((onset_time + baseline_start) * sfreq)
    bl_end_sample = int((onset_time + baseline_end) * sfreq)

    # Post-onset sample indices
    post_start_sample = int((onset_time + post_start) * sfreq)
    post_end_sample = int((onset_time + post_end) * sfreq)

    if bl_start_sample < 0 or post_end_sample > mean_power_alpha.shape[1]:
        continue

    # Baseline power (dB)
    bl_alpha = mean_power_alpha[:, bl_start_sample:bl_end_sample].mean(axis=1)
    bl_beta = mean_power_beta[:, bl_start_sample:bl_end_sample].mean(axis=1)
    bl_alpha_db = 10 * np.log10(bl_alpha + 1e-10)
    bl_beta_db = 10 * np.log10(bl_beta + 1e-10)

    # Post-onset power (dB), baseline-corrected
    alpha_post = 10 * np.log10(mean_power_alpha[:, post_start_sample:post_end_sample] + 1e-10)
    beta_post = 10 * np.log10(mean_power_beta[:, post_start_sample:post_end_sample] + 1e-10)

    alpha_post_bl = alpha_post.mean(axis=1) - bl_alpha_db
    beta_post_bl = beta_post.mean(axis=1) - bl_beta_db

    alpha_event_powers.append(alpha_post_bl)
    beta_event_powers.append(beta_post_bl)

# Grand average across events
alpha_avg = np.mean(alpha_event_powers, axis=0)
beta_avg = np.mean(beta_event_powers, axis=0)

# ---------- 6. Plot Grand Average Topographies ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
mne.viz.plot_topomap(alpha_avg, raw.info, axes=axes[0], show=False, cmap='viridis', contours=6)
axes[0].set_title("Alpha Avg (dB, BL-removed)")

mne.viz.plot_topomap(beta_avg, raw.info, axes=axes[1], show=False, cmap='viridis', contours=6)
axes[1].set_title("Beta Avg (dB, BL-removed)")
plt.suptitle("Grand Avg Topos (0–1s post-imagery)", fontsize=14)
plt.show()

# Define time bins (100 ms each) from 0 to 1 second post-imagery
time_bin_ms = 100
bin_samples = int((time_bin_ms / 1000) * sfreq)
post_start_offset = 0  # relative to imagery onset
post_end_offset = int(1.0 * sfreq)
time_bins = list(range(post_start_offset, post_end_offset, bin_samples))

# Initialize lists for binned topographies and C3 power over time
alpha_binned_topos = []
beta_binned_topos = []
c3_alpha_power = []
c3_beta_power = []

# Get index of C3 channel
try:
    c3_idx = raw.ch_names.index('C3')
except ValueError:
    c3_idx = None
    result = "C3 channel not found in data."

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


# ---------- 7. C3 Power Over Time (100 ms bins, 0–1s) ----------
c3_idx = raw.ch_names.index('C3')
bin_samples = int(0.1 * sfreq)
c3_alpha_power = []
c3_beta_power = []

for bin_start in range(0, int(1.0 * sfreq), bin_samples):
    bin_end = bin_start + bin_samples
    alpha_vals, beta_vals = [], []

    for idx in imagery_indices:
        onset_sample = int(annotations.onset[idx] * sfreq)
        bl_start = int((annotations.onset[idx] - 2.0) * sfreq)
        bl_end = int((annotations.onset[idx] - 1.8) * sfreq)

        bl_alpha = mean_power_alpha[:, bl_start:bl_end].mean(axis=1)
        bl_beta = mean_power_beta[:, bl_start:bl_end].mean(axis=1)
        bl_alpha_db = 10 * np.log10(bl_alpha + 1e-10)
        bl_beta_db = 10 * np.log10(bl_beta + 1e-10)

        alpha_bin = mean_power_alpha[:, onset_sample + bin_start:onset_sample + bin_end].mean(axis=1)
        beta_bin = mean_power_beta[:, onset_sample + bin_start:onset_sample + bin_end].mean(axis=1)

        alpha_bin_db = 10 * np.log10(alpha_bin + 1e-10) - bl_alpha_db
        beta_bin_db = 10 * np.log10(beta_bin + 1e-10) - bl_beta_db

        alpha_vals.append(alpha_bin_db[c3_idx])
        beta_vals.append(beta_bin_db[c3_idx])

    c3_alpha_power.append(np.mean(alpha_vals))
    c3_beta_power.append(np.mean(beta_vals))

# Plot C3 over time
times = np.arange(0, 1.0, 0.1)
plt.plot(times, c3_alpha_power, label='Alpha (C3)', marker='o')
plt.plot(times, c3_beta_power, label='Beta (C3)', marker='s')
plt.xlabel('Time (s)')
plt.ylabel('Power (dB, BL-removed)')
plt.title('C3 Power Over Time')
plt.legend()
plt.grid(True)
plt.show()

# ---------- 8. Time-Frequency Contour at C3 (±5s around first imagery, Safe Version) ----------
optimized_freqs = np.arange(8, 31, 2)
n_cycles = 5  # Already defined earlier
sfreq = raw.info['sfreq']
c3_idx = raw.ch_names.index('C3')

# Time window: -5 to +5 seconds relative to first imagery onset
trim_start_sample = int((first_imagery_time - 5.0) * sfreq)
trim_end_sample = int((first_imagery_time + 5.0) * sfreq)

# Ensure trimming indices are within data bounds
total_samples = raw.n_times
trim_start_sample = max(trim_start_sample, 0)
trim_end_sample = min(trim_end_sample, total_samples)

# Extract C3 data for the time window, shape: (1, n_times)
raw_c3_data = raw.get_data(picks=[c3_idx])[:, trim_start_sample:trim_end_sample]

# Compute Morlet power: result shape (1, n_freqs, n_times)
power_c3 = mne.time_frequency.tfr_array_morlet(
    raw_c3_data[np.newaxis, :], sfreq=sfreq, freqs=optimized_freqs, n_cycles=n_cycles, output='power'
)[0]  # Now shape: (1, n_freqs, n_times)

# Remove channel dimension (since it's just C3), now shape: (n_freqs, n_times)
power_c3 = power_c3[0]

# Compute baseline indices within the trimmed data
baseline_start_sec = -2.0
baseline_end_sec = -1.8
baseline_start_sample = int((first_imagery_time + baseline_start_sec) * sfreq) - trim_start_sample
baseline_end_sample = int((first_imagery_time + baseline_end_sec) * sfreq) - trim_start_sample

# Validate baseline indices
if 0 <= baseline_start_sample < baseline_end_sample <= power_c3.shape[1]:
    baseline = power_c3[:, baseline_start_sample:baseline_end_sample].mean(axis=1, keepdims=True)  # (n_freqs, 1)
else:
    raise ValueError(
        f"Baseline indices out of range: start={baseline_start_sample}, end={baseline_end_sample}, "
        f"data length={power_c3.shape[1]}"
    )

# Baseline-corrected power (dB)
power_c3_db = 10 * np.log10((power_c3 + 1e-10) / (baseline + 1e-10))  # shape: (n_freqs, n_times)

# Time vector for x-axis (seconds relative to imagery onset)
times_sec = np.linspace(-5, 5, power_c3_db.shape[1])

# Plot contour with fixed color scale: -5 to 5 dB
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(
#     times_sec, optimized_freqs, power_c3_db, levels=40, cmap='viridis', vmin=-5, vmax=5
# )
# plt.colorbar(contour, label='Power (dB, BL-removed)')
# plt.xlabel('Time (s) relative to Imagery Onset')
# plt.ylabel('Frequency (Hz)')
# plt.title('C3 Time-Frequency Power (Baseline Corrected)')
# plt.axvline(x=0, color='r', linestyle='--', label='Imagery Onset')
# plt.legend()
# plt.show()


# Get original min and max from the power values
orig_min = power_c3_db.min()
orig_max = power_c3_db.max()

# Avoid division by zero if all values are the same
if orig_max == orig_min:
    power_c3_db_scaled = np.zeros_like(power_c3_db)  # All values zero
else:
    # Min-max scale to [0, 1]
    power_norm = (power_c3_db - orig_min) / (orig_max - orig_min)
    # Scale to [-5, 5]
    power_c3_db_scaled = power_norm * 10 - 5

# Plot contour using scaled power values
plt.figure(figsize=(10, 6))
contour = plt.contourf(
    times_sec, optimized_freqs, power_c3_db_scaled, levels=40, cmap='viridis', vmin=-5, vmax=5
)
plt.colorbar(contour, label='Scaled Power (dB)')
plt.xlabel('Time (s) relative to Imagery Onset')
plt.ylabel('Frequency (Hz)')
plt.title('C3 Time-Frequency Power (Scaled to [-5, 5] dB)')
plt.axvline(x=0, color='r', linestyle='--', label='Imagery Onset')
plt.legend()
plt.show()





# Export to CSV
# df_c3 = pd.DataFrame(power_c3_db, index=optimized_freqs, columns=np.round(times_sec, 3))
# df_c3.index.name = 'Frequency (Hz)'
# df_c3.to_csv('c3_time_frequency_power_db_trimmed.csv')
# print("C3 power saved to: c3_time_frequency_power_db_trimmed.csv")

