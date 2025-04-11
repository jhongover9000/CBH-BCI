import numpy as np
import mne
import scipy.io
import matplotlib.pyplot as plt

# === Config ===
data_dir = "./lsl_data/combined/"
file_ext = '.fif'
filename = "MI_ME_10_2"
use_mat = False  # Set to True for .mat files
save_topos = True

# === Load Data ===
if use_mat:
    mat = scipy.io.loadmat(f"{filename}.mat")
    data = mat['data']
    timestamps = mat['timestamps_ms'][0] / 1000  # Convert to seconds
    marker_labels = [str(label[0]) for label in mat['markers'][0]]
    marker_rel_times = mat['marker_rel_times_ms'][0] / 1000  # seconds
    sfreq = float(mat['sfreq'][0][0])
    channels = [str(ch[0]) for ch in mat['channels'][0]]
    
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # Create Annotations
    annotations = mne.Annotations(onset=marker_rel_times.tolist(),
                                  duration=[0]*len(marker_rel_times),
                                  description=marker_labels)
    raw.set_annotations(annotations)
else:
    raw = mne.io.read_raw_fif(f"{data_dir}{filename}.fif", preload=True)

# === Apply Standard 10-20 Montage ===
montage = mne.channels.make_standard_montage('standard_1020')  # or 'standard_1010' for higher-density caps
raw.set_montage(montage)
print("✅ Applied standard 10-20 montage to EEG data.")

# === Plot Raw Data ===

raw.plot(duration=5.0, n_channels=len(raw.ch_names), scalings='auto', title='Raw EEG')

# === Filter and Preprocess ===
raw.notch_filter(50)
raw.filter(1, 40, fir_design='firwin')
raw.plot(duration=5.0, n_channels=len(raw.ch_names), scalings='auto', title='BPF')

raw.set_eeg_reference('average')
raw.plot(duration=5.0, n_channels=len(raw.ch_names), scalings='auto', title='CAR')

# === Extract MI Epochs ===
events, event_id = mne.events_from_annotations(raw)

if 'imagery' not in event_id:
    raise ValueError("No 'Imagery' annotations found.")

epoch_tmin = -3
epoch_tmax = 5  # seconds
epochs = mne.Epochs(raw, events, event_id=event_id['imagery'],
                    tmin=epoch_tmin, tmax=epoch_tmax,
                    baseline=(-2, -1.8), detrend=1, preload=True, reject=None)

import numpy as np
from scipy.io import savemat

# === Assume `epochs` already created ===
# data shape: (n_epochs, n_channels, n_times)
epoch_data = epochs.get_data()
sfreq = epochs.info['sfreq']
channels = epochs.info['ch_names']
times = epochs.times  # in seconds
labels = epochs.events[:, 2]  # Event IDs (e.g., 1 for 'Imagery')

# Optional: get label mapping
event_id_map = epochs.event_id  # {'Imagery': 1, ...}

# === Save to .mat ===
savemat("epochs_data.mat", {
    'epoch_data': epoch_data,  # (n_epochs, n_channels, n_times)
    'times': times,
    'sfreq': sfreq,
    'channels': channels,
    'labels': labels,
    'event_id_map': event_id_map
})

print("✅ Saved epochs to 'epochs_data.mat'")

'''

# Define frequency band (e.g., 8–12 Hz for alpha)
freq_band = (8, 12)
freqs = np.linspace(freq_band[0], freq_band[1], 5)  # 5 freqs in band
n_cycles = freqs / 2.  # Typical: cycles = freq / 2

# Time-frequency decomposition
power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                      return_itc=False, average=True, decim=1)

# Collapse across frequency dimension → mean power per timepoint
band_power = power.copy().crop(fmin=freq_band[0], fmax=freq_band[1]).data.mean(axis=0)



# === Topographies Over Time ===
epoch_length_ms = int((epoch_tmax - epoch_tmin) * 1000)
time_windows = np.arange(0, epoch_length_ms + 1, 200) / 1000  # seconds

import math

# # === Set Up Grid for Subplots ===
n_times = len(time_windows)
cols = 5  # Number of columns in the figure
rows = math.ceil(n_times / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

# === Plot All Topographies Into Subplots ===
for idx, t in enumerate(time_windows):
    epochs.average().plot_topomap(times=t, ch_type='eeg', time_unit='s',
                                  axes=axes[idx], show=False, colorbar=False)
    axes[idx].set_title(f"{int(t*1000)} ms")

# Hide any unused subplots
for ax in axes[n_times:]:
    ax.axis('off')

fig.suptitle("Topographies Over Time (Imagery Epochs)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Show once
plt.show()

# Save if needed
if save_topos:
    fig.savefig("topographies_batched.png")
'''
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# # === Prepare Figure and Initial Plot ===
# fig = plt.figure(figsize=(6, 6))
# gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)

# # Topomap axis
# ax_topo = fig.add_subplot(gs[0])
# # Colorbar axis
# ax_cbar = fig.add_subplot(gs[1])

# plt.subplots_adjust(bottom=0.25)  # Make space for slider

# evoked = epochs.average()

# # Initial topomap at first time window
# evoked.plot_topomap(times=time_windows[0], ch_type='eeg',
#                     axes=(ax_topo, ax_cbar), show=False, colorbar=True)
# fig.suptitle(f"Topography at {int(time_windows[0]*1000)} ms")

# # === Slider Setup ===
# ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
# slider = Slider(ax_slider, 'Time (ms)', 0, time_windows[-1]*1000,
#                 valinit=time_windows[0]*1000, valstep=200)

# # === Update Function ===
# def update(val):
#     t_sec = slider.val / 1000
#     ax_topo.clear()
#     ax_cbar.clear()
#     evoked.plot_topomap(times=t_sec, ch_type='eeg',
#                         axes=(ax_topo, ax_cbar), show=False, colorbar=True)
#     fig.suptitle(f"Topography at {int(t_sec*1000)} ms")
#     fig.canvas.draw_idle()

# slider.on_changed(update)
# plt.show()
