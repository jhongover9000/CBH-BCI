import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ion()
plt.show()
print(plt.isinteractive())

# Replace this with the actual filename of your saved FIF file.
fname = 'MI_ME_10_2.fif'

# Load the raw data (preload=True allows epoch extraction)
raw = mne.io.read_raw_fif(fname, preload=True)

# === Plot Raw Data ===

raw.plot(duration=5.0, n_channels=len(raw.ch_names), scalings='auto', title='Raw EEG')

plt = raw.plot()
plt.show()

# Check the annotations to verify marker names
print("Annotations:", raw.annotations.description)

# Extract events and the mapping from annotations.
# This function converts each annotation into an event (with a numeric code).
events, event_dict_all = mne.events_from_annotations(raw)
print("All events:", event_dict_all)

# Create a new event dictionary that includes only the desired markers.
# We assume that your markers are named "execution" and "imagery".
event_id = {}
if 'execution' in event_dict_all:
    event_id['execution'] = event_dict_all['execution']
if 'imagery' in event_dict_all:
    event_id['imagery'] = event_dict_all['imagery']

print("Filtered event_id:", event_id)

# Now extract epochs using tmin=-2 and tmax=2 seconds relative to marker onset.
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-2, tmax=2, preload=True)
print(epochs)

# Optionally, you can inspect the epochs:
epochs.plot()
