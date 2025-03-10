import numpy as np
import os

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"

# Data Configurations
data_version = 'v3'
new_version = 'v4'
mit_data = True
if(mit_data):
    data_filename = f"mit_subject_data_{data_version}.npz"
    save_path = f"{data_dir}mit_subject_data_{new_version}.npz"
else:
    data_filename = f"subject_data_{data_version}.npz"
    save_path = f"{data_dir}subject_data_{new_version}.npz"

# Load Data
data = np.load(data_dir + data_filename)

X = data['X']  # EEG Trials
y = data['y']  # Labels
subject_ids = data['subject_ids']  # Subject IDs

print("Original Data Shape:", X.shape)

# Identify unique trials
unique_indices = np.unique(X, axis=0, return_index=True)[1]
X_cleaned = X[unique_indices]
y_cleaned = y[unique_indices]
subject_ids_cleaned = subject_ids[unique_indices]

print("Cleaned Data Shape:", X_cleaned.shape)

# ===== OUTLIER REMOVAL =====

# Compute per-channel mean and std for outlier detection
channel_means = np.mean(X_cleaned, axis=(0, 2), keepdims=True)  # Mean per channel
channel_stds = np.std(X_cleaned, axis=(0, 2), keepdims=True)  # Std per channel

# Clip values beyond ±5 standard deviations
X_cleaned = np.clip(X_cleaned, channel_means - 5 * channel_stds, channel_means + 5 * channel_stds)

# ===== STANDARDIZATION (Per-Channel) =====

# Standardize per channel (zero mean, unit variance per channel)
X_cleaned = (X_cleaned - channel_means) / channel_stds

# ===== BAD CHANNEL DETECTION =====

# Compute variance per channel
global_channel_variance = np.var(X_cleaned, axis=(0, 2))
threshold = np.median(global_channel_variance) * 3  # 3x median variance as threshold
bad_channels = np.where(global_channel_variance > threshold)[0]

print(f"Bad Channels Detected: {bad_channels.tolist()}")

# Save cleaned dataset
np.savez(save_path, X=X_cleaned, y=y_cleaned, subject_ids=subject_ids_cleaned)

print("EEG Value Range After Processing:")
print("Min:", X_cleaned.min(), "Max:", X_cleaned.max(), "Mean:", X_cleaned.mean(), "Std:", X_cleaned.std())
print(f"Cleaned dataset saved to {save_path}")