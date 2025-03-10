import numpy as np

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"

# Load the dataset (update path if necessary)
data_path = f"{data_dir}/subject_data_v7.npz"
data = np.load(data_path)

X = data['X']
y = data['y']
subject_ids = data['subject_ids']

# 1. Check dataset shapes
print("X shape:", X.shape)  # Should be (trials, channels, timepoints)
print("y shape:", y.shape)  # Should match number of trials
print("subject_ids shape:", subject_ids.shape)  # Should match number of trials

# 2. Check unique labels and their distribution
unique_labels, label_counts = np.unique(y, return_counts=True)
print("Label Distribution:", dict(zip(unique_labels, label_counts)))

# 3. Check for train-test overlap (unique subjects)
unique_subjects = np.unique(subject_ids)
print("Unique Subjects:", len(unique_subjects))

# 4. Check for duplicate trials
num_unique_trials = len(np.unique(X, axis=0))
num_total_trials = X.shape[0]
print("Duplicate Trials:", num_total_trials - num_unique_trials)

# 5. Check EEG signal value range
print("EEG Value Range:")
print("Min:", X.min(), "Max:", X.max(), "Mean:", X.mean(), "Std:", X.std())

import matplotlib.pyplot as plt

plt.hist(X.flatten(), bins=100)
plt.xlabel("EEG Signal Value")
plt.ylabel("Count")
plt.title("Distribution of EEG Values")
plt.show()
