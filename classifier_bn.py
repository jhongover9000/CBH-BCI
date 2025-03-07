# ==================================================================================================
# IMPORTS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from datetime import datetime
import gc
from models.EEGModels import EEGNet
from models.atcnet_new import ATCNet_
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# ==================================================================================================
# VARIABLES

# Set Model Choice ('EEGNet' or 'ATCNet')
model_choice = 'ATCNet'  # Change to 'EEGNet' if needed

# GPU Check
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"

# Data Configurations
data_version = 'v2'
data_filename = f"subject_data_{data_version}.npz"

# Load Data
data = np.load(data_dir + data_filename)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# LOSO Cross-Validation
n_splits = len(np.unique(subject_ids))
epochs = 70
batch_size = 16
learning_rate = 0.00005
nb_classes = 2

# Timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# ==================================================================================================
# TRAINING LOOP

accuracy_per_subject = []

print("Starting Training...")

for subject in np.unique(subject_ids):
    print(f"Processing Subject {subject}...")

    # Initialize Model
    model = ATCNet_(nb_classes, X.shape[1], X.shape[2]) if model_choice == 'ATCNet' else EEGNet(nb_classes, X.shape[1], X.shape[2])

    # Load Pre-trained Weights
    try:
        model.load_weights(ref_weights_dir + f"{model_choice}_weights.h5", by_name=True, skip_mismatch=True)
        print(f"Loaded pre-trained weights for {model_choice}")
    except:
        print("No pre-trained weights found. Training from scratch.")

    # LOSO Data Split
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]
    # Expand dimensions to match (batch_size, 1, channels, time_samples)
    X_train = np.expand_dims(X[train_index], axis=1)  # (batch, 1, channels, time)
    X_test = np.expand_dims(X[test_index], axis=1)  # (batch, 1, channels, time)
    y_train, y_test = to_categorical(y[train_index], nb_classes), to_categorical(y[test_index], nb_classes)

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f"{saved_weights_dir}{timestamp}_best_model_subject_{subject}.weights.h5",
                        monitor='val_loss', save_weights_only=True)
    ]

    # Train Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)

    # Evaluate Model
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_subject.append(scores[1])

print("Training Complete.")

