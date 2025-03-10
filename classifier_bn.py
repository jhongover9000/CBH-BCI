# ==================================================================================================
# IMPORTS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from datetime import datetime
import gc
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from collections import defaultdict
import random
import psutil

from models.EEGModels import EEGNet
from models.atcnet_new import ATCNet_

# ==================================================================================================
# VARIABLES

# Set Seed for Robustness Testing
SEED = 42  # Change this value to test robustness
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set Model Choice ('EEGNet' or 'ATCNet')
model_choice = 'ATCNet'  # Change to 'EEGNet' if needed

# SHAP Analysis Toggle
shap_on = False  # Set to False if you don't need SHAP analysis

# FIX TENSORFLOW MEMORY GROWTH
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Prevent TensorFlow from pre-allocating GPU memory
    except RuntimeError as e:
        print("GPU Memory Growth Error:", e)

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"

# Data Configurations
data_version = 'v4'
mit_data = True
if(mit_data):
    data_filename = f"mit_subject_data_{data_version}.npz"
else:
    data_filename = f"subject_data_{data_version}.npz"

# Load Data
data = np.load(data_dir + data_filename)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

for subj in np.unique(subject_ids):
    subj_idx = np.where(subject_ids == subj)
    X[subj_idx] = (X[subj_idx] - np.mean(X[subj_idx])) / np.std(X[subj_idx])

# LOSO Cross-Validation
n_splits = len(np.unique(subject_ids))
epochs = 70
batch_size = 16
learning_rate = 0.00005
nb_classes = 2

# Timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# Evaluation Tracking
conf_matrix_accum = np.zeros((nb_classes, nb_classes))
accuracy_per_subject = []
loss_per_subject = []
shap_values_all = []
y_test_all = []
y_pred_all = []
misclassified_trials_all = defaultdict(list)
misclassification_stats_all = defaultdict(int)
misclassified_trials_per_subject = []


def print_memory_usage():
    """Prints current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"🛑 Memory Usage: {mem_usage:.2f} GB")


def clear_tf_memory():
    """Clears TensorFlow's GPU memory."""
    K.clear_session()
    gc.collect()

print("Starting Training...")

# ==================================================================================================
# TRAINING LOOP (LOSO)
for subject in np.unique(subject_ids):
    print(f"Processing Subject {subject}...")
    print_memory_usage()

    # Initialize Model
    model = ATCNet_(nb_classes, X.shape[1], X.shape[2]) if model_choice == 'ATCNet' else EEGNet(nb_classes, X.shape[1], X.shape[2])

    # Load Pre-trained Weights
    # try:
    #     model.load_weights(ref_weights_dir + f"{model_choice}_weights.h5", by_name=True, skip_mismatch=True)
    #     print(f"Loaded pre-trained weights for {model_choice}")
    # except:
    #     print("No pre-trained weights found. Training from scratch.")

    # LOSO Data Split
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]

    train_trials = set(map(tuple, X[train_index].reshape(X[train_index].shape[0], -1)))
    test_trials = set(map(tuple, X[test_index].reshape(X[test_index].shape[0], -1)))

    overlapping_trials = train_trials.intersection(test_trials)
    print(f"Number of overlapping trials: {len(overlapping_trials)}")


    # Fix input shape for ATCNet
    X_train = np.expand_dims(X[train_index], axis=1)  # (batch, 1, channels, time)
    X_test = np.expand_dims(X[test_index], axis=1)  # (batch, 1, channels, time)

    y_train = to_categorical(y[train_index], nb_classes)
    y_test = to_categorical(y[test_index], nb_classes)

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f"{saved_weights_dir}{timestamp}_best_model_subject_{subject}.weights.h5", monitor='val_loss', save_weights_only = True)
    ]

    print_memory_usage()

    # Train Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)

    print_memory_usage()

    # Evaluate Model
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_subject.append(scores[1])
    loss_per_subject.append(scores[0])

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Append Accuracies    
    y_test_all.append(y_test.argmax(axis=-1))
    y_pred_all.append(y_pred)

    # SHAP Analysis
    if shap_on:
        print("Computing SHAP values...")
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
        shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  
        shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["BatchMatMulV2"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["Neg"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        # Set up 300 random points for shap
        background = np.array(X_train[np.random.choice(X_train.shape[0], 300, replace=False)])
        print(np.shape(X_test))
        # Create DeepExplainer model
        e = shap.DeepExplainer(model, background)
        print(e)
        shap_values = e.shap_values(X_test, check_additivity=False)
        shap_values_all.append(shap_values)

    # Misclassification Tracking
    misclassified_indices = np.where(y_pred_classes != y_test_classes)[0]
    misclassified_trials_per_subject.append(misclassified_indices)
    for idx in misclassified_indices:
        misclassified_trials_all[subject].append(idx)
        misclassification_stats_all[subject] += 1

    # Compute confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_test_classes, y_pred_classes, num_classes=nb_classes)
    conf_matrix_accum += conf_matrix.numpy()

    print(f"Subject {subject} - Accuracy: {scores[1] * 100:.2f}%")

    # Clear memory
    del model, X_train, y_train, X_test, y_test
    print_memory_usage()
    clear_tf_memory()
    gc.collect()
# ==================================================================================================
# SAVE RESULTS

# Save accuracy results
results_filename = f"{results_dir}{timestamp}_accuracy_results_LOSO.txt"
with open(results_filename, "w") as f:
    f.write("Per-Subject Accuracies:\n")
    for subject, acc in zip(np.unique(subject_ids), accuracy_per_subject):
        f.write(f"Subject {subject}: {acc * 100:.2f}%\n")
    avg_accuracy = np.mean(accuracy_per_subject) * 100
    f.write(f"\nAverage Accuracy Across Subjects: {avg_accuracy:.2f}%\n")

# Save SHAP values
if shap_on:
    with open(f"{shap_dir}{timestamp}_shap_values.pkl", "wb") as fp:
        pickle.dump(shap_values_all, fp)
    with open(f"{shap_dir}{timestamp}_y_test_all", "wb") as fp:  # Pickling
        pickle.dump(y_test_all, fp)
    with open(f"{shap_dir}{timestamp}_y_pred_all", "wb") as fp:  # Pickling
        pickle.dump(y_pred_all, fp)

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_accum, annot=True, fmt=".2f", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(f"{results_dir}{timestamp}_confusion_matrix.png")

print(f"Results saved to {results_filename}")

'''
# ==================================================================================================
# FINAL TRAINING ON ALL DATA

print("Training Final Model on All Subjects...")

# Initialize Final Model
final_model = ATCNet_(nb_classes, X.shape[1], X.shape[2]) if model_choice == 'ATCNet' else EEGNet(nb_classes, X.shape[1], X.shape[2])

# Fix input shape
X_all = np.expand_dims(X, axis=1)  # (batch, 1, channels, time)
y_all = to_categorical(y, nb_classes)

# Compile Final Model
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Final Model
final_model.fit(X_all, y_all, batch_size=batch_size, epochs=epochs, verbose=1)

# Save Final Model
final_model.save_weights(f"{saved_weights_dir}{timestamp}_ATC_final_model.weights.h5")
print("Final Model Trained and Saved.")
'''