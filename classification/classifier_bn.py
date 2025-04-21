'''
CLASSIFIER (W MODEL BATCH NORMALIZATION) - Incremental LOSO

Description: Classification training for a model with batch normalization,
             modified for incremental LOSO processing and result appending
             to handle memory limitations and allow resuming.
Should be run as 'python -m classification.classifier_bn.py'

Joseph Hong (Modified for Incremental Processing)

'''

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
from sklearn.preprocessing import StandardScaler
import json # Added for easier dictionary saving/loading

from models.EEGModels import EEGNet
from models.atcnet_new import ATCNet_

# ==================================================================================================
# VARIABLES

# Seeds
SEED = 128
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set Model Choice ('EEGNet' or 'ATCNet')
model_choice = 'ATCNet'  # Change to 'EEGNet' if needed

# SHAP Analysis Toggle
shap_on = False  # Set to False if you don't need SHAP analysis

# FIX TENSORFLOW MEMORY GROWTH (PARTIAL FIX)
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
progress_file = "./progress_tracker.txt" # File to track progress

# --- Create Directories if they don't exist ---
os.makedirs(saved_weights_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(shap_dir, exist_ok=True)
if shap_on:
    # Create a subdirectory for individual subject SHAP files
    shap_subject_dir = os.path.join(shap_dir, "subject_shap_values")
    os.makedirs(shap_subject_dir, exist_ok=True)


# Data Configurations
data_type = 'bci'
data_version = 'v1'
if(data_type == 'mit'):
    data_filename = f"mit_subject_data_{data_version}.npz"
    weight_name = "ST"
elif (data_type == 'xon'):
    data_filename = f"xon_subject_data_{data_version}.npz"
    weight_name = "XON"
elif (data_type == 'bci'):
    data_filename = f"bci_subject_data_{data_version}.npz"
    weight_name = "BCI"
else:
    data_filename = f"subject_data_{data_version}.npz"
    weight_name = "NT"

# Load Data
try:
    data = np.load(os.path.join(data_dir, data_filename))
    X = data['X']
    y = data['y']
    subject_ids = data['subject_ids']
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {os.path.join(data_dir, data_filename)}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# Standard scaling per subject (Keep this as is based on original script)
# Note: A stricter LOSO might fit scaler only on training subjects *within* the loop
scalers = {}
unique_subject_list = np.unique(subject_ids)
for subj in unique_subject_list:
    subj_idx = np.where(subject_ids == subj)[0] # Get indices for the subject
    if len(subj_idx) == 0:
        print(f"Warning: No data found for subject {subj}. Skipping scaling for this subject.")
        continue

    # Reshape data for scaling: (n_trials, n_features) where n_features = channels * timepoints
    original_shape = X[subj_idx].shape
    X_subj = X[subj_idx].reshape(original_shape[0], -1)

    scaler = StandardScaler()
    try:
        X_subj_scaled = scaler.fit_transform(X_subj)
        # Reshape back to original shape: (n_trials, channels, timepoints) or similar
        X[subj_idx] = X_subj_scaled.reshape(original_shape)
        scalers[subj] = scaler
    except ValueError as e:
        print(f"Error scaling data for subject {subj}: {e}. Check if data for this subject is valid.")
        # Decide how to handle: skip subject, use unscaled data, etc.
        # For now, we'll proceed but this subject's data won't be scaled.


# LOSO Cross-Validation Parameters
n_splits = len(unique_subject_list)
epochs = 70
batch_size = 16
learning_rate = 0.00005
nb_classes = len(np.unique(y)) # Determine number of classes from data
weight_decay = 0.01

# Timestamp for this run (used for all related output files)
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# --- Define output file paths ---
accuracy_log_file = os.path.join(results_dir, f"{timestamp}_accuracy_log_LOSO.csv")
predictions_file = os.path.join(results_dir, f"{timestamp}_predictions_LOSO.npz")
misclassification_file = os.path.join(results_dir, f"{timestamp}_misclassifications_LOSO.json")

# Evaluation Tracking (Only confusion matrix needs accumulation)
conf_matrix_accum = np.zeros((nb_classes, nb_classes))
# Removed: accuracy_per_subject, loss_per_subject, shap_values_all, y_test_all, y_pred_all
# Removed: misclassified_trials_all, misclassification_stats_all, misclassified_trials_per_subject
# These will be handled by appending to files


# --- Initialize Result Files (write headers if new) ---
if not os.path.exists(accuracy_log_file):
    with open(accuracy_log_file, 'w') as f:
        f.write("Subject,Loss,Accuracy\n") # CSV Header

# For NPZ, we'll load existing if available, otherwise start fresh
all_predictions = {}
if os.path.exists(predictions_file):
     try:
         loaded_data = np.load(predictions_file, allow_pickle=True)
         # Assuming data is saved under a key, e.g., 'predictions'
         # Adjust key if necessary based on how you save it later
         if 'predictions' in loaded_data:
              all_predictions = loaded_data['predictions'].item() # .item() retrieves the dictionary
         else:
              print(f"Warning: Predictions file {predictions_file} exists but has unexpected format. Starting fresh.")
              all_predictions = {} # Initialize empty if format is wrong
     except Exception as e:
          print(f"Warning: Could not load existing predictions file {predictions_file}. Starting fresh. Error: {e}")
          all_predictions = {}
else:
     all_predictions = {} # Initialize empty if file doesn't exist


# For JSON misclassification data
all_misclassifications = {}
if os.path.exists(misclassification_file):
    try:
        with open(misclassification_file, 'r') as f:
            all_misclassifications = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode existing misclassification file {misclassification_file}. Starting fresh.")
        all_misclassifications = {}
    except Exception as e:
        print(f"Warning: Could not load existing misclassification file {misclassification_file}. Starting fresh. Error: {e}")
        all_misclassifications = {}
else:
    all_misclassifications = {}


def print_memory_usage():
    """Prints current memory usage for debugging."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"🛑 Memory Usage: {mem_usage:.2f} GB")


def clear_tf_memory():
    """Clears TensorFlow's GPU memory and runs garbage collection."""
    K.clear_session()
    gc.collect()
    print("Cleared TF session and ran garbage collection.")


# --- Determine Starting Point ---
start_subject_index = 0
if os.path.exists(progress_file):
    try:
        with open(progress_file, 'r') as f:
            last_processed_index_str = f.read().strip()
            if last_processed_index_str: # Check if the file is not empty
                 last_processed_index = int(last_processed_index_str)
                 start_subject_index = last_processed_index + 1
                 print(f"Resuming from subject index {start_subject_index} (after subject {unique_subject_list[last_processed_index]})")
            else:
                 print("Progress file is empty. Starting from the beginning.")
    except ValueError:
        print(f"Warning: Could not parse index from {progress_file}. Starting from beginning.")
    except FileNotFoundError:
         print("Progress file not found. Starting from the beginning.")
    except IndexError:
         print(f"Warning: Index in {progress_file} is out of bounds for current subject list. Starting from beginning.")
         # This could happen if the subject list changed between runs. Resetting is safest.
         start_subject_index = 0
    except Exception as e:
         print(f"Error reading progress file: {e}. Starting from beginning.")


print("Starting Training...")
# ==================================================================================================
# TRAINING LOOP (Incremental LOSO)

for i in range(start_subject_index, len(unique_subject_list)):
    subject = unique_subject_list[i]
    print(f"\n===== Processing Subject {subject} (Index: {i}) =====")
    print_memory_usage()

    # --- Check if subject results already exist (optional skip) ---
    # This is useful if you only want to generate missing results,
    # but be careful if you want to re-run a specific subject.
    # if subject in all_predictions: # Check if predictions exist for this subject
    #      print(f"Results for subject {subject} already found in {predictions_file}. Skipping.")
    #      # Update progress tracker even if skipped
    #      with open(progress_file, 'w') as f:
    #           f.write(str(i))
    #      continue # Skip to the next subject


    # --- Prepare Data for Current Fold ---
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]

    # Simple overlap check (optional, can be removed if not needed)
    # train_trials_set = set(map(tuple, X[train_index].reshape(X[train_index].shape[0], -1)))
    # test_trials_set = set(map(tuple, X[test_index].reshape(X[test_index].shape[0], -1)))
    # overlapping_trials = train_trials_set.intersection(test_trials_set)
    # print(f"Number of overlapping trials (based on raw data): {len(overlapping_trials)}")


    # Select data for current fold
    X_train_subj = X[train_index]
    y_train_subj = y[train_index]
    X_test_subj = X[test_index]
    y_test_subj = y[test_index]

    # Reshape input for models (ATCNet expects time dim last)
    # Assuming X has shape (trials, channels, timepoints)
    # EEGNet might need (trials, channels, timepoints, 1)
    # ATCNet needs (trials, 1, channels, timepoints) -> This needs adjustment based on actual ATCNet input
    # The original script used np.expand_dims(X, axis=1) suggesting (trials, 1, channels, timepoints)
    if model_choice == 'ATCNet':
        X_train = np.expand_dims(X_train_subj, axis=1) # (batch, 1, channels, time)
        X_test = np.expand_dims(X_test_subj, axis=1)   # (batch, 1, channels, time)
    elif model_choice == 'EEGNet':
         # EEGNet standard input is often (trials, channels, timepoints, 1)
         # Check the specific implementation of EEGNet used
         # Assuming standard input:
         X_train = np.expand_dims(X_train_subj, axis=-1) # (batch, channels, time, 1)
         X_test = np.expand_dims(X_test_subj, axis=-1)   # (batch, channels, time, 1)
         # If EEGNet implementation takes (trials, 1, channels, timepoints), use:
         # X_train = np.expand_dims(X_train_subj, axis=1)
         # X_test = np.expand_dims(X_test_subj, axis=1)
    else:
         # Default or other models
         X_train = X_train_subj
         X_test = X_test_subj


    y_train = to_categorical(y_train_subj, nb_classes)
    y_test = to_categorical(y_test_subj, nb_classes)


    # --- Initialize and Compile Model ---
    print(f"Input shape for model: X_train: {X_train.shape}, X_test: {X_test.shape}")
    K.clear_session() # Clear previous model graph from memory
    gc.collect()

    if model_choice == 'ATCNet':
        model = ATCNet_(nb_classes, X_train.shape[2], X_train.shape[3]) # Adjust params based on ATCNet input
    elif model_choice == 'EEGNet':
         # Ensure EEGNet parameters match the input shape after potential expansion
         model = EEGNet(nb_classes=nb_classes, Chans=X_train.shape[1], Samples=X_train.shape[2]) # Adjust if needed
    else:
         print(f"Error: Unknown model_choice '{model_choice}'")
         continue # Skip this subject if model choice is invalid


    # Load Pre-trained Weights (Optional)
    # try:
    #     model.load_weights(ref_weights_dir + f"{model_choice}_weights.h5", by_name=True, skip_mismatch=True)
    #     print(f"Loaded pre-trained weights for {model_choice}")
    # except Exception as e:
    #     print(f"Could not load pre-trained weights: {e}. Training from scratch.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # model.summary() # Optional: print model summary

    # --- Callbacks ---
    model_checkpoint_path = os.path.join(saved_weights_dir, f"{timestamp}_best_model_subject_{subject}.weights.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1), # Increased patience slightly
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1), # Adjusted min_lr
        ModelCheckpoint(model_checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0) # Save only best weights
    ]

    print_memory_usage()

    # --- Train Model ---
    print(f"Training model for subject {subject}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2 # Use 2 for less verbose output per epoch, 1 for more detail
    )

    print("Training finished.")
    print_memory_usage()

    # --- Load Best Weights ---
    # EarlyStopping restores best weights automatically if restore_best_weights=True
    # If not using that, or for certainty, load the saved best weights:
    try:
         print(f"Loading best weights from {model_checkpoint_path}")
         model.load_weights(model_checkpoint_path)
    except Exception as e:
         print(f"Warning: Could not load best weights for subject {subject} from {model_checkpoint_path}. Using weights from end of training. Error: {e}")


    # --- Evaluate Model ---
    print(f"Evaluating model for subject {subject}...")
    scores = model.evaluate(X_test, y_test, verbose=0)
    subject_loss = scores[0]
    subject_accuracy = scores[1]
    print(f"Subject {subject} - Loss: {subject_loss:.4f}, Accuracy: {subject_accuracy * 100:.2f}%")

    # --- Append Accuracy Results ---
    try:
        with open(accuracy_log_file, 'a') as f:
            f.write(f"{subject},{subject_loss:.6f},{subject_accuracy:.6f}\n")
    except Exception as e:
        print(f"Error writing to accuracy log file: {e}")


    # --- Predict on test data ---
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_test_classes = np.argmax(y_test, axis=1) # Original integer labels

    # --- Append Predictions ---
    # Store both true labels and predicted probabilities (more informative)
    subject_predictions = {
        'y_test': y_test_classes,
        'y_pred_prob': y_pred_prob.astype(np.float32) # Use float32 to save space
    }
    all_predictions[subject] = subject_predictions
    try:
         # Save the entire dictionary to the NPZ file after processing each subject
         # Use allow_pickle=True if storing complex objects, though numpy arrays are fine
         np.savez_compressed(predictions_file, predictions=all_predictions) # Use compression
         print(f"Appended predictions for subject {subject} to {predictions_file}")
    except Exception as e:
         print(f"Error saving predictions to NPZ file: {e}")


    # --- SHAP Analysis (if enabled) ---
    if shap_on:
        print("Computing SHAP values...")
        try:
            # Apply necessary SHAP patches (keep these updated based on TF/SHAP versions)
            shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["BatchMatMulV2"] = shap.explainers._deep.deep_tf.passthrough
            shap.explainers._deep.deep_tf.op_handlers["Neg"] = shap.explainers._deep.deep_tf.passthrough

            # Select background data (use a smaller subset of training data)
            # Ensure background data has the same shape as X_test
            num_background_samples = min(100, X_train.shape[0]) # Limit background size
            background_indices = np.random.choice(X_train.shape[0], num_background_samples, replace=False)
            background = X_train[background_indices]
            print(f"Using {num_background_samples} background samples for SHAP.")
            print(f"Test data shape for SHAP: {X_test.shape}")

            # Create DeepExplainer
            e = shap.DeepExplainer(model, background)

            # Compute SHAP values (consider batching for very large X_test)
            shap_values = e.shap_values(X_test, check_additivity=False) # Returns list (one per output class) of arrays

            # Save SHAP values for this subject to a separate file
            shap_filename = os.path.join(shap_subject_dir, f"{timestamp}_subject_{subject}_shap.pkl")
            # Consider saving as .npz if only numpy arrays: np.savez_compressed(shap_filename, *shap_values)
            try:
                with open(shap_filename, "wb") as fp:
                    pickle.dump(shap_values, fp)
                print(f"Saved SHAP values for subject {subject} to {shap_filename}")
            except Exception as e:
                print(f"Error saving SHAP values for subject {subject}: {e}")

            del e, shap_values, background # Clean up SHAP objects
            gc.collect()

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Error during SHAP analysis for subject {subject}: {e}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Continue without SHAP for this subject if it fails


    # --- Misclassification Tracking ---
    misclassified_indices = np.where(y_pred_classes != y_test_classes)[0]

    # --- FIX: Convert subject ID (key) to string for JSON compatibility ---
    subject_key = str(subject) # Convert numpy int to python string

    # Store original indices relative to the subject's test set
    all_misclassifications[subject_key] = misclassified_indices.tolist() # Use the string key

    # Save updated misclassification dictionary
    try:
        with open(misclassification_file, 'w') as f:
            json.dump(all_misclassifications, f, indent=4) # Now uses string keys
    except TypeError as e: # Catch specific error if needed, though conversion should fix it
         print(f"Error writing misclassification data (TypeError): {e}")
    except Exception as e:
        print(f"Error writing misclassification data: {e}")


    # --- Update Accumulated Confusion Matrix ---
    conf_matrix = tf.math.confusion_matrix(y_test_classes, y_pred_classes, num_classes=nb_classes)
    conf_matrix_accum += conf_matrix.numpy()


    # --- Clean Up Memory for Next Iteration ---
    del model, X_train, y_train, X_test, y_test, X_train_subj, y_train_subj, X_test_subj, y_test_subj
    del y_pred_prob, y_pred_classes, y_test_classes, conf_matrix, history, callbacks # Explicitly delete large objects
    print_memory_usage()
    clear_tf_memory() # Clear TF session memory
    gc.collect() # Force garbage collection
    print(f"===== Finished processing Subject {subject} =====")

    # --- Update Progress Tracker ---
    try:
        with open(progress_file, 'w') as f:
            f.write(str(i)) # Write the index of the subject just completed
    except Exception as e:
        print(f"Error updating progress tracker file: {e}")


# ==================================================================================================
# POST-PROCESSING (After loop finishes or is interrupted)

print("\n===== LOSO Cross-Validation Complete =====")

# --- Calculate and Print Average Accuracy from Log File ---
try:
    results_df = pd.read_csv(accuracy_log_file)
    if not results_df.empty:
         average_accuracy = results_df['Accuracy'].mean()
         print(f"Average Accuracy across {len(results_df)} processed subjects: {average_accuracy * 100:.2f}%")
         # Optionally save summary stats
         summary_file = os.path.join(results_dir, f"{timestamp}_summary_LOSO.txt")
         with open(summary_file, "w") as f:
              f.write(f"Timestamp: {timestamp}\n")
              f.write(f"Model: {model_choice}\n")
              f.write(f"Data Type: {data_type} ({data_filename})\n")
              f.write(f"Number of Subjects Processed: {len(results_df)}\n")
              f.write(f"Average Loss: {results_df['Loss'].mean():.4f}\n")
              f.write(f"Average Accuracy: {average_accuracy * 100:.2f}%\n")
              f.write("\nAccumulated Confusion Matrix:\n")
              f.write(np.array2string(conf_matrix_accum, precision=2))
              f.write("\n\nAccuracy Log File: " + accuracy_log_file)
              f.write("\nPredictions File: " + predictions_file)
              f.write("\nMisclassifications File: " + misclassification_file)
              if shap_on:
                   f.write("\nSHAP Values Directory: " + shap_subject_dir)
         print(f"Summary saved to {summary_file}")
    else:
         print("Accuracy log file is empty. Cannot calculate average.")
except FileNotFoundError:
     print(f"Accuracy log file not found at {accuracy_log_file}. Cannot calculate average.")
except Exception as e:
     print(f"Error reading or processing accuracy log file: {e}")


# --- Save Accumulated Confusion Matrix ---
if np.sum(conf_matrix_accum) > 0: # Check if any processing was done
    try:
        plt.figure(figsize=(8, 6))
        # Normalize confusion matrix
        conf_matrix_norm = conf_matrix_accum.astype('float') / conf_matrix_accum.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm) # Handle potential NaNs if a row sum is 0

        sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(nb_classes), yticklabels=range(nb_classes))
        plt.title(f"Accumulated Normalized Confusion Matrix ({timestamp})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        conf_matrix_filename = os.path.join(results_dir, f"{timestamp}_confusion_matrix.png")
        plt.savefig(conf_matrix_filename)
        plt.close() # Close the plot to free memory
        print(f"Confusion matrix saved to {conf_matrix_filename}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
else:
    print("Accumulated confusion matrix is empty. Skipping plot.")


# --- Optional: Delete Progress Tracker on Full Completion ---
if start_subject_index == len(unique_subject_list): # Check if the loop completed fully
     print("LOSO process completed fully.")
     try:
          if os.path.exists(progress_file):
               os.remove(progress_file)
               print(f"Removed progress tracker file: {progress_file}")
     except Exception as e:
          print(f"Error removing progress tracker file: {e}")
     # Keep the progress file allows you to know the last run completed successfully.


# ==================================================================================================
# FINAL TRAINING ON ALL DATA (Optional - Run separately if needed)
# This section remains largely the same, but consider memory if dataset is huge.
# It should only run if the LOSO loop *fully completed*.

if start_subject_index == len(unique_subject_list): # Only run if LOSO finished
    print("\n===== Training Final Model on All Subjects... =====")
    clear_tf_memory() # Ensure clean state
    print_memory_usage()

    # Initialize Final Model
    if model_choice == 'ATCNet':
        final_model = ATCNet_(nb_classes, X.shape[1], X.shape[2])
    elif model_choice == 'EEGNet':
        final_model = EEGNet(nb_classes, X.shape[1], X.shape[2])
    else:
        print(f"Error: Unknown model_choice '{model_choice}' for final model.")
        final_model = None

    if final_model:
        # Prepare all data
        if model_choice == 'ATCNet':
            X_all = np.expand_dims(X, axis=1) # (batch, 1, channels, time)
        elif model_choice == 'EEGNet':
            X_all = np.expand_dims(X, axis=-1) # (batch, channels, time, 1) - Adjust if needed
        else:
             X_all = X

        y_all = to_categorical(y, nb_classes)
        print(f"Final training data shape: X_all: {X_all.shape}, y_all: {y_all.shape}")

        # Compile Final Model
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), # Use same LR or adjust
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train Final Model (Consider fewer epochs or different callbacks if needed)
        print("Fitting final model...")
        final_model.fit(X_all, y_all, batch_size=batch_size, epochs=epochs, verbose=1) # Use same epochs or adjust

        # Save Final Model Weights
        final_weights_path = os.path.join(saved_weights_dir, f"{timestamp}_FINAL_{model_choice}_{weight_name}.weights.h5")
        try:
            final_model.save_weights(final_weights_path)
            print(f"Final Model Weights Saved to {final_weights_path}")
        except Exception as e:
            print(f"Error saving final model weights: {e}")

        del final_model, X_all, y_all # Clean up
        clear_tf_memory()
        gc.collect()
else:
     print("\nLOSO process did not complete fully. Skipping final model training.")


print("\n===== Script Finished =====")