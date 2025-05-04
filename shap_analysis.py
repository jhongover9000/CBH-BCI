"""
SHAP Electrode Importance Analysis with Channel Names

Description:
  Loads consolidated results (including SHAP values) from a specific LOSO run,
  incorporates electrode names from a .mat reference file, and calculates the
  overall importance of each electrode/channel based on mean absolute SHAP values
  aggregated across time, trials, and subjects.

Usage:
  python analyze_shap_importance.py <timestamp> [--target_class <int>] [--top_n <int>] [--mat_var <var_name>]

Example:
  python analyze_shap_importance.py 20250423153000 --target_class 1 --top_n 10 --mat_var channel_names

Input Files (relative to script location):
  - results/<timestamp>_consolidated_results.pkl
  - reference/channels_60.mat (or specified .mat file)

Output:
  - Prints the top N important electrodes with names (if available).
  - Saves a bar plot: results/<timestamp>_electrode_importance.png (with names)
  - Saves numerical scores: results/<timestamp>_electrode_importance_scores.csv (with names)
"""

import os
import numpy as np
import pickle
import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io # <-- Import SciPy

# --- Configuration ---
# Axes for SHAP array aggregation (adjust if needed)
# Based on SHAP output shape like: (trials, 1, channels, timepoints, classes)
channel_axis = 2
time_axis = 3
# Reference file info
ref_dir = "reference"
ref_mat_filename = "channels_60.mat"
# Default variable name inside .mat file holding channel names
default_mat_variable_name = "channel_names"
# --------------------


def load_consolidated_data(filepath):
    """Loads the consolidated data dictionary from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded consolidated data from {filepath}")
        if not isinstance(data, dict):
            print("Error: Loaded data is not a dictionary.")
            return None
        return data
    except FileNotFoundError:
        print(f"Error: Consolidated results file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading consolidated data from {filepath}: {e}")
        return None

def load_channel_names_from_mat(base_dir, mat_var_name):
    """Loads channel names from the reference .mat file."""
    mat_filepath = os.path.join(base_dir, ref_dir, ref_mat_filename)
    print(f"\nLoading channel names from: {mat_filepath}")
    try:
        mat_data = scipy.io.loadmat(mat_filepath)
        if mat_var_name not in mat_data:
            print(f"Error: Variable '{mat_var_name}' not found in {mat_filepath}.")
            print(f"Available variables: {list(mat_data.keys())}")
            return None

        channel_names_raw = mat_data[mat_var_name]
        # Attempt to extract strings, handling common .mat structures (like cell arrays)
        try:
            # Common case: cell array loaded as nested object array
            electrode_names = [str(arr[0]) for arr in channel_names_raw.flatten()]
        except (TypeError, IndexError):
            try:
                # Simpler case: maybe already an array of strings or objects convertible to string
                 electrode_names = [str(name) for name in channel_names_raw.flatten()]
            except Exception as e_inner:
                 print(f"Warning: Could not automatically extract string names from '{mat_var_name}'. Error: {e_inner}")
                 return None

        print(f"Successfully loaded {len(electrode_names)} channel names.")
        return electrode_names

    except FileNotFoundError:
        print(f"Error: Channel reference file not found at {mat_filepath}")
        return None
    except Exception as e:
        print(f"Error loading .mat file {mat_filepath}: {e}")
        return None


# --- calculate_electrode_importance function remains the same as the previous version ---
# (It correctly returns overall_channel_importance, num_channels or None)
def calculate_electrode_importance(consolidated_data, target_class_index):
    """
    Calculates electrode importance by aggregating SHAP values.
    Handles SHAP values stored as a list of arrays or a single array
    with classes in the last dimension.

    Returns:
        tuple(np.ndarray, int): (Array of importance scores, num_channels)
                                 Returns None if calculation fails.
    """
    subject_importance_list = []
    num_channels = None
    processed_subjects_count = 0
    # Infer number of classes from the first valid SHAP array if possible
    nb_classes_inferred = None

    print(f"\nCalculating importance based on SHAP values for class {target_class_index}...")

    for subject_id, subject_data in consolidated_data.items():
        shap_values = subject_data.get('shap_values')

        if shap_values is None:
            print(f" - Skipping Subject {subject_id}: 'shap_values' key is None or missing.")
            continue

        shap_array_for_class = None # Initialize target array for this subject/class

        # --- Check SHAP data format ---
        if isinstance(shap_values, (list, tuple)):
            # Case 1: Data is a list/tuple (Original expectation)
            if len(shap_values) > target_class_index:
                element = shap_values[target_class_index]
                if isinstance(element, np.ndarray):
                    if element.ndim == 4: # Expecting shape like (trials, ?, chans, time)
                        shap_array_for_class = element
                        if nb_classes_inferred is None: nb_classes_inferred = len(shap_values)
                    else:
                        print(f" - Skipping Subject {subject_id}: SHAP list element has unexpected ndim ({element.ndim}). Expected 4.")
                        continue
                else:
                    print(f" - Skipping Subject {subject_id}: SHAP list element at index {target_class_index} not NumPy array (Type: {type(element)}).")
                    continue
            else:
                 print(f" - Skipping Subject {subject_id}: SHAP list length {len(shap_values)} too short for target index {target_class_index}.")
                 continue

        elif isinstance(shap_values, np.ndarray):
            # Case 2: Data is a single NumPy array (As found in inspection)
            # Expecting shape like (trials, ?, chans, time, classes) = 5 dims
            # print(f" - Subject {subject_id}: Processing single SHAP array (shape: {shap_values.shape})") # Verbose
            expected_ndim_single_array = 5
            if shap_values.ndim == expected_ndim_single_array:
                num_classes_in_array = shap_values.shape[-1]
                if nb_classes_inferred is None: nb_classes_inferred = num_classes_in_array
                elif nb_classes_inferred != num_classes_in_array:
                     print(f"Error: Inconsistent number of classes found in SHAP arrays! Expected {nb_classes_inferred}, found {num_classes_in_array} for Subject {subject_id}. Aborting.")
                     return None

                if num_classes_in_array > target_class_index:
                    shap_array_for_class = shap_values[..., target_class_index]
                    # print(f"   - Extracted SHAP data for class {target_class_index} (shape: {shap_array_for_class.shape})") # Verbose
                else:
                    print(f" - Skipping Subject {subject_id}: Target class index {target_class_index} is out of bounds for last dimension size {num_classes_in_array}.")
                    continue
            else:
                 print(f" - Skipping Subject {subject_id}: Single SHAP array has unexpected ndim ({shap_values.ndim}). Expected {expected_ndim_single_array}.")
                 continue
        else:
            # Case 3: Data is of an unexpected type
            print(f" - Skipping Subject {subject_id}: SHAP data is not list/tuple or NumPy array (Type: {type(shap_values)}).")
            continue
        # --- End format check ---


        if shap_array_for_class is None:
            print(f" - Skipping Subject {subject_id}: Failed to extract valid SHAP numpy array for class {target_class_index} after checks.")
            continue

        # --- Validate SHAP array dimensions (should be 4D now) ---
        expected_ndim_after_selection = 4
        if shap_array_for_class.ndim != expected_ndim_after_selection:
             print(f" - Skipping Subject {subject_id}: Extracted SHAP array for class {target_class_index} has unexpected ndim ({shap_array_for_class.ndim}). Expected {expected_ndim_after_selection}.")
             continue

        # Determine number of channels (use the pre-defined channel_axis = 2)
        try:
            current_num_channels = shap_array_for_class.shape[channel_axis]
        except IndexError:
             print(f" - Skipping Subject {subject_id}: Invalid channel_axis ({channel_axis}) for extracted SHAP array shape {shap_array_for_class.shape}.")
             continue

        if num_channels is None:
            num_channels = current_num_channels
            # print(f"Determined number of channels: {num_channels} (from Subject {subject_id})") # Verbose
        elif num_channels != current_num_channels:
            print(f"Error: Inconsistent number of channels found! Expected {num_channels}, found {current_num_channels} for Subject {subject_id}. Aborting.")
            return None

        # Calculate mean absolute SHAP value across the time dimension (axis=3)
        try:
            mean_abs_shap_over_time = np.mean(np.abs(shap_array_for_class), axis=time_axis)
        except np.AxisError:
             print(f" - Skipping Subject {subject_id}: Invalid time_axis ({time_axis}) for extracted SHAP array shape {shap_array_for_class.shape}.")
             continue
        except Exception as e:
             print(f" - Skipping Subject {subject_id}: Error during time aggregation: {e}")
             continue

        # Calculate mean absolute SHAP value across trials (axis=0) for this subject
        try:
             subject_mean_abs_shap_per_channel = np.mean(mean_abs_shap_over_time, axis=0)
        except Exception as e:
             print(f" - Skipping Subject {subject_id}: Error during trial aggregation: {e}")
             continue

        # Squeeze result and ensure it's a 1D array of channel importances
        subject_mean_abs_shap_per_channel = np.squeeze(subject_mean_abs_shap_per_channel)
        if subject_mean_abs_shap_per_channel.ndim == 0:
             subject_mean_abs_shap_per_channel = np.array([subject_mean_abs_shap_per_channel.item()])
        elif subject_mean_abs_shap_per_channel.ndim != 1 or len(subject_mean_abs_shap_per_channel) != num_channels:
             print(f" - Skipping Subject {subject_id}: Unexpected shape after trial aggregation and squeeze ({subject_mean_abs_shap_per_channel.shape}). Expected ({num_channels},).")
             continue

        subject_importance_list.append(subject_mean_abs_shap_per_channel)
        processed_subjects_count += 1

    if not subject_importance_list:
        print("\nError: No valid SHAP data could be processed for any subject after checks. Cannot calculate overall importance.")
        return None

    print(f"\nAggregating importance across {processed_subjects_count} successfully processed subjects...")
    try:
        overall_channel_importance = np.mean(np.array(subject_importance_list), axis=0)
    except Exception as e:
         print(f"Error during final aggregation across subjects: {e}")
         return None

    if overall_channel_importance.shape != (num_channels,):
         print(f"Error: Final aggregated importance has unexpected shape {overall_channel_importance.shape}. Expected ({num_channels},).")
         return None

    print("Aggregation complete.")
    return overall_channel_importance, num_channels


# --- Modify plot_importance to accept names ---
def plot_importance(scores, num_channels, channel_names, timestamp, output_dir):
    """
    Generates and saves a bar plot of electrode importance.
    Uses channel names if provided and valid.
    """
    if scores is None or num_channels is None:
        print("Skipping plotting due to missing scores or channel count.")
        return

    output_filename = os.path.join(output_dir, f"{timestamp}_electrode_importance.png")
    print(f"\nGenerating importance plot: {output_filename}")

    # Determine labels: Use names if valid, otherwise use indices
    use_names = False
    if channel_names is not None and len(channel_names) == num_channels:
        channel_labels = channel_names
        x_label = "Electrode Name"
        use_names = True
        print("Using electrode names for plot labels.")
    else:
        if channel_names is not None: # Names loaded but didn't match count
             print(f"Warning: Number of electrode names ({len(channel_names)}) does not match number of channels ({num_channels}). Using indices for plot labels.")
        channel_labels = [str(i) for i in range(num_channels)]
        x_label = "Electrode / Channel Index"

    try:
        channel_indices = np.arange(num_channels)
        plt.figure(figsize=(max(12, num_channels * 0.4), 7)) # Adjusted size
        plt.bar(channel_indices, scores) # Plot against numerical index
        plt.xticks(ticks=channel_indices, labels=channel_labels, rotation=90, fontsize=8) # Set labels
        plt.xlabel(x_label)
        plt.ylabel("Mean Absolute SHAP Value (Aggregated)")
        plt.title(f"Overall Electrode Importance (Run: {timestamp})")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error generating plot: {e}")

# --- Modify save_scores to accept names ---
def save_scores(scores, num_channels, channel_names, timestamp, output_dir):
     """Saves the numerical importance scores to a CSV file, including names if available."""
     if scores is None or num_channels is None:
        print("Skipping saving scores due to missing data.")
        return

     output_filename = os.path.join(output_dir, f"{timestamp}_electrode_importance_scores.csv")
     print(f"Saving numerical importance scores: {output_filename}")

     try:
          channel_indices = np.arange(num_channels)
          data_dict = {
               'channel_index': channel_indices,
               'mean_abs_shap': scores
          }
          # Add names column if valid
          if channel_names is not None and len(channel_names) == num_channels:
               data_dict['channel_name'] = channel_names
               print("Including electrode names in CSV file.")
               col_order = ['channel_name', 'channel_index', 'mean_abs_shap'] # Preferred order
          else:
               print("Warning: Electrode names not available or count mismatch. Saving scores with indices only.")
               col_order = ['channel_index', 'mean_abs_shap']


          df = pd.DataFrame(data_dict)
          df = df.sort_values(by='mean_abs_shap', ascending=False)
          df = df[col_order] # Reorder columns
          df.to_csv(output_filename, index=False)
          print("Scores saved successfully.")
     except Exception as e:
          print(f"Error saving scores: {e}")


def main(run_timestamp, target_class_index, top_n, mat_var_name):
    """Main function to perform SHAP analysis."""
    print(f"--- SHAP Electrode Importance Analysis for Timestamp: {run_timestamp} ---")

    # --- Define Paths ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    consolidated_filepath = os.path.join(results_dir, f"{run_timestamp}_consolidated_results.pkl")

    # Ensure results directory exists for outputs
    os.makedirs(results_dir, exist_ok=True)

    # --- Load Channel Names ---
    electrode_names = load_channel_names_from_mat(base_dir, mat_var_name)
    # Proceed even if names are None, functions will handle fallback

    # --- Load Consolidated SHAP/Prediction Data ---
    consolidated_data = load_consolidated_data(consolidated_filepath)
    if consolidated_data is None:
        sys.exit(1)

    # --- Calculate Importance ---
    result = calculate_electrode_importance(consolidated_data, target_class_index)

    if result is not None:
        electrode_scores, num_channels = result # Unpack only if result is not None

        # --- Validate names count against channel count from SHAP ---
        valid_names = False
        if electrode_names is not None:
            if len(electrode_names) == num_channels:
                print(f"Number of channels from SHAP ({num_channels}) matches number of names loaded ({len(electrode_names)}).")
                valid_names = True
            else:
                print(f"Warning: Number of loaded electrode names ({len(electrode_names)}) does not match number of channels found in SHAP data ({num_channels}). Will use indices only.")
                electrode_names = None # Discard invalid names


        # --- Print Top N ---
        sorted_indices = np.argsort(electrode_scores)[::-1]
        print(f"\n--- Top {min(top_n, num_channels)} Most Important Electrodes (Class {target_class_index}) ---")
        for i in range(min(top_n, num_channels)):
            idx = sorted_indices[i]
            name_str = f" ({electrode_names[idx]})" if valid_names else "" # Add name if available
            print(f"Rank {i+1}: Idx {idx}{name_str}, Score: {electrode_scores[idx]:.6f}")


        # --- Plotting (Pass potentially None names) ---
        plot_importance(electrode_scores, num_channels, electrode_names, run_timestamp, results_dir)

        # --- Save Scores (Pass potentially None names) ---
        save_scores(electrode_scores, num_channels, electrode_names, run_timestamp, results_dir)

    else:
        # Handle the case where calculation failed
        print("\nAnalysis could not be completed because electrode importance calculation failed.")
        sys.exit(1)

    print("\n--- Analysis Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Electrode Importance using Consolidated SHAP Values.")
    parser.add_argument("timestamp",
                        help="The timestamp string (YYYYMMDDHHMMSS) of the run to analyze.")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Index of the target output class for SHAP analysis (default: 1).")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top important electrodes to print (default: 10).")
    parser.add_argument("--mat_var_name", type=str, default="channels",
                        help="MATLAB Variable for channels.")

    args = parser.parse_args()

    # Basic validation
    if not (args.timestamp.isdigit() and len(args.timestamp) == 14):
         print(f"Warning: Timestamp format looks incorrect. Expected 14 digits (YYYYMMDDHHMMSS), got '{args.timestamp}'")
         # Proceed anyway

    if args.target_class < 0:
        print("Error: --target_class must be a non-negative integer.")
        sys.exit(1)

    if args.top_n < 1:
        print("Error: --top_n must be a positive integer.")
        sys.exit(1)

    main(args.timestamp, args.target_class, args.top_n, args.mat_var_name)