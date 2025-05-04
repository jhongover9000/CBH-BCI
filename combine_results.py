"""
Consolidate LOSO Results Script

Description:
  Loads the predictions/CM NPZ file and individual subject SHAP pickle files
  from a completed or partially completed run of the incremental LOSO script
  (classifier_bn_incremental_persistent.py) and combines them into a single
  dictionary, saved as a pickle file for easier downstream analysis.

Usage:
  python consolidate_results.py <timestamp>

Example:
  python consolidate_results.py 20250423153000

Input Files (relative to script location):
  - results/<timestamp>_predictions_LOSO.npz
  - shap/<timestamp>_subject_shap_values/subject_<subject_id>_shap.pkl (for each subject)
  - (Optional) progress_tracker.json (to check run status)

Output File (relative to script location):
  - results/<timestamp>_consolidated_results.pkl
"""

import os
import numpy as np
import pickle
import argparse
import sys
import json

def load_predictions(predictions_filepath):
    """Loads the predictions dictionary from the NPZ file."""
    try:
        loaded_npz = np.load(predictions_filepath, allow_pickle=True)
        if 'predictions' in loaded_npz:
            # .item() retrieves the dictionary object stored in the array
            predictions_dict = loaded_npz['predictions'].item()
            print(f"Successfully loaded predictions for {len(predictions_dict)} subjects from {predictions_filepath}")
            return predictions_dict
        else:
            print(f"Error: 'predictions' key not found in {predictions_filepath}")
            return None
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_filepath}")
        return None
    except Exception as e:
        print(f"Error loading predictions NPZ file {predictions_filepath}: {e}")
        return None

def load_shap_values(shap_filepath):
    """Loads SHAP values from a pickle file."""
    try:
        with open(shap_filepath, 'rb') as f:
            shap_data = pickle.load(f)
        # print(f"  - Loaded SHAP values from {shap_filepath}") # Optional verbose log
        return shap_data
    except FileNotFoundError:
        print(f"  - Warning: SHAP file not found: {shap_filepath}")
        return None # Indicate missing SHAP data
    except pickle.UnpicklingError:
        print(f"  - Error: Could not unpickle SHAP file: {shap_filepath}")
        return None
    except Exception as e:
        print(f"  - Error loading SHAP file {shap_filepath}: {e}")
        return None

def check_run_status(progress_filepath, expected_subjects):
    """Checks the progress tracker to see if the run fully completed."""
    try:
        with open(progress_filepath, 'r') as f:
            progress_data = json.load(f)
        last_processed_index = progress_data.get('last_processed_index', -1)
        # Expected number of subjects is usually len(unique_subject_list)
        # We compare index (0-based) with count-1
        if last_processed_index >= expected_subjects - 1:
            print(f"Progress tracker indicates run fully completed ({last_processed_index+1}/{expected_subjects} subjects).")
            return True
        else:
            print(f"Warning: Progress tracker indicates run may be incomplete ({last_processed_index+1}/{expected_subjects} subjects processed).")
            return False
    except FileNotFoundError:
        print("Warning: Progress tracker file not found. Cannot verify run completion status.")
        return False
    except Exception as e:
        print(f"Warning: Error reading progress tracker {progress_filepath}: {e}")
        return False


def main(run_timestamp):
    """Main function to consolidate results."""
    print(f"--- Consolidating Results for Timestamp: {run_timestamp} ---")

    # --- Define Paths ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    shap_base_dir = os.path.join(base_dir, "shap")
    progress_file = os.path.join(base_dir, "progress_tracker.json") # To check completion

    predictions_filepath = os.path.join(results_dir, f"{run_timestamp}_predictions_LOSO.npz")
    shap_subject_dir = os.path.join(shap_base_dir, f"{run_timestamp}_subject_shap_values")
    output_filepath = os.path.join(results_dir, f"{run_timestamp}_consolidated_results.pkl")

    # --- Load Predictions ---
    predictions_data = load_predictions(predictions_filepath)
    if predictions_data is None:
        sys.exit(1) # Exit if predictions can't be loaded

    subjects = list(predictions_data.keys())
    if not subjects:
        print("Error: No subjects found in the predictions data.")
        sys.exit(1)

    print(f"Processing data for {len(subjects)} subjects found in predictions file.")

    # --- (Optional) Check Run Status ---
    # Note: This requires knowing the *expected* total number of subjects.
    # You might need to load the original data file again or pass it as an arg.
    # For simplicity, we'll just proceed but show how the check could work.
    # num_expected_subjects = 5 # Replace with actual expected count if known
    # check_run_status(progress_file, num_expected_subjects)


    # --- Consolidate Data ---
    consolidated_data = {}
    subjects_missing_shap = []

    for subject_id in subjects:
        print(f"Processing Subject: {subject_id}")
        subject_pred_data = predictions_data[subject_id]

        # Initialize entry for this subject in the output dictionary
        consolidated_data[subject_id] = {
            'y_test': subject_pred_data.get('y_test', None),
            'y_pred_prob': subject_pred_data.get('y_pred_prob', None),
            'confusion_matrix': subject_pred_data.get('confusion_matrix', None),
            'shap_values': None # Placeholder for SHAP
        }

        # Construct path to this subject's SHAP file
        # Assumes timestamp is part of the directory name, not the individual file name
        shap_filepath = os.path.join(shap_subject_dir, f"subject_{subject_id}_shap.pkl")

        # Load SHAP values for this subject
        shap_values = load_shap_values(shap_filepath)

        if shap_values is not None:
            consolidated_data[subject_id]['shap_values'] = shap_values
        else:
            # Keep track of subjects whose SHAP files were missing or failed to load
            subjects_missing_shap.append(subject_id)
            print(f"  -> Proceeding without SHAP data for subject {subject_id}.")


    # --- Report Missing SHAP Files ---
    if subjects_missing_shap:
        print("\n--- Summary of Missing SHAP Files ---")
        print(f"Could not load SHAP data for the following {len(subjects_missing_shap)} subjects:")
        print(subjects_missing_shap)
        print("Their 'shap_values' entry in the output file will be None.")
    else:
        print("\nSuccessfully loaded SHAP data for all processed subjects.")


    # --- Save Consolidated Data ---
    print(f"\nSaving consolidated data to: {output_filepath}")
    try:
        with open(output_filepath, 'wb') as f:
            pickle.dump(consolidated_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Consolidated data saved successfully.")
    except Exception as e:
        print(f"Error saving consolidated data to {output_filepath}: {e}")
        sys.exit(1)

    print("\n--- Consolidation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate Incremental LOSO Results (Predictions & SHAP Values)")
    parser.add_argument("timestamp",
                        help="The timestamp string (YYYYMMDDHHMMSS) of the run to consolidate.")

    # Ensure timestamp argument is provided
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # Handle potential -h or --help arguments gracefully before assuming argv[1] is the timestamp
    if sys.argv[1] in ['-h', '--help']:
         parser.print_help(sys.stderr)
         sys.exit(0)

    # Simple check for timestamp format (basic length check)
    timestamp_arg = sys.argv[1]
    if not (timestamp_arg.isdigit() and len(timestamp_arg) == 14):
         print(f"Error: Timestamp format looks incorrect. Expected 14 digits (YYYYMMDDHHMMSS), got '{timestamp_arg}'")
         # Still try to run, maybe it's a different format the user intended
         # sys.exit(1) # Uncomment to enforce format check strictly

    # Call main function (argparse handles help automatically now)
    args = parser.parse_args()
    main(args.timestamp)