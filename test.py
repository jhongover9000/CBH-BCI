import pickle
import numpy as np

filepath = '/mnt/c/Users/Joseph/Documents/GitHub/CBH-BCI/results/20250422190952_consolidated_results.pkl' # Use your actual path

print(f"Loading file: {filepath}")
try:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print("File loaded successfully.")

    if not isinstance(data, dict) or not data:
         print("Error: Loaded data is not a non-empty dictionary.")
         exit()

    # Get a list of subject IDs from the loaded data
    subject_ids = list(data.keys())
    if not subject_ids:
         print("Error: No subject keys found in the dictionary.")
         exit()

    # Check the first subject (or choose a specific one)
    subject_to_check = subject_ids[0]
    print(f"\nChecking data structure for Subject ID: {subject_to_check}")

    if subject_to_check not in data:
        print(f"Error: Subject ID {subject_to_check} not found in data keys.")
        exit()

    subject_data = data[subject_to_check]

    if 'shap_values' not in subject_data:
        print("'shap_values' key is missing for this subject.")
    else:
        shap_val = subject_data['shap_values']
        print(f"Type of 'shap_values': {type(shap_val)}")

        if isinstance(shap_val, (list, tuple)):
            print(f"Length of list/tuple: {len(shap_val)}")
            for i, item in enumerate(shap_val):
                print(f"  Element {i} type: {type(item)}")
                if isinstance(item, np.ndarray):
                    print(f"  Element {i} shape: {item.shape}")
        elif isinstance(shap_val, np.ndarray):
            print(f"It's a single NumPy array with shape: {shap_val.shape}")
        elif shap_val is None:
            print("Value is None.")
        else:
            print(f"Value is of an unexpected type: {shap_val}")

except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
except Exception as e:
    print(f"An error occurred: {e}")