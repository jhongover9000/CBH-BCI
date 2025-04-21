import numpy as np

# Use the correct timestamp for the file you want to load
timestamp = "YYYYMMDDHHMMSS" # Replace with the actual timestamp
predictions_file = f"./results/{timestamp}_predictions_LOSO.npz"

try:
    loaded_data = np.load(predictions_file, allow_pickle=True)
    # The data is saved under the key 'predictions' and is a dictionary object
    all_predictions_loaded = loaded_data['predictions'].item()

    # Example: Access data for a specific subject (e.g., subject '5')
    subject_id = 5 # Or iterate through all_predictions_loaded.keys()
    if subject_id in all_predictions_loaded:
        subject_data = all_predictions_loaded[subject_id]
        print(f"Data for Subject {subject_id}:")

        # Access the confusion matrix
        subject_cm = subject_data['confusion_matrix']
        print("Confusion Matrix:")
        print(subject_cm)
        # subject_cm[0, 0] = True Negatives (Correct Rest)
        # subject_cm[1, 1] = True Positives (Correct MI)
        # subject_cm[0, 1] = False Positives (Rest classified as MI)
        # subject_cm[1, 0] = False Negatives (MI classified as Rest)

        # You can also access y_test and y_pred_prob if needed
        # print("True Labels (y_test):", subject_data['y_test'])
        # print("Predicted Probabilities (y_pred_prob):", subject_data['y_pred_prob'])

    else:
        print(f"Subject {subject_id} not found in the predictions file.")

except FileNotFoundError:
    print(f"Error: Predictions file not found at {predictions_file}")
except Exception as e:
    print(f"An error occurred while loading or processing the file: {e}")