'''
CBH_DEMO

Description: Script for CBH Lab Tour BMI Demo. Limited functionality, hardcoded.
Both datasets should be at 200Hz, so 400 timepoints for each trial.

Joseph Hong

'''

# =============================================================
# =============================================================
# INCLUDES
import numpy as np
import argparse
import random
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os

# =============================================================
# =============================================================
# VARIABLES

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"

# Datasets
DATASET_SUPERNUMERARY = "subject_data_st.npz"
DATASET_NATURAL = "subject_data_nt.npz"

# Weights
weight_filename_st = "ATCNet_ST"
weight_filename_nt = "ATCNet_NT"

num_classes = 2

num_channels_st = 60
num_timepoints_st = 400

num_channels_nt = 19
num_timepoints_nt = 200

# Ports (Edit!)
com_port = "/dev/ttyUSB0"
com_baudrate = 115200
tcp_ip = "127.0.0.1"
tcp_port = 5005

# Command Line Arguments
is_verbose = False

# =============================================================
# =============================================================
# FUNCTIONS

# Print time and message
def logTime(message):
    if(is_verbose):
        print("===================================")
        print(message, datetime.now())
        print("")

# =============================================================
# =============================================================
# EXECUTION

# Argument parsing
parser = argparse.ArgumentParser(description="BCI System")
parser.add_argument('--supernumerary', action='store_true', help="BMI system for supernumerary effector MI")
parser.add_argument('--natural', action='store_true', help="BMI system for natural effector MI")
parser.add_argument('--verbose', action='store_true', help="Enable logging of times and processes")
args = parser.parse_args()

# Check if not either ST or NT
if not (args.supernumerary or args.natural):
    parser.error("Either --supernumerary or --natural must be specified.")

# Load Model
from models.atcnet_new import ATCNet_
if args.supernumerary:
    # Load Model for ST
    model = ATCNet_(num_classes, num_channels_st, num_timepoints_st)
    model.load_weights(f"{ref_weights_dir}{weight_filename_st}.weights.h5", skip_mismatch=True)

    # Assign EEG Dataset
    dataset_path = f"{data_dir}{DATASET_SUPERNUMERARY}"

    from comm_controller import COMPortSignalSender
    bci = COMPortSignalSender(com_port, com_baudrate)

elif args.natural:
    # Load Model for NT
    model = ATCNet_(num_classes, num_channels_nt, num_timepoints_nt)
    model.load_weights(f"{ref_weights_dir}{weight_filename_nt}.weights.h5", skip_mismatch=True)

    # Assign EEG Dataset
    dataset_path = f"{data_dir}{DATASET_NATURAL}"

    from broadcasting import TCP_Server
    bci = TCP_Server.TCPServer(tcp_ip, tcp_port)

else:

    raise ValueError("Please specify either --supernumerary or --natural.")

logTime("Model Compilation Complete.")

# Load dataset
print(f"Loading dataset from {dataset_path}...")
data = np.load(dataset_path)
X, y, subject_ids = data['X'], data['y'], data['subject_ids']
X_all = np.expand_dims(X, axis=1)  # (batch, 1, channels, time)

# Initialize Connection
bci.initialize_connection()

# GUI for user input
def select_label(label):
    """ Select trials based on user choice of MI or Rest """

    clear = lambda: os.system('cls')

    trials = np.where(y == label)[0]
    if len(trials) == 0:
        messagebox.showerror("Error", "No trials found for the selected label.")
        return
    
    # Randomly select a subject and a trial
    selected_trial = random.choice(trials)
    selected_subject = subject_ids[selected_trial]
    print(f"Selected Subject: {selected_subject}, Trial: {selected_trial}, Label: {label}")
    
    # Get EEG data for the selected trial
    eeg_data = X_all[selected_trial]
    
    # Perform classification
    probability = model.predict(eeg_data[np.newaxis, :, :])
    prediction = probability.argmax(axis=-1)
    print(f"Classification Result: {prediction}")
    
    # Use classification
    bci.use_classification(prediction)

    # Update GUI labels
    subject_label.config(text=f"Subject: {selected_subject}")
    trial_label.config(text=f"Trial: {selected_trial}")
    true_label.config(text=f"True Label: {label}")
    predicted_label.config(text=f"Predicted Label: {prediction}")

# Create GUI
root = tk.Tk()
root.title("Select Motor Imagery State")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#ffffff", padx=20, pady=20, relief=tk.RIDGE, borderwidth=5)
frame.pack(pady=20)

tk.Label(frame, text="Select Motor Imagery State", font=("Arial", 14, "bold"), bg="#ffffff").pack(pady=10)

mi_button = tk.Button(frame, text="Motor Imagery", font=("Arial", 12), bg="#4CAF50", fg="white", width=15, height=2,
                      command=lambda: select_label(1))
mi_button.pack(pady=5)

rest_button = tk.Button(frame, text="Rest", font=("Arial", 12), bg="#008CBA", fg="white", width=15, height=2,
                         command=lambda: select_label(0))
rest_button.pack(pady=5)

# Labels to display classification results
subject_label = tk.Label(frame, text="Subject: ", font=("Arial", 12), bg="#ffffff")
subject_label.pack()
trial_label = tk.Label(frame, text="Trial: ", font=("Arial", 12), bg="#ffffff")
trial_label.pack()
true_label = tk.Label(frame, text="True Label: ", font=("Arial", 12), bg="#ffffff")
true_label.pack()
predicted_label = tk.Label(frame, text="Predicted Label: ", font=("Arial", 12), bg="#ffffff")
predicted_label.pack()

root.mainloop()

bci.disconnect()