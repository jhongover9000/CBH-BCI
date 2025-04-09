'''
BCI.py
--------
BCI Main Code Implementation

Description: Main code for the BCI system. Select emulation
or livestreaming mode. Reads data, preprocesses if needed,
and classifies according to use case.

Joseph Hong

'''

# =============================================================
# =============================================================
# Includes
from BCI_Functions import preprocess_raw
# from Classifiers import All_Models as classifiers
from models.atcnet_new import ATCNet_

import numpy as np
from datetime import datetime
import os
import argparse
import mne
import threading
import tkinter as tk

# =============================================================
# =============================================================
# Variables

# Command line arguments
is_virtual = False
preprocess_true = True
is_finished = False
is_verbose = False
is_lsl = False
is_broadcasting = False

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"
weight_filename = "ATCNet_Xon_NEW" #<>.weights.h5

# Emulator Variables
vhdr_name_loc = ""
raw_eeg_loc = ""
latency = ""

# Livestreamer Variables (Editable)
streamer_ip = "0.0.0.0"
streamer_port = 0

# Preprocessing Variables (Editable)
f_list = [7, 49]
new_freq = None

# Classifier Variables (Editable)
num_classes = 2
dropout_rate = 0.2
# Classifier Variables (Autoinitialized)
epoch_duration = 0

# Data Info Variables (Autoinitialized)
sfreq = 0
sampling_interval_us = 0
num_channels = 0
ch_names = []

# Main Window (Partially Editable) - Minumum 2 seconds OR 413 timepoints (FIR min)
window_size_s = 2
window_size_ms = window_size_s * 1000  # Milliseconds
window_size_us = window_size_ms * 1000  # Microseconds

# Receiver Variables (Editable)
seconds_to_read = 600

# OPERATING WINDOW - This is the number of seconds you want to use for classification
operating_window_size_s = 1
operating_window_size_ms = operating_window_size_s * 1000  # Milliseconds
operating_window_size_us = operating_window_size_s * 1000  # Microseconds
operating_overlap_percent = 50
overlap_percent = 50

# Receiver Variables (Autoinitialized)
overlap_timepoints = 0
seconds_ctr = 0

# Destination Variables (Editable)
com_port_num = 0
process_ip = "0.0.0.0"


# =============================================================
# =============================================================
# Functions

# Initialize BCI System Type
def initialize_bci(args):
    if(args.virtual):
        from receivers import virtual_receiver
        return virtual_receiver.Emulator()
    # special case, hardcoded, new receiver for CAIR demo
    elif(args.supernumerary and args.lsl):
        from receivers import cair_receiver
        return cair_receiver.CAIRReceiver()
    # elif (args.lsl):
    #     from receivers import lsl_receiver
    #     return lsl_receiver.LSLReceiver(broadcast = is_broadcasting)
    else:
        from receivers import livestream_receiver
        return livestream_receiver.LivestreamReceiver()
    
def start_plot(self):
        self.root = tk.Tk()
        self.root.title("BCI Real-time Data")

        # Add your GUI elements here (e.g., plots, labels)
        label = tk.Label(self.root, text="Real-time EEG Data")
        label.pack()

        # Start the Tkinter event loop - THIS IS ESSENTIAL
        self.root.mainloop()

# Function to run the GUI in a separate thread
def run_gui(bci_instance):
    bci_instance.start_plot()

# Print time and message
def logTime(message):
    if(is_verbose):
        print("===================================")
        print(message, datetime.now())
        print("")

# =============================================================
# =============================================================
# Execution
if __name__ == "__main__":

    # Grab arguments from command line
    parser = argparse.ArgumentParser(description="BCI System")

    # Add arguments
    parser.add_argument('--virtual', action='store_true',
                        help="Enable the virtual streaming for EEG data using an emulator")
    parser.add_argument('--preprocess', action='store_true', help="Enable data preprocessing")
    parser.add_argument('--verbose', action='store_true', help="Enable logging of times and processes")
    parser.add_argument('--lsl', action='store_true', help="Stream using LSL")
    parser.add_argument('--broadcast', action='store_true', help="Broadcast to other application")
    parser.add_argument('--nfreq', type=float, default=None, help="Set a new sampling frequency for EEG data (default: keep original)")
    parser.add_argument('--supernumerary', action='store_true', help="Predictions used are sent to supernumerary thumb")

    # Parse arguments, initialize variables
    args = parser.parse_args()
    is_virtual = args.virtual
    is_verbose = args.verbose
    is_lsl = args.lsl
    new_freq = args.nfreq
    if(new_freq == None):
        new_freq = 250
    is_broadcasting = args.broadcast
    preprocess_true = args.preprocess

    # Print all parsed arguments
    print("===================================")
    print("Parsed Command-Line Arguments:")
    print(f"  Virtual Mode:        {is_virtual}")
    print(f"  LSL Mode:        {is_lsl}")
    print(f"  Enable Preprocessing: {args.preprocess}")
    print(f"  Broadcasting: {args.broadcast}")
    print(f"  Verbose Mode:        {is_verbose}")
    print(f"  Sampling Frequency:  {new_freq if new_freq else 'Original'}")
    print("===================================")

    # Initialize BCI object
    bci = initialize_bci(args)

    # Start the GUI in a separate thread
    if hasattr(bci, 'start_plot') and callable(bci.start_plot):
        gui_thread = threading.Thread(target=run_gui, args=(bci,))
        gui_thread.daemon = True
        gui_thread.start()
    else:
        print("Warning: The 'bci' object does not have a 'start_plot' method to run the GUI.")

    # Initialize connection and variables
    logTime("Initializing Connection...")
    sfreq, ch_names, num_channels, data_buffer = bci.initialize_connection()
    # Calculate sampling interval, overlap timepoints, classification epoch
    sampling_interval_us = int((1 / sfreq) * 1000000)
    overlap_timepoints = int(operating_window_size_ms * (overlap_percent / 100) * (sfreq / 1000))
    print(overlap_timepoints)
    epoch_duration = int(window_size_s * sfreq)

    # If different frequency has been designated for downsampling (and classification)
    if (new_freq):
        epoch_duration = int(window_size_s * new_freq)

    print("===================================")
    print("Connection Initialized.")
    print("")
    # Notes: Initialize sampling interval via sfreq. Livestream receiver should receive 1 packet for initial details. Emulator receiver should get variables from info.

    # Set Up Classification Model - TBD
    logTime("Compiling Model...")
    model = ATCNet_(num_classes, num_channels, epoch_duration)
    model.load_weights(ref_weights_dir + f"{weight_filename}.weights.h5", skip_mismatch=True)
    logTime("Model Compilation Complete.")
    # Note: We may need to move this earlier for the livestream due to the possibility of the TCP buffer overflowing, but this would affect the epoch_duration definition.
    import asyncio
    import random
    first = True
    # Main Loop
    print("===================================")
    print("Starting to Receive.")
    print("")
    while not is_finished:
        # # Attempt to acquire data
        # try:
            # Acquire raw data packet
            data = bci.get_data()

            # If acquisition is successful, extend buffer
            if data is not None:
                # print(data)
                # Add to the data buffer
                data_buffer = np.concatenate((data_buffer, data), axis=1)

                # Check if buffer has reached window size, then process
                if len(data_buffer[0]) >= window_size_us // sampling_interval_us:
                    # Clear terminal
                    clear = lambda: os.system('cls')

                    logTime("Window Size Reached:")
                    # Preprocess Data (rawdata, frequencies for BPF, info for making mne array)
                    if preprocess_true:
                        data_block = preprocess_raw(data_buffer, f_list, bci.info, new_freq, is_lsl, is_verbose)
                        # If splitting, use command line arguments as parameters
                        # is_lsl, downsample <value>, etc.
                        # Note: Possibly split preprocessing pipeline to add/remove parts in command line arguments
                    else:
                        # NEED TO FORMAT THIS CONSIDERING THE MODEL
                        data_block = data_buffer
                    # Note: Copy the data_buffer (raw data) and preprocess copy if needed. If no preprocessing, simply return data as is.

                    # Perform Classification
                    logTime("Classification Started:")
                    probability = model.predict(data_block)
                    prediction = probability.argmax(axis=-1)
                    
                    # if (first):
                    #     prediction = 1
                    #     first = False

                    logTime(f"Classified {prediction}:")

                    # Perform Task Based on Implementation
                    logTime("BCI Task Started:")
                    bci.use_classification(prediction)
                    # Note: Sending serial port trigger (physical hardware) OR communicating with separate process (software)
                    logTime("BCI Task Finished:")

                    # Update data_buffer contents (remove old data) based on overlap
                    buffer_length = np.shape(data_buffer)[1]

                    # Check starting index - should be 0??
                    index = int(buffer_length - window_size_us // sampling_interval_us)

                    # For all channels, keep only latter portion of data
                    logTime("Data Buffer Update Started:")
                    # print((operating_window_size_s*sfreq) - overlap_timepoints)
                    data_buffer = data_buffer[:, int(operating_window_size_s*sfreq) - overlap_timepoints:]

                    # Clear Buffer if MI Predicted
                    if (prediction == 1):
                        data_buffer = np.zeros((num_channels, 0))
                    # print(data_buffer)
                    # buffer_length = np.shape(data_buffer)[1]
                    # print(buffer_length)
                    logTime("Data Buffer Updated:")

                    # Temporary Ending Condition, TBD
                    seconds_ctr += 1
                    if seconds_ctr == seconds_to_read:
                        is_finished = True
                        logTime(f"Finished {seconds_to_read} seconds:")
                # Buffer Size Check End
            # Acquisition Check End
        # Catch Errors in Acquisition
        # except Exception as loop_error:
        #     print(f"Error in main loop: {loop_error}")
        #     break
        # Try/Except End
    # Main Loop End

    # Disconnect Stream
    bci.disconnect()
    # Notes: Close connection
    print("===================================")
    print("Stream Disconnected Successfully.")

# Execution End