"""
LSL RECEIVER

Description:
- Processes .mat files where each row represents a different subject.
- Each subject has Rest and MI trials in separate columns.
- Applies bandpass filtering, referencing, and channel selection.
- Saves the processed data in .npz format for later use in classification.

Joseph Hong

"""
# =============================================================
# =============================================================
# INCLUDES

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import mne
from datetime import datetime

from collections import Counter

from broadcasting import TCP_Server_Offloaded
import tkinter as tk
import threading
import time
import asyncio
from comm_controller import COMPortSignalSender

# Ports (Edit - Need to check through cmd)
com_port = "/dev/ttyS0"
com_baudrate = 115200

#==========================================================================================
#==========================================================================================
# HELPER FUNCTIONS

# Check for duplicate channel names and make them unique
def make_channel_names_unique(channel_names):
    counts = Counter(channel_names)
    seen = {}
    unique_names = []

    for ch in channel_names:
        if counts[ch] > 1:
            seen[ch] = seen.get(ch, -1) + 1
            unique_names.append(f"{ch}_{seen[ch]}")
        else:
            unique_names.append(ch)

    return unique_names

#==========================================================================================
#==========================================================================================
# CLASS DEFINITION

class CAIRReceiver:
    def __init__(self, stream_type="EEG", broadcast = False, com_port_finger = "/dev/ttyUSB0",
                com_port_feedback = "/dev/ttyACM0", start_gui_callback=None, bci_instance=None):
        # Check for EEG LSL Stream
        print("Looking for an EEG stream...")
        streams = resolve_byprop('type', stream_type)

        if not streams:
            raise RuntimeError("No EEG stream found. Ensure the LSL stream is active.")

        self.inlet = StreamInlet(streams[0])
        self.channel_names = []
        self.sampling_frequency = 0
        self.channel_count = 0
        self.data = None
        self.broadcasting = broadcast
        self.server = None
        self.start_gui_callback = start_gui_callback
        self.bci_instance = bci_instance
        self.finger_moving = False
        self.port_finger = COMPortSignalSender(com_port_finger,115200)
        self.port_feedback = COMPortSignalSender(com_port_feedback,115200)

        print(f"Connected to EEG stream: {streams[0].name()}")
        self.initialize_connection()

        # If broadcasting classification (for further applications), set up UDP server
        if broadcast:
            self.server = TCP_Server_Offloaded.TCPServer()
            self.server.initialize_connection()

        # Start GUI if a callback and bci instance are provided
        if self.start_gui_callback and self.bci_instance:
            gui_thread = threading.Thread(target=self._run_gui_wrapper)
            gui_thread.daemon = True  # Allow the main program to exit even if the thread is still running
            gui_thread.start()

    def finger_flex(self):
        # If finger is not moving, flex finger followed by extension
        if(not self.finger_moving):
            self.finger_moving = True
            self.port_finger.send_signal("v100")
            self.port_feedback.send_signal("v100")
            asyncio.sleep(2)
            self.port_finger.send_signal("v0")
            self.port_feedback.send_signal("v0")
            asyncio.sleep(2)
            self.finger_moving = False

    def _run_gui_wrapper(self):
        self.start_gui_callback(self.bci_instance)

    # Initialize LSL connection
    def initialize_connection(self):
        # Initialize COM Ports
        self.port_finger.initialize_connection()
        self.port_feedback.initialize_connection()

        # Retrieve LSL stream info
        info = self.inlet.info()
        self.channel_count = info.channel_count()
        self.sampling_frequency = info.nominal_srate()
        self.channel_names = [info.desc().child("channels").child("channel").child_value("label")
                              for _ in range(self.channel_count)]

        # Extract channel names from the LSL stream
        ch_list = info.desc().child("channels").first_child()
        channel_names = []
        while ch_list.name() == "channel":
            channel_names.append(ch_list.child_value("label"))
            ch_list = ch_list.next_sibling()

        # Ensure unique channel names (needed for making MNE array)
        self.channel_names = make_channel_names_unique(channel_names)

        # Identify EEG channels (exclude accelerometer or other non-EEG sensors)
        eeg_channels = [ch for ch in self.channel_names if not ch.startswith("acc")]
        # Note: if accel info can be used later on, then send elsewhere?

        # Store non-EEG channels separately
        self.non_eeg_channels = [ch for ch in self.channel_names if ch.startswith("acc")]

        # Filter channel count to only EEG channels (for inference)
        self.channel_count = len(eeg_channels)

        # Create MNE info object with only EEG channels
        ch_types = ["eeg"] * self.channel_count
        self.info = mne.create_info(eeg_channels, ch_types=ch_types, sfreq=self.sampling_frequency)

        # Apply montage only to EEG channels
        self.info.set_montage("standard_1020", on_missing="ignore")

        print(f"EEG Channels: {eeg_channels}")
        print(f"Non-EEG Channels (Stored Separately): {self.non_eeg_channels}")

        # Initialize data buffer
        self.data = np.zeros((self.channel_count, 0))
        print(f"Channels: {self.channel_count}, Sampling Frequency: {self.sampling_frequency} Hz")
        print(f"Channel Names: {self.channel_names}")
        print("===================================")
        print("Initial Data Shape:", np.shape(self.data))

        return self.sampling_frequency, self.channel_names, self.channel_count, self.data
    
    def start_plot(self):
        self.root = tk.Tk()
        self.root.title("BCI Real-time Data")

        # GUI WILL GO HERE
        label = tk.Label(self.root, text="Real-time EEG Data")
        label.pack()

        # Start the Tkinter event loop - THIS IS ESSENTIAL
        self.root.mainloop()

    # Get data from LSL stream if there is anything
    def get_data(self):
        sample, timestamp = self.inlet.pull_sample()

        if sample:
            sample = np.array(sample).reshape((len(self.channel_names), 1))  # Reshape for channel-wise stacking

            # Extract EEG data by filtering out non-EEG channels
            eeg_data = np.array([sample[i] for i, ch in enumerate(self.channel_names) if not ch.startswith("acc")])

            # Optional: Store non-EEG (accelerometer) data separately
            self.accelerometer_data = {ch: sample[i] for i, ch in enumerate(self.channel_names) if ch.startswith("acc")}

            # print(eeg_data)

            return eeg_data  # Return only EEG data

        return None  # No data received

    # Use for classification goes here, depending on what you're using
    def use_classification(self, prediction):
        print(prediction,  datetime.now())
        # Use classification
        if prediction == 0:
            print("Rest")
        elif prediction == 1:
            self.finger_flex()
            print("MI")
        else:
            print("??")

    def disconnect(self):
        # Close connection
        print("Connection closed.")