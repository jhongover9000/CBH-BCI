import time
import numpy as np
import mne

class Emulator:
    def __init__(self, fileName="MIT33"):
        self.vhdr_file = f"../RawData/{fileName}.vhdr"
        self.eeg_file = f"../RawData/{fileName}.eeg"
        self.channel_count = 0
        self.sampling_frequency = 0
        self.sampling_interval_us = 0
        self.channel_names = []
        self.latency = 0.016
        self.chunk_size = None
        self.current_index = 0
        self.raw_data = None
        self.buffer_size = None
        self.info = None

    def initialize_connection(self):
        self.raw_data = mne.io.read_raw_brainvision(self.vhdr_file, preload=True)
        self.sampling_frequency = int(self.raw_data.info["sfreq"])
        self.sampling_interval_us = (1 / self.sampling_frequency) * 1e6
        self.chunk_size = int(self.sampling_frequency / 50)
        self.buffer_size = int(self.sampling_frequency * 2)
        self.channel_names = self.raw_data.ch_names
        self.channel_count = len(self.channel_names)
        self.info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.sampling_frequency,
            ch_types="eeg"
        )
        print(f"Running EEG Emulator with {self.channel_count} channels at {self.sampling_frequency} Hz")
        print(f"Chunk size is set to: {self.chunk_size} per channel.")
        return self.sampling_frequency, self.channel_names, self.channel_count,  self.raw_data.get_data()

    def get_data(self):
        time.sleep(self.latency)
        total_samples = self.raw_data._data.shape[1]

        chunk_end = min(self.current_index + self.chunk_size, total_samples)
        data_chunk = self.raw_data._data[:, self.current_index:chunk_end]
        self.current_index = chunk_end

        return data_chunk

    def use_classification(self, prediction):
        if prediction == 0:
            print("Rest")
        elif prediction == 1:
            print("Flex")
        else:
            print("Extend")

    def disconnect(self):
        self.current_index = 0
        print("Disconnected from EEG emulator")
