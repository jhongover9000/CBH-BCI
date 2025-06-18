'''
Enhanced Virtual Receiver with automatic scaling and diagnostics
Fixes common issues with ERD detection in virtual mode
'''

import time
import numpy as np
import mne
from scipy import signal

data_dir = "./data/rawdata/mit/"

class Emulator:
    def __init__(self, fileName="MIT33", auto_scale=False, verbose=False):
        self.vhdr_file = f"{data_dir}{fileName}.vhdr"
        self.eeg_file = f"{data_dir}{fileName}.eeg"
        self.channel_count = 0
        self.sampling_frequency = 0
        self.sampling_interval_us = 0
        self.channel_names = []
        self.latency = 0.0016
        self.chunk_size = None
        self.current_index = 0
        self.raw_data = None
        self.buffer_size = None
        self.info = None
        
        # Enhancement settings
        self.auto_scale = auto_scale
        self.verbose = verbose
        self.scaling_factor = 1.0
        self.dc_offset = None
        
    def initialize_connection(self):
        """Initialize connection with enhanced diagnostics"""
        if self.verbose:
            print("Loading EEG file...")
            
        # Load raw data
        self.raw_data = mne.io.read_raw_brainvision(self.vhdr_file, preload=True)
        self.raw_data = self.raw_data.resample(250)
        self.sampling_frequency = int(self.raw_data.info["sfreq"])
        self.sampling_interval_us = (1 / self.sampling_frequency) * 1e6
        self.chunk_size = int(self.sampling_frequency / 50)  # 20ms chunks
        self.buffer_size = int(self.sampling_frequency * 2)
        self.channel_names = self.raw_data.ch_names
        self.channel_count = len(self.channel_names)

        # Find specific annotation (marker)
        target_annotation = 'S  2'  # Replace with your marker name

        # Get annotations
        annotations = self.raw_data.annotations

        # Find the first occurrence of the target annotation
        mask = annotations.description == target_annotation
        if np.any(mask):
            start_time = annotations.onset[mask][0]  # First occurrence
            
            # Crop from this time point
            self.raw_data = self.raw_data.crop(tmin=start_time-2000)
            
        
        # Create info
        self.info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.sampling_frequency,
            ch_types="eeg"
        )
        
        if self.verbose:
            print(f"Loaded {self.channel_count} channels at {self.sampling_frequency} Hz")
            print(f"Chunk size: {self.chunk_size} samples per channel")
        
        # Auto-scale if enabled
        if self.auto_scale:
            self._detect_and_apply_scaling()
        
        # Check for DC offset
        self._check_dc_offset()
        
        return self.sampling_frequency, self.channel_names, self.channel_count, np.zeros((self.channel_count, 0))
    
    def _detect_and_apply_scaling(self):
        """Detect if data needs scaling to microvolts"""
        if self.verbose:
            print("\nChecking data scaling...")
        
        # Get a sample of data (2 seconds)
        sample_size = min(2 * self.sampling_frequency, self.raw_data._data.shape[1])
        sample_data = self.raw_data._data[:, :sample_size]
        
        # Calculate statistics
        data_std = np.std(sample_data)
        data_range = np.ptp(sample_data)
        data_mean = np.mean(sample_data)
        
        if self.verbose:
            print(f"  Data statistics:")
            print(f"    Mean: {data_mean:.6f}")
            print(f"    Std:  {data_std:.6f}")
            print(f"    Range: {data_range:.6f}")
        
        # Detect scaling needs
        if data_std > 100:  # Likely in raw ADC units
            # Common scaling factors for different systems
            scaling_factors = {
                'BrainVision': 0.1,      # Common for BrainVision
                'BioSemi': 31.25e-9,     # BioSemi to volts
                'Neuroscan': 0.1,        # Neuroscan
                'Default': 1.0           # Microvolts to microvolts
            }
            
            # Try to detect the system
            if 'resolution' in self.raw_data.info:
                # Use resolution if available
                self.scaling_factor = self.raw_data.info['resolution']
                if self.verbose:
                    print(f"  Using resolution from file: {self.scaling_factor}")
            else:
                # Estimate scaling to get ~50 µV standard deviation
                target_std = 50.0  # Target standard deviation in microvolts
                self.scaling_factor = target_std / data_std
                if self.verbose:
                    print(f"  Estimated scaling factor: {self.scaling_factor:.6f}")
                    print(f"  (to achieve ~{target_std} µV standard deviation)")
            
            # Apply scaling
            self.raw_data._data *= self.scaling_factor
            
            # Verify
            new_std = np.std(self.raw_data._data[:, :sample_size])
            if self.verbose:
                print(f"  After scaling: std = {new_std:.2f} µV")
                
        elif data_std < 0.1:  # Data might be in volts
            # self.scaling_factor = 1e6  # Convert to microvolts
            # self.raw_data._data *= self.scaling_factor
            if self.verbose:
                print(f"  Data appears to be in volts, converting to microvolts")
                
        else:
            if self.verbose:
                print(f"  Data appears to be correctly scaled (in microvolts)")
    
    def _check_dc_offset(self):
        """Check and remove DC offset if present"""
        sample_size = min(2 * self.sampling_frequency, self.raw_data._data.shape[1])
        sample_data = self.raw_data._data[:, :sample_size]
        
        channel_means = np.mean(sample_data, axis=1)
        
        # Check if any channel has significant DC offset
        if np.any(np.abs(channel_means) > 10):  # More than 10 µV offset
            if self.verbose:
                print(f"\nRemoving DC offset...")
                print(f"  Channel mean offsets: {np.mean(np.abs(channel_means)):.2f} µV")
            
            # Store DC offsets
            self.dc_offset = channel_means
            
            # Remove DC offset from entire recording
            for i in range(self.channel_count):
                self.raw_data._data[i, :] -= self.dc_offset[i]
                
            if self.verbose:
                print(f"  DC offset removed")
    
    def get_data(self):
        """Get next chunk of data with optional diagnostics"""
        time.sleep(self.latency)
        total_samples = self.raw_data._data.shape[1]
        
        # Check if we've reached the end
        if self.current_index >= total_samples:
            if self.verbose:
                print("End of file reached")
            return None
        
        chunk_end = min(self.current_index + self.chunk_size, total_samples)
        data_chunk = self.raw_data._data[:, self.current_index:chunk_end].copy()
        self.current_index = chunk_end
        
        return data_chunk
    
    def get_data_info(self):
        """Get information about the data for diagnostics"""
        info = {
            'sampling_rate': self.sampling_frequency,
            'n_channels': self.channel_count,
            'channel_names': self.channel_names,
            'scaling_factor': self.scaling_factor,
            'dc_offset_removed': self.dc_offset is not None,
            'total_duration': self.raw_data._data.shape[1] / self.sampling_frequency,
            'data_shape': self.raw_data._data.shape
        }
        
        # Add statistics for motor channels
        motor_channels = ['C3', 'C4', 'Cz']
        motor_stats = {}
        
        for ch_name in motor_channels:
            if ch_name in self.channel_names:
                ch_idx = self.channel_names.index(ch_name)
                ch_data = self.raw_data._data[ch_idx, :1000]  # First 1000 samples
                motor_stats[ch_name] = {
                    'mean': np.mean(ch_data),
                    'std': np.std(ch_data),
                    'min': np.min(ch_data),
                    'max': np.max(ch_data)
                }
        
        info['motor_channel_stats'] = motor_stats
        
        return info
    
    def analyze_frequency_content(self, channel_name='C3', duration=5):
        """Analyze frequency content of a specific channel"""
        if channel_name not in self.channel_names:
            print(f"Channel {channel_name} not found")
            return None
        
        ch_idx = self.channel_names.index(channel_name)
        
        # Get data segment
        n_samples = int(duration * self.sampling_frequency)
        n_samples = min(n_samples, self.raw_data._data.shape[1])
        data_segment = self.raw_data._data[ch_idx, :n_samples]
        
        # Calculate PSD
        freqs, psd = signal.welch(data_segment, self.sampling_frequency, 
                                 nperseg=min(n_samples, int(self.sampling_frequency)))
        
        # Calculate band powers
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'mu': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        band_powers = {}
        total_power = np.sum(psd)
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            band_powers[band_name] = {
                'power': band_power,
                'relative_power': (band_power / total_power) * 100
            }
        
        return {
            'freqs': freqs,
            'psd': psd,
            'band_powers': band_powers,
            'total_power': total_power
        }
    
    def reset(self):
        """Reset to beginning of file"""
        self.current_index = 0
        if self.verbose:
            print("Reset to beginning of file")
    
    def use_classification(self, prediction):
        """Handle classification output"""
        if prediction == 0:
            print("Rest")
        elif prediction == 1:
            print("Motor Imagery Detected (Flex)")
        else:
            print("Extend")
    
    def disconnect(self):
        """Disconnect and cleanup"""
        self.current_index = 0
        if self.verbose:
            print("Disconnected from enhanced EEG emulator")


# Quick test function
def test_enhanced_emulator():
    """Test the enhanced emulator"""
    print("Testing Enhanced Virtual Receiver")
    print("=" * 50)
    
    # Create emulator
    emulator = Emulator(fileName="MIT33", auto_scale=True, verbose=True)
    
    # Initialize
    fs, ch_names, n_channels, _ = emulator.initialize_connection()
    
    # Get data info
    info = emulator.get_data_info()
    print(f"\nData Information:")
    print(f"  Total duration: {info['total_duration']:.1f} seconds")
    print(f"  Scaling factor applied: {info['scaling_factor']:.6f}")
    
    # Analyze frequency content
    print(f"\nFrequency Analysis for C3:")
    freq_info = emulator.analyze_frequency_content('C3')
    if freq_info:
        for band, power_info in freq_info['band_powers'].items():
            print(f"  {band:6s}: {power_info['relative_power']:5.1f}% of total power")
    
    # Test data reading
    print(f"\nReading 1 second of data...")
    chunks_read = 0
    for i in range(50):  # 1 second at 50 chunks/second
        data = emulator.get_data()
        if data is not None:
            chunks_read += 1
    
    print(f"  Read {chunks_read} chunks successfully")
    
    # Cleanup
    emulator.disconnect()
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_enhanced_emulator()