'''
Enhanced Virtual Receiver with automatic scaling, diagnostics, annotation detection, and broadcasting
Fixes common issues with ERD detection in virtual mode and adds TCP broadcasting capability
'''

import time
import numpy as np
import mne
from scipy import signal
from broadcasting import TCP_Server
import threading

data_dir = "./data/rawdata/cbh/"

class Emulator:
    def __init__(self, fileName="CBH0018", auto_scale=False, verbose=False, broadcast=False):
        self.vhdr_file = f"{data_dir}{fileName}.vhdr"
        self.eeg_file = f"{data_dir}{fileName}.eeg"
        self.channel_count = 0
        self.sampling_frequency = 0
        self.sampling_interval_us = 0
        self.channel_names = []
        self.latency = 0.005
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
        
        # Annotation tracking
        self.annotations = None
        self.annotation_onsets = None
        self.annotation_descriptions = None
        self.data_start_time = 0  # Track when the cropped data starts
        
        # Broadcasting feature
        self.broadcasting = broadcast
        self.server = None

        # If broadcasting classification (for further applications), set up TCP server
        if broadcast:
            self.server = TCP_Server.TCPServer()
            self.server.initialize_connection()
            if self.verbose:
                print("TCP Server Initialized for broadcasting")
        
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

        # Store original annotations before cropping
        # self.annotations = self.raw_data.annotations.copy()
        
        # Find specific annotation (marker)
        target_annotation = 'Stimulus/S  4'  # Replace with your marker name

        # # Get annotations
        # annotations = self.raw_data.annotations

        # # Find the first occurrence of the target annotation
        # mask = annotations.description == target_annotation
        # if np.any(mask):
        #     start_time = annotations.onset[mask][15]  # First occurrence
        #     self.data_start_time = start_time  # Store the actual start time

        #     print (self.data_start_time)

        # self.data_start_time = 800
            
        # # Crop from this time point
        # self.raw_data = self.raw_data.crop(tmin=self.data_start_time)
            
        # Update annotations after cropping - adjust onset times relative to new start
        self.annotations = self.raw_data.annotations
        self.annotation_onsets = self.annotations.onset
        self.annotation_descriptions = self.annotations.description
        
        if self.verbose:
            print(f"Found {len(self.annotation_onsets)} annotations after cropping")
            if len(self.annotation_onsets) > 0:
                print(f"First few annotations: {list(self.annotation_descriptions[:5])}")
        
        # Create info
        self.info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.sampling_frequency,
            ch_types="eeg"
        )
        
        if self.verbose:
            print(f"Loaded {self.channel_count} channels at {self.sampling_frequency} Hz")
            print(f"Chunk size: {self.chunk_size} samples per channel")
            if self.broadcasting:
                print("Broadcasting enabled - classification results will be sent via TCP")
        
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
    
    def _check_annotations_in_chunk(self, start_sample, end_sample):
        """Check if any annotations fall within the current chunk"""
        if self.annotation_onsets is None or len(self.annotation_onsets) == 0:
            return []
        
        # Convert sample indices to time
        start_time = start_sample / self.sampling_frequency
        end_time = end_sample / self.sampling_frequency
        
        # Find annotations within this time window
        annotations_in_chunk = []
        for i, onset_time in enumerate(self.annotation_onsets):
            if start_time <= onset_time < end_time:
                annotations_in_chunk.append({
                    'onset_time': onset_time,
                    'description': self.annotation_descriptions[i],
                    'sample_index': int(onset_time * self.sampling_frequency)
                })
        
        return annotations_in_chunk
    
    def get_data(self):
        """Get next chunk of data with annotation detection"""
        time.sleep(self.latency)
        total_samples = self.raw_data._data.shape[1]
        
        # Check if we've reached the end
        if self.current_index >= total_samples:
            # if self.verbose:
                # print("End of file reached")
            return None, []
        
        chunk_end = min(self.current_index + self.chunk_size, total_samples)
        
        # Check for annotations in this chunk
        annotations_in_chunk = self._check_annotations_in_chunk(self.current_index, chunk_end)
        
        # Convert to same format as livestream receiver
        new_annotations = []
        
        # Print annotation information if found
        if annotations_in_chunk:
            for annotation in annotations_in_chunk:
                if annotation['description'] != 'Stimulus/S  1':
                    current_time = self.current_index / self.sampling_frequency
                    # print(f"📍 ANNOTATION DETECTED at t={annotation['onset_time']:.3f}s "
                    #     f"(sample {annotation['sample_index']}): '{annotation['description']}'")
                    
                    # Create annotation in same format as livestream
                    new_annotations.append({
                        'time': annotation['onset_time'],
                        'description': annotation['description'],
                        'type': 'Stimulus',  # Virtual annotations are usually stimulus markers
                        'sample': annotation['sample_index'],
                        'position': annotation['sample_index'] - self.current_index,
                        'channel': -1  # Virtual annotations don't have channel info
                    })
            
        # Get the data chunk
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
            'data_shape': self.raw_data._data.shape,
            'n_annotations': len(self.annotation_onsets) if self.annotation_onsets is not None else 0,
            'broadcasting_enabled': self.broadcasting
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
    
    def get_all_annotations(self):
        """Get all annotations with their timing information"""
        if self.annotation_onsets is None:
            return []
        
        annotations_list = []
        for i, (onset, desc) in enumerate(zip(self.annotation_onsets, self.annotation_descriptions)):
            annotations_list.append({
                'index': i,
                'onset_time': onset,
                'sample_index': int(onset * self.sampling_frequency),
                'description': desc
            })
        
        return annotations_list
    
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
        """Handle classification output with broadcasting support"""
        if prediction == 0:
            if self.verbose:
                print("Rest")
        elif prediction == 1:
            if self.verbose:
                print("Motor Imagery Detected (Flex)")
            # Send TAP message if broadcasting is enabled
            if self.broadcasting and self.server:
                try:
                    self.server.send_message_tcp("TAP")
                    if self.verbose:
                        print("Sent TAP command via TCP")
                except Exception as e:
                    if self.verbose:
                        print(f"Error sending TCP message: {e}")
        else:
            if self.verbose:
                print("Extend")
    
    def send_custom_message(self, message):
        """Send a custom message via TCP if broadcasting is enabled"""
        if self.broadcasting and self.server:
            try:
                self.server.send_message_tcp(message)
                if self.verbose:
                    print(f"Sent custom message via TCP: {message}")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"Error sending TCP message: {e}")
                return False
        else:
            if self.verbose:
                print("Broadcasting not enabled or server not initialized")
            return False
    
    def get_server_status(self):
        """Get the status of the TCP server"""
        if self.broadcasting and self.server:
            return {
                'broadcasting_enabled': True,
                'server_initialized': self.server is not None,
                'server_object': self.server
            }
        else:
            return {
                'broadcasting_enabled': False,
                'server_initialized': False,
                'server_object': None
            }
    
    def disconnect(self):
        """Disconnect and cleanup"""
        self.current_index = 0
        
        # Close TCP server if it was initialized
        if self.broadcasting and self.server:
            try:
                # Note: The actual TCP_Server class might have a different cleanup method
                # You may need to adjust this based on the TCP_Server implementation
                if hasattr(self.server, 'close') or hasattr(self.server, 'disconnect'):
                    if hasattr(self.server, 'close'):
                        self.server.close()
                    else:
                        self.server.disconnect()
                    if self.verbose:
                        print("TCP server disconnected")
            except Exception as e:
                if self.verbose:
                    print(f"Error closing TCP server: {e}")
        
        if self.verbose:
            print("Disconnected from enhanced EEG emulator")


# Quick test function
def test_enhanced_emulator_with_broadcasting():
    """Test the enhanced emulator with annotation detection and broadcasting"""
    print("Testing Enhanced Virtual Receiver with Annotation Detection and Broadcasting")
    print("=" * 80)
    
    # Test without broadcasting first
    print("\n1. Testing without broadcasting:")
    emulator = Emulator(fileName="MIT33", auto_scale=True, verbose=True, broadcast=False)
    
    # Initialize
    fs, ch_names, n_channels, _ = emulator.initialize_connection()
    
    # Get data info
    info = emulator.get_data_info()
    print(f"\nData Information:")
    print(f"  Total duration: {info['total_duration']:.1f} seconds")
    print(f"  Number of annotations: {info['n_annotations']}")
    print(f"  Scaling factor applied: {info['scaling_factor']:.6f}")
    print(f"  Broadcasting enabled: {info['broadcasting_enabled']}")
    
    # Test classification without broadcasting
    print(f"\nTesting classification without broadcasting:")
    emulator.use_classification(0)  # Rest
    emulator.use_classification(1)  # Motor Imagery
    
    # Cleanup
    emulator.disconnect()
    
    # Test with broadcasting
    print("\n" + "="*50)
    print("2. Testing with broadcasting:")
    try:
        emulator_broadcast = Emulator(fileName="MIT33", auto_scale=True, verbose=True, broadcast=True)
        
        # Initialize
        fs, ch_names, n_channels, _ = emulator_broadcast.initialize_connection()
        
        # Get server status
        server_status = emulator_broadcast.get_server_status()
        print(f"\nServer Status:")
        print(f"  Broadcasting enabled: {server_status['broadcasting_enabled']}")
        print(f"  Server initialized: {server_status['server_initialized']}")
        
        # Test classification with broadcasting
        print(f"\nTesting classification with broadcasting:")
        emulator_broadcast.use_classification(0)  # Rest
        emulator_broadcast.use_classification(1)  # Motor Imagery (should send TAP)
        
        # Test custom message
        print(f"\nTesting custom message:")
        emulator_broadcast.send_custom_message("CUSTOM_COMMAND")
        
        # Cleanup
        emulator_broadcast.disconnect()
        
    except ImportError:
        print("TCP_Server module not available - broadcasting feature cannot be tested")
        print("Make sure the 'broadcasting' module with TCP_Server is available")
    except Exception as e:
        print(f"Error testing broadcasting: {e}")
    
    print("\nTest completed!")


def quick_test():
    """Quick test without broadcasting for basic functionality"""
    print("Quick Test - Enhanced Virtual Receiver")
    print("=" * 40)
    
    # Create emulator without broadcasting
    emulator = Emulator(fileName="MIT33", auto_scale=True, verbose=True, broadcast=False)
    
    # Initialize
    fs, ch_names, n_channels, _ = emulator.initialize_connection()
    
    # Read a few chunks
    print(f"\nReading 2 seconds of data...")
    for i in range(100):  # 2 seconds at 50 chunks/second
        data = emulator.get_data()
        if data is None:
            break
    
    # Test classification
    emulator.use_classification(1)
    
    # Cleanup
    emulator.disconnect()
    
    print("Quick test completed!")


if __name__ == "__main__":
    # Run quick test by default, full test with broadcasting if TCP_Server is available
    try:
        from broadcasting import TCP_Server
        test_enhanced_emulator_with_broadcasting()
    except ImportError:
        print("TCP_Server not available, running quick test without broadcasting...")
        quick_test()