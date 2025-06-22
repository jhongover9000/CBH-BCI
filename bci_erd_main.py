'''
BCI_ERD.py
--------
Main BCI ERD System Launcher

Description: Main launcher that coordinates all components:
- Livestream receiver for data collection
- ERD detection system with preprocessing
- Broadcasting system for output

Joseph Hong
'''

import numpy as np
import time
from datetime import datetime
import argparse
import threading
import queue
from collections import deque
import mne
from scipy import signal
from scipy.signal import butter, filtfilt
import tkinter as tk
from tkinter import ttk


# =============================================================
# CONFIGURATION
# =============================================================

class BCIConfig:
    """Central configuration for BCI ERD system"""
    
    # Buffer settings
    MAIN_BUFFER_DURATION = 30.0      # seconds
    BASELINE_BUFFER_DURATION = 20.0  # seconds (≤ main buffer)
    OPERATING_WINDOW_DURATION = 0.5 # seconds
    
    # ERD detection settings
    ERD_CHANNELS = ['C3','FC3', 'CP3', 'C1', 'C5', 'FC5', 'FC1', 'CP5', 'CP1']
    # ERD_CHANNELS = ['C3']
    ERD_BAND = (8, 20)  # mu band in Hz
    ERD_THRESHOLD = 60.0  # percent
    AVERAGE_ERDS = True  # average electrodes (vs individual ERD check)
    
    # Preprocessing settings
    USE_CAR = True  # Common Average Reference
    ARTIFACT_METHOD = 'threshold'  # 'ica' or 'threshold'
    ARTIFACT_THRESHOLD = 100.0  # microvolts
    SPATIAL_METHOD = '' # 'laplacian'
    
    # Connection settings
    VIRTUAL = False
    LIVESTREAM_IP = "169.254.1.147"
    LIVESTREAM_PORT = 51244
    
    # Output settings
    BROADCAST_ENABLED = True
    VERBOSE = False
    GUI_ENABLED = True
    SHOW_ANNOTATIONS = True  # Show annotation markers and correlation
    
    # Session settings
    SESSION_DURATION = 600  # seconds
    UPDATE_INTERVAL = 0.5   # seconds
    


# =============================================================
# PREPROCESSING FUNCTIONS
# =============================================================

class Preprocessor:
    """Preprocessing pipeline for EEG data"""
    
    def __init__(self, sampling_freq, channel_names):
        self.fs = sampling_freq
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        
        # Initialize filters
        self._init_filters()
        
    def _init_filters(self):
        """Initialize bandpass filters"""
        nyquist = self.fs / 2
        
        # Mu band filter
        self.mu_b, self.mu_a = butter(4, 
                                     [BCIConfig.ERD_BAND[0]/nyquist, 
                                      BCIConfig.ERD_BAND[1]/nyquist], 
                                     btype='band')
        
        # BPF (0.5-40 Hz)
        self.artifact_b, self.artifact_a = butter(4, 
                                                 [0.5/nyquist, 
                                                  min(40/nyquist, 0.99)], 
                                                 btype='band')
    
    def preprocess_data(self, data, selected_channels=None):
        """
        Full preprocessing pipeline
        
        Args:
            data: numpy array (n_channels, n_samples)
            selected_channels: list of channel indices for ERD
            
        Returns:
            preprocessed_data: full preprocessed data
            erd_data: preprocessed data for selected channels only
        """

        # Step 1: Basic Causal Filter (BPF)
        filtered_data = self._causal_filter(data)

        # Step 2: Artifact rejection
        ar_data = self._reject_artifacts(filtered_data)

        # Possibly ICA?
        
        # Step 3: Common Average Reference
        if BCIConfig.USE_CAR:
            car_data = self._apply_car(ar_data)
        else:
            car_data = ar_data

        # Step 4: Spatial Filtering
        if BCIConfig.SPATIAL_METHOD:
            spatial_data = self._spatial_filtering(car_data)
        else:
            spatial_data = car_data
        
        # Step 5: Extract and filter ERD channels
        if selected_channels is not None:

            erd_data = spatial_data[selected_channels, :]

            # We don't make another BPF for the specific band as
            # it may distort data and causes further overhead

            return erd_data
        
        return spatial_data
    
    def _causal_filter(self, data):
        """Apply basic frequency filter for artifact removal"""
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch, :] = filtfilt(self.artifact_b, self.artifact_a, data[ch, :])
        return filtered
    
    def _apply_car(self, data):
        """Apply Common Average Reference"""
        # Calculate average across all channels
        car_signal = np.mean(data, axis=0)
        
        # Subtract from each channel
        car_data = data - car_signal[np.newaxis, :]
        
        return car_data
    
    def _reject_artifacts(self, data):
        """Reject artifacts using selected method"""
        if BCIConfig.ARTIFACT_METHOD == 'threshold':
            return self._threshold_artifact_rejection(data)
        elif BCIConfig.ARTIFACT_METHOD == 'ica':
            return self._ica_artifact_rejection(data)
        else:
            return data
        
    def _spatial_filtering(self, data):
        """Apply spatial filtering to data (AFTER preprocessing, before power extraction)"""
        if BCIConfig.SPATIAL_METHOD == 'laplacian':
            return self._laplacian_spatial_filter(data)
        # idk if there are other methods we want then this can go here
    
    def _threshold_artifact_rejection(self, data):
        """Simple threshold-based artifact rejection"""
        clean_data = data.copy()
        
        # Find samples exceeding threshold
        artifact_mask = np.abs(data) > BCIConfig.ARTIFACT_THRESHOLD
        
        # Replace artifacts with interpolated values
        for ch in range(data.shape[0]):
            if np.any(artifact_mask[ch, :]):
                artifact_indices = np.where(artifact_mask[ch, :])[0]
                
                # Simple interpolation
                for idx in artifact_indices:
                    if 0 < idx < len(data[ch]) - 1:
                        clean_data[ch, idx] = (data[ch, idx-1] + data[ch, idx+1]) / 2
        
        return clean_data
    
    def _laplacian_spatial_filter(self, data):
        """Laplacian spatial filter for artifact rejection"""
        # Create Laplacian montage for motor channels
        laplacian_data = data.copy()
        
        # Find neighbor channels for each channel
        # This is simplified - in practice, use actual electrode positions
        for i, ch_name in enumerate(self.channel_names):
            if 'C3' in ch_name:
                # Small laplacian (5-point method)
                # neighbors = self._find_neighbors(i, ['FC3', 'CP3', 'C1', 'C5'])
                # Large laplacian (9-point method)
                neighbors = self._find_neighbors(i, ['FC3', 'CP3', 'C1', 'C5', 'FC5', 'FC1', 'CP5', 'CP1'])
            elif 'C4' in ch_name:
                # Small laplacian (5-point method)
                neighbors = self._find_neighbors(i, ['FC4', 'CP4', 'C2', 'C6'])
                # Large laplacian (9-point method)
            elif 'Cz' in ch_name:
                # Small laplacian (5-point method)
                neighbors = self._find_neighbors(i, ['FCz', 'CPz', 'C1', 'C2'])
                # Large laplacian (9-point method)
            else:
                continue
            
            if neighbors:
                # Apply Laplacian: channel - mean(neighbors)
                laplacian_data[i, :] = data[i, :] - np.mean(data[neighbors, :], axis=0)
        
        return laplacian_data
    
    def _find_neighbors(self, channel_idx, neighbor_names):
        """Find indices of neighbor channels"""
        neighbors = []
        for name in neighbor_names:
            for i, ch_name in enumerate(self.channel_names):
                if name in ch_name:
                    neighbors.append(i)
                    break
        return neighbors
    
    def _ica_artifact_rejection(self, data):
        """ICA-based artifact rejection (simplified)"""
        # Note: Full ICA is computationally expensive
        # MNE ICA IMPLEMENTATION WILL GO HERE
        print("Warning: ICA artifact rejection not fully implemented")
        return data
    
    def _apply_mu_filter(self, data):
        """Apply mu band filter to selected channels"""
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch, :] = filtfilt(self.mu_b, self.mu_a, data[ch, :])
        return filtered
    
    def calculate_band_power(self, data):
        """Calculate power in mu band"""
        # Use Welch's method for power calculation
        powers = []
        
        for ch in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch, :], 
                                     self.fs, 
                                     nperseg=min(len(data[ch, :]), int(self.fs)))
            
            # Extract mu band power
            mu_mask = (freqs >= BCIConfig.ERD_BAND[0]) & (freqs <= BCIConfig.ERD_BAND[1])
            mu_power = np.mean(psd[mu_mask])
            powers.append(mu_power)
        
        return np.array(powers)


# =============================================================
# ERD DETECTION SYSTEM
# =============================================================

class ERDDetectionSystem:
    """ERD Detection with integrated preprocessing"""
    
    def __init__(self, sampling_freq, channel_names):
        self.fs = sampling_freq
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(sampling_freq, channel_names)
        
        # Find ERD channel indices
        self.erd_channel_indices = self._find_erd_channels()
        
        # Baseline state
        self.baseline_buffer = deque(maxlen=int(BCIConfig.BASELINE_BUFFER_DURATION * self.fs))
        self.baseline_power = None
        self.baseline_calculated = False
        
        # Statistics
        self.erd_history = deque(maxlen=100)
        
    def _find_erd_channels(self):
        """Find indices of ERD channels"""
        indices = []
        for target in BCIConfig.ERD_CHANNELS:
            found = False
            for i, ch_name in enumerate(self.channel_names):
                if target in ch_name:
                    indices.append(i)
                    found = True
                    break
            
            if not found and BCIConfig.VERBOSE:
                print(f"Warning: Channel {target} not found")
        
        if not indices:
            print("Warning: No ERD channels found, using first 3")
            indices = list(range(min(3, self.n_channels)))
        
        print(f"ERD channel indices: {indices} ({[self.channel_names[i] for i in indices]})")
        return indices
    
    def add_to_baseline(self, data):
        """Add data to baseline buffer"""
        # Add samples to baseline buffer (maintain individual samples for proper deque behavior)
        for i in range(data.shape[1]):
            self.baseline_buffer.append(data[:, i])
        
        # Check if baseline buffer is full
        if BCIConfig.VERBOSE:
            print(f"Baseline buffer: {len(self.baseline_buffer)}/{self.baseline_buffer.maxlen}")
        
        if len(self.baseline_buffer) >= self.baseline_buffer.maxlen:
            if not self.baseline_calculated:
                self.calculate_baseline()

    def empty_baseline(self):
        """Reset baseline buffer"""
        self.baseline_buffer = deque(maxlen=int(BCIConfig.BASELINE_BUFFER_DURATION * self.fs))
        self.baseline_power = None
        self.baseline_calculated = False
        print("Baseline buffer reset")
    
    def calculate_baseline(self, main_buffer=None, force=False):
        """Calculate baseline power from buffer"""
        
        # Check if we have enough data
        if not force and len(self.baseline_buffer) < self.baseline_buffer.maxlen:
            if BCIConfig.VERBOSE:
                print(f"Baseline buffer not full: {len(self.baseline_buffer)}/{self.baseline_buffer.maxlen}")
            return False
                
        # Prepare baseline data
        if force and main_buffer is not None:
            # Manual baseline: use recent data from main buffer
            # print(f"Manual baseline - main buffer length: {len(main_buffer)}")
            
            # Convert main_buffer (deque of individual samples) to numpy array
            if len(main_buffer) < self.baseline_buffer.maxlen:
                print(f"Warning: Main buffer ({len(main_buffer)}) smaller than baseline requirement ({self.baseline_buffer.maxlen})")
                # Use all available data
                main_buffer_array = np.array(list(main_buffer))
            else:
                # Use most recent baseline_buffer.maxlen samples
                recent_samples = list(main_buffer)[-self.baseline_buffer.maxlen:]
                main_buffer_array = np.array(recent_samples)
            
            # Shape should be (n_samples, n_channels), transpose to (n_channels, n_samples)
            if main_buffer_array.ndim == 2:
                baseline_data = main_buffer_array.T
                # print(f"Manual baseline data shape: {baseline_data.shape}")
            else:
                print(f"Error: Unexpected main buffer shape: {main_buffer_array.shape}")
                return False
                
        else:
            # Normal baseline: use baseline_buffer
            if len(self.baseline_buffer) == 0:
                print("Error: Baseline buffer is empty")
                return False
                
            # Convert deque to numpy array and transpose
            baseline_buffer_array = np.array(list(self.baseline_buffer))
            
            # Shape should be (n_samples, n_channels), transpose to (n_channels, n_samples)
            if baseline_buffer_array.ndim == 2:
                baseline_data = baseline_buffer_array.T
                # print(f"Normal baseline data shape: {baseline_data.shape}")
            else:
                print(f"Error: Unexpected baseline buffer shape: {baseline_buffer_array.shape}")
                return False
        
        # Validate data shape
        if baseline_data.shape[0] != self.n_channels:
            print(f"Error: Expected {self.n_channels} channels, got {baseline_data.shape[0]}")
            return False
        
        if baseline_data.shape[1] < self.fs:  # At least 1 second of data
            print(f"Warning: Limited baseline data: {baseline_data.shape[1]} samples ({baseline_data.shape[1]/self.fs:.2f}s)")
        
        try:
            # Preprocess baseline data (extract only ERD channels)
            erd_data = self.preprocessor.preprocess_data(baseline_data, self.erd_channel_indices)
            # print(f"ERD data shape after preprocessing: {erd_data.shape}")
            
            # Calculate baseline power
            self.baseline_power = self.preprocessor.calculate_band_power(erd_data)
            self.baseline_calculated = True
            
            if BCIConfig.VERBOSE:
                print(f"Baseline power calculated successfully:")
                for i, ch_idx in enumerate(self.erd_channel_indices):
                    ch_name = self.channel_names[ch_idx]
                    print(f"  {i} {ch_name}: {self.baseline_power[i]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Error calculating baseline power: {e}")
            if BCIConfig.VERBOSE:
                import traceback
                traceback.print_exc()
            return False
    
    def detect_erd(self, data):
        """
        Detect ERD in operating window - IMPROVED
        
        Args:
            data: numpy array (n_channels, n_samples) - operating window
            
        Returns:
            detected: bool - whether ERD was detected
            erd_values: dict - ERD percentage for each channel
        """
        if not self.baseline_calculated:
            if BCIConfig.VERBOSE:
                print("Baseline not calculated yet")
            return False, {}
        
        if self.baseline_power is None:
            if BCIConfig.VERBOSE:
                print("Baseline power is None")
            return False, {}
        
        try:
            # Preprocess data (extract only ERD channels)
            erd_data = self.preprocessor.preprocess_data(data, self.erd_channel_indices)
            
            # Calculate current power
            current_power = self.preprocessor.calculate_band_power(erd_data)
            
            # Calculate ERD percentage
            erd_values = {}
            detected_channels = []
            
            for i, ch_idx in enumerate(self.erd_channel_indices):
                if self.baseline_power[i] > 0:
                    erd_percent = ((self.baseline_power[i] - current_power[i]) / 
                                  self.baseline_power[i]) * 100
                    # print(f"Power: {current_power}")
                    # print(f"Baseline: {self.baseline_power}")
                    # print(f"ERD: {erd_percent}")
                    
                    ch_name = self.channel_names[ch_idx]
                    erd_values[ch_name] = erd_percent
                    
                    # Check for individual ERD detection
                    if erd_percent > BCIConfig.ERD_THRESHOLD:
                        detected_channels.append(ch_name)
                else:
                    # Handle zero baseline power
                    ch_name = self.channel_names[ch_idx]
                    erd_values[ch_name] = 0.0
                    if BCIConfig.VERBOSE:
                        print(f"Warning: Zero baseline power for {ch_name}")
            
            # Store in history
            if erd_values:
                avg_erd = np.mean(list(erd_values.values()))
                self.erd_history.append(avg_erd)
            else:
                avg_erd = 0.0
            
            # Determine if ERD detected
            if BCIConfig.AVERAGE_ERDS:
                # Use average ERD across channels
                detected = avg_erd > BCIConfig.ERD_THRESHOLD
            else:
                # Use individual channel detection
                detected = len(detected_channels) > 0
            
            if detected and BCIConfig.VERBOSE:
                print(f"ERD DETECTED! Channels: {detected_channels}, Avg: {avg_erd:.1f}%")
            
            return detected, erd_values
            
        except Exception as e:
            print(f"Error in ERD detection: {e}")
            if BCIConfig.VERBOSE:
                import traceback
                traceback.print_exc()
            return False, {}


# =============================================================
# MAIN BCI SYSTEM
# =============================================================

class BCIERDSystem:
    """Main BCI ERD System coordinating all components"""
    
    def __init__(self, config=None):
        self.config = config or BCIConfig()
        
        # Components
        self.receiver = None
        self.erd_detector = None
        self.broadcaster = None
        
        # Buffers
        self.main_buffer = None
        self.main_buffer_size = 0
        
        # State
        self.running = False
        self.baseline_ready = False
        self.session_start_time = None
        
        # Statistics
        self.detection_count = 0
        self.sample_count = 0

        # Annotation tracking
        self.annotations = []  # Store all annotations encountered
        self.annotation_times = []  # Times relative to session start
        self.last_annotation_check = 0
        
        # ERD detection tracking for comparison
        self.erd_detection_times = []  # Times when ERD was detected
        
        # GUI
        self.gui_queue = queue.Queue() if self.config.GUI_ENABLED else None
        self.gui_thread = None
        
    def initialize(self, args):
        """Initialize all components"""
        print("Initializing BCI ERD System...")
        
        # Initialize receiver
        self._init_receiver(args)
        
        # Initialize connection
        print("Establishing connection...")
        self.fs, self.ch_names, self.n_channels, _ = self.receiver.initialize_connection()
        print(f"Connected: {self.n_channels} channels at {self.fs} Hz")
        
        # Calculate buffer sizes
        self.main_buffer_size = int(self.config.MAIN_BUFFER_DURATION * self.fs)
        self.operating_window_size = int(self.config.OPERATING_WINDOW_DURATION * self.fs)
        
        # Initialize main buffer
        self.main_buffer = deque(maxlen=self.main_buffer_size)
        
        # Initialize ERD detector
        self.erd_detector = ERDDetectionSystem(self.fs, self.ch_names)
        
        # Initialize broadcaster if enabled
        if self.config.BROADCAST_ENABLED:
            self._init_broadcaster()
        
        # Start GUI if enabled
        if self.config.GUI_ENABLED:
            self._start_gui()
        
        print("Initialization complete!")
        
    def _init_receiver(self, args):
        """Initialize appropriate receiver"""
        if args.virtual:
            from receivers import virtual_receiver
            self.receiver = virtual_receiver.Emulator(verbose=BCIConfig.VERBOSE)
        else:
            from receivers import livestream_receiver
            self.receiver = livestream_receiver.LivestreamReceiver(
                address=self.config.LIVESTREAM_IP,
                port=self.config.LIVESTREAM_PORT,
                broadcast=self.config.BROADCAST_ENABLED
            )
    
    def _init_broadcaster(self):
        """Initialize broadcasting component"""
        # Broadcaster is handled by receiver in this implementation
        pass
    
    def _start_gui(self):
        """Start GUI monitor in separate thread"""
        from bci_erd_gui import ERDMonitorGUI
        self.gui_thread = threading.Thread(target=self._run_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()
    
    def _run_gui(self):
        """Run GUI in separate thread"""
        from bci_erd_gui import ERDMonitorGUI
        gui = ERDMonitorGUI(self.gui_queue, self, BCIConfig.ERD_THRESHOLD)
        gui.run()
    
    def run(self):
        """Main processing loop"""
        print("\n" + "="*60)
        print("Starting ERD Detection")
        print("="*60)
        
        self.running = True
        self.session_start_time = time.time()
        last_update_time = time.time()
        
        # Phase tracking
        phase = "COLLECTING_BASELINE"
        
        try:
            while self.running and (time.time() - self.session_start_time) < self.config.SESSION_DURATION:
                
                # Handle inputs
                self.handle_keyboard_input()
                
                # Get data from receiver
                data = self.receiver.get_data()
                
                if data is None:
                    continue
                
                self.sample_count += data.shape[1]
                
                # Add data to main buffer (sample by sample to maintain deque behavior)
                for i in range(data.shape[1]):
                    self.main_buffer.append(data[:, i])
                
                # Phase 1: Fill baseline buffer
                if phase == "COLLECTING_BASELINE":
                    # Add to baseline buffer
                    self.erd_detector.add_to_baseline(data)
                    
                    # Check if baseline is ready
                    if self.erd_detector.baseline_calculated:
                        phase = "DETECTING"
                        print("\n✓ Baseline established! Starting ERD detection...\n")
                        self.baseline_ready = True
                    else:
                        # Show progress
                        progress = len(self.erd_detector.baseline_buffer) / self.erd_detector.baseline_buffer.maxlen
                        if time.time() - last_update_time > 0.5:
                            print(f"\rCollecting baseline: {progress*100:.1f}%", end='', flush=True)
                
                # Phase 2: ERD Detection
                elif phase == "DETECTING" and len(self.main_buffer) >= self.operating_window_size:
                    
                    self.manual_baseline_calculation()
                    
                    # Extract operating window from end of main buffer
                    window_data = np.array(list(self.main_buffer))[-self.operating_window_size:].T
                    
                    # Detect ERD
                    detected, erd_values = self.erd_detector.detect_erd(window_data)
                    
                    # Handle detection
                    if detected:
                        self.detection_count += 1
                        current_runtime = time.time() - self.session_start_time
                        self.erd_detection_times.append(current_runtime)
                        
                        if self.config.BROADCAST_ENABLED:
                            self.receiver.use_classification(1)  # Send TAP command
                    
                    # Check for annotations if using virtual receiver
                    self._check_for_annotations()
                    
                    # Update display
                    current_time = time.time()
                    if current_time - last_update_time >= self.config.UPDATE_INTERVAL:
                        self._update_display(detected, erd_values, current_time)
                        last_update_time = current_time
                        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            if self.config.VERBOSE:
                import traceback
                traceback.print_exc()
        finally:
            self.cleanup()

    def _check_for_annotations(self):
        """Check for annotations if using virtual receiver"""
        # Only check if annotations are enabled and using virtual receiver
        if not self.config.SHOW_ANNOTATIONS:
            return
            
        # Only check if using virtual receiver with annotation support
        if hasattr(self.receiver, 'annotation_onsets') and hasattr(self.receiver, 'current_index'):
            current_time = self.receiver.current_index / self.fs
            
            # Check for new annotations since last check
            if self.receiver.annotation_onsets is not None:
                for i, onset in enumerate(self.receiver.annotation_onsets):
                    # Check if annotation is in the current time window and hasn't been recorded
                    if self.last_annotation_check <= onset < current_time:
                        runtime = onset  # Already relative to data start
                        desc = self.receiver.annotation_descriptions[i]
                        
                        # Skip certain annotations (like baseline markers)
                        if desc != 'Stimulus/S  1':
                            self.annotations.append({
                                'time': runtime,
                                'description': desc,
                                'sample': int(onset * self.fs)
                            })
                            self.annotation_times.append(runtime)
                            
                            # Print annotation with timing comparison
                            print(f"\n📍 ANNOTATION: '{desc}' at t={runtime:.3f}s")
                            
                            # Check for recent ERD detections (within 2 seconds)
                            recent_detections = [t for t in self.erd_detection_times 
                                               if abs(t - runtime) < 2.0]
                            if recent_detections:
                                for det_time in recent_detections:
                                    diff = det_time - runtime
                                    # print(f"   ✓ ERD detected {abs(diff):.3f}s {'after' if diff > 0 else 'before'} annotation")
                            # else:
                                # print(f"   ✗ No ERD detection within 2s of annotation")
            
            self.last_annotation_check = current_time
    
    def _update_display(self, detected, erd_values, current_time):
        """Update console and GUI displays"""
        runtime = current_time - self.session_start_time
        if BCIConfig.VIRTUAL:
            current_time = self.receiver.current_index / self.fs
            runtime = current_time
        detection_rate = (self.detection_count / runtime) * 60
        
        # Console output
        if not self.config.VERBOSE:
            erd_str = " | ".join([f"{ch}:{erd:.1f}%" for ch, erd in erd_values.items()])
            avg_erd = np.mean(list(erd_values.values()))
            erd_values['avg'] = avg_erd
            status = "DETECTED" if detected else "--------"
            if(detected):
                print(f"[{runtime:6.1f}s] Avg: {avg_erd} | {erd_str} | {status} | Count: {self.detection_count}")
        else:
            avg_erd = np.mean(list(erd_values.values()))
            erd_values['avg'] = avg_erd

        # GUI update
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait({
                    'detected': detected,
                    'erd_values': {'avg':avg_erd},
                    'runtime': runtime,
                    'count': self.detection_count,
                    'rate': detection_rate,
                    'annotations': self.annotations,
                    'baseline_ready': self.baseline_ready
                })
            except queue.Full:
                pass
    
    def manual_baseline_calculation(self):
        """Manually trigger baseline calculation - FIXED"""
        if self.erd_detector is None:
            print("ERD detector not initialized")
            return False
            
        if len(self.main_buffer) == 0:
            print("Main buffer is empty")
            return False
            
        # print(f"Manual baseline calculation with {len(self.main_buffer)} samples...")
        success = self.erd_detector.calculate_baseline(main_buffer=self.main_buffer, force=True)
        
        if success:
            self.baseline_ready = True
            if BCIConfig.VERBOSE:
                print("✓ Manual baseline calculation successful")
        else:
            if BCIConfig.VERBOSE:
                print("✗ Manual baseline calculation failed")
            
        return success
    
    def reset_baseline(self):
        """Reset baseline and start over"""
        if self.erd_detector:
            self.erd_detector.empty_baseline()
            self.baseline_ready = False
            print("Baseline reset - will recalculate automatically")
    
    # Add diagnostic method
    def print_system_status(self):
        """Print detailed system status for debugging"""
        print("\n" + "="*50)
        print("BCI SYSTEM STATUS")
        print("="*50)
        
        print(f"Main buffer length: {len(self.main_buffer) if self.main_buffer else 0}")
        print(f"Baseline ready: {self.baseline_ready}")
        
        if self.erd_detector:
            print(f"Baseline buffer length: {len(self.erd_detector.baseline_buffer)}")
            print(f"Baseline calculated: {self.erd_detector.baseline_calculated}")
            print(f"ERD channels: {[self.ch_names[i] for i in self.erd_detector.erd_channel_indices]}")
            
            if self.erd_detector.baseline_power is not None:
                print("Baseline powers:")
                for i, ch_idx in enumerate(self.erd_detector.erd_channel_indices):
                    ch_name = self.ch_names[ch_idx]
                    print(f"  {ch_name}: {self.erd_detector.baseline_power[i]:.6f}")
            else:
                print("Baseline power: None")
        
        print("="*50 + "\n")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Disconnect receiver
        if self.receiver:
            self.receiver.disconnect()
        
        # Print summary
        if self.session_start_time:
            total_runtime = time.time() - self.session_start_time
            print("\n" + "="*60)
            print("Session Summary")
            print("="*60)
            print(f"Total Runtime:     {total_runtime:.1f} seconds")
            print(f"Samples Processed: {self.sample_count}")
            print(f"ERD Detections:    {self.detection_count}")
            print(f"Detection Rate:    {self.detection_count/total_runtime*60:.1f} per minute")
            
            # Annotation summary
            if self.annotations:
                print(f"\nAnnotations Found: {len(self.annotations)}")
                print("\nAnnotation-Detection Correlation:")
                print("-" * 40)
                
                for ann in self.annotations:
                    # Find closest ERD detection
                    if self.erd_detection_times:
                        closest_det = min(self.erd_detection_times, 
                                        key=lambda x: abs(x - ann['time']))
                        time_diff = closest_det - ann['time']
                        
                        if abs(time_diff) < 2.0:  # Within 2 seconds
                            print(f"  {ann['description']:20s} at {ann['time']:6.2f}s → "
                                  f"ERD at {closest_det:6.2f}s (Δ={time_diff:+.2f}s)")
                        else:
                            print(f"  {ann['description']:20s} at {ann['time']:6.2f}s → "
                                  f"No ERD within 2s")
                    else:
                        print(f"  {ann['description']:20s} at {ann['time']:6.2f}s → "
                              f"No ERD detections")
            
            print("="*60)

    def handle_keyboard_input(self):
        """Handle keyboard input for debugging"""
        try:
            import select
            import sys
            
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                
                if key == 'b':
                    self.manual_baseline_calculation()
                elif key == 'r':
                    self.reset_baseline()
                elif key == 's':
                    self.print_system_status()
                elif key == 'q':
                    self.running = False
                elif key == 'd':
                    BCIConfig.ERD_THRESHOLD -= 5
                elif key == 'f':
                    BCIConfig.ERD_THRESHOLD += 5
                    
        except:
            pass  # Not available on all systems


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BCI ERD Detection System")
    
    # Connection options
    parser.add_argument('--virtual', action='store_true', default=BCIConfig.VIRTUAL,
                       help="Use virtual receiver (testing)")
    parser.add_argument('--ip', default=BCIConfig.LIVESTREAM_IP,
                       help="Livestream IP address")
    parser.add_argument('--port', type=int, default=BCIConfig.LIVESTREAM_PORT,
                       help="Livestream port")
    
    # ERD options
    parser.add_argument('--channels', nargs='+', default=BCIConfig.ERD_CHANNELS,
                       help="Channels for ERD detection")
    parser.add_argument('--threshold', type=float, default=BCIConfig.ERD_THRESHOLD,
                       help="ERD detection threshold (%)")
    parser.add_argument('--baseline-duration', type=float, default=BCIConfig.BASELINE_BUFFER_DURATION,
                       help="Baseline duration (seconds)")
    parser.add_argument('--window', type=float, default=BCIConfig.OPERATING_WINDOW_DURATION,
                       help="Operating window duration (seconds)")
    
    # Preprocessing options
    parser.add_argument('--no-car', action='store_true',
                       help="Disable Common Average Reference")
    parser.add_argument('--artifact-method', choices=['threshold', 'laplacian', 'ica'],
                       default=BCIConfig.ARTIFACT_METHOD,
                       help="Artifact rejection method")
    
    # Output options
    parser.add_argument('--no-broadcast', action='store_true',
                       help="Disable broadcasting")
    parser.add_argument('--no-gui', action='store_true',
                       help="Disable GUI")
    parser.add_argument('--verbose', action='store_true',
                       help="Verbose output")
    parser.add_argument('--no-annotations', action='store_true',
                       help="Disable annotation tracking")
    
    # Session options
    parser.add_argument('--duration', type=int, default=BCIConfig.SESSION_DURATION,
                       help="Session duration (seconds)")
    
    args = parser.parse_args()
    
    # Update configuration
    BCIConfig.LIVESTREAM_IP = args.ip
    BCIConfig.LIVESTREAM_PORT = args.port
    BCIConfig.ERD_CHANNELS = args.channels
    BCIConfig.ERD_THRESHOLD = args.threshold
    BCIConfig.BASELINE_BUFFER_DURATION = args.baseline_duration
    BCIConfig.OPERATING_WINDOW_DURATION = args.window
    BCIConfig.USE_CAR = not args.no_car
    BCIConfig.ARTIFACT_METHOD = args.artifact_method
    BCIConfig.BROADCAST_ENABLED = not args.no_broadcast
    BCIConfig.GUI_ENABLED = not args.no_gui
    BCIConfig.VERBOSE = args.verbose
    BCIConfig.SHOW_ANNOTATIONS = not args.no_annotations
    BCIConfig.SESSION_DURATION = args.duration
    BCIConfig.VIRTUAL = args.virtual

    # Validate baseline duration
    if BCIConfig.BASELINE_BUFFER_DURATION > BCIConfig.MAIN_BUFFER_DURATION:
        print(f"Error: Baseline duration ({BCIConfig.BASELINE_BUFFER_DURATION}s) cannot exceed main buffer ({BCIConfig.MAIN_BUFFER_DURATION}s)")
        return
    
    # Print configuration
    print("BCI ERD System Configuration")
    print("="*60)
    print(f"Mode:              {'Virtual' if args.virtual else 'Livestream'}")
    print(f"Channels:          {BCIConfig.ERD_CHANNELS}")
    print(f"Threshold:         {BCIConfig.ERD_THRESHOLD}%")
    print(f"Baseline Duration: {BCIConfig.BASELINE_BUFFER_DURATION}s")
    print(f"Operating Window:  {BCIConfig.OPERATING_WINDOW_DURATION}s")
    print(f"Preprocessing:     CAR={'Yes' if BCIConfig.USE_CAR else 'No'}, Artifacts={BCIConfig.ARTIFACT_METHOD}")
    print(f"Broadcasting:      {'Yes' if BCIConfig.BROADCAST_ENABLED else 'No'}")
    print(f"GUI:               {'Yes' if BCIConfig.GUI_ENABLED else 'No'}")
    print(f"Annotations:       {'Yes' if BCIConfig.SHOW_ANNOTATIONS else 'No'}")
    print("="*60 + "\n")
    
    # Create and run system
    system = BCIERDSystem()
    system.initialize(args)
    system.run()


if __name__ == "__main__":
    main()