'''
BCI ERD SYSTEM MAIN

Desc: Unified script for ERD detection system

Joseph Hong
'''

'''
bci_system_complete.py
---------------------
Complete streamlined BCI ERD System with all features integrated
Supports both GUI and keyboard control modes
'''

import numpy as np
import time
from datetime import datetime
import threading
import queue
from collections import deque
import mne
from scipy import signal
from scipy.signal import butter, filtfilt
import sys
import select
import os
import joblib

# Import receivers
from receivers import virtual_receiver, livestream_receiver
from csp_svm_integration import CSPSVMDetector, RobustBaseline


# =============================================================
# CONFIGURATION
# =============================================================

class BCIConfig:
    """Central configuration for BCI ERD system"""
    
    # Buffer settings
    MAIN_BUFFER_DURATION = 10.0      # seconds
    BASELINE_BUFFER_DURATION = 5.0  # seconds (≤ main buffer)
    OPERATING_WINDOW_DURATION = 1.0  # seconds
    WINDOW_OVERLAP = 0.5            # overlap fraction (0.5 = 50% overlap)
    
    # ERD detection settings
    # ERD_CHANNELS = ['C3','FC3', 'CP3', 'C1', 'C5', 'FC5', 'FC1', 'CP5', 'CP1']
    ERD_CHANNELS = ['C3']
    ERD_BAND = (8, 20)  # mu band in Hz
    ERD_THRESHOLD = -30.0  # percent (negative for desynchronization)
    AVERAGE_ERDS = True  # average electrodes (vs individual ERD check)
    
    # Moving average settings
    BASELINE_MA_WINDOWS = 1      # number of windows to average for baseline
    ERD_MA_WINDOWS = 2           # number of windows to average for ERD detection
    USE_MOVING_AVERAGE = False    # enable moving average smoothing
    
    # Baseline settings
    BASELINE_METHOD = 'robust'  # 'standard' or 'robust'
    ROBUST_METHOD = 'trimmed_mean'  # for robust baseline 'median', 'trimmed_mean', 'mean'
    SLIDING_BASELINE = True     # Use sliding baseline that updates continuously
    SLIDING_BASELINE_DURATION = 5.0  # seconds of data to use for sliding baseline
    
    # ERD Calculation Method
    ERD_CALCULATION_METHOD = 'welch'  # Options: 'percentage', 'db_correction', 'welch', 'consecutive'
    BASELINE_CALCULATION_METHOD = 'robust'  # Options: 'standard', 'robust', 'welch'
    
    # Consecutive window detection
    MIN_CONSECUTIVE_WINDOWS = 2  # Minimum consecutive windows for ERD confirmation
    CONSECUTIVE_WINDOWS = 3  # Buffer size for consecutive window tracking
    
    # dB correction scaling
    DB_SCALE_FACTOR = 10  # Scaling factor for dB to percentage-like display

    # Preprocessing settings
    USE_CAR = True  # Common Average Reference
    ARTIFACT_METHOD = 'threshold'  # 'ica' or 'threshold'
    ARTIFACT_THRESHOLD = 1000.0  # microvolts
    SPATIAL_METHOD = ''  # 'laplacian' or ''
    
    # Connection settings
    VIRTUAL = True
    LIVESTREAM_IP = "169.254.1.147"
    LIVESTREAM_PORT = 51244
    
    # Output settings
    BROADCAST_ENABLED = False
    VERBOSE = False
    GUI_ENABLED = True
    MINIMAL_GUI = False  # Minimal GUI without plotting
    SHOW_ANNOTATIONS = True
    
    # Session settings
    SESSION_DURATION = 6000  # seconds
    
    # Advanced features
    USE_CSP_SVM = False
    AUTO_TRAIN_CSP = True
    CSP_MULTIBAND = True
    AUTO_TRAIN_ANNOTATIONS = False  # Auto-train from livestream annotations
    TRAINING_ANNOTATIONS = {
        'rest': ['Stimulus/S  4', 'S  4'],
        'mi': ['Stimulus/S  3', 'S  3']
    }


# =============================================================
# PREPROCESSING
# =============================================================

class Preprocessor:
    """Preprocessing pipeline for EEG data"""
    
    def __init__(self, sampling_freq, channel_names):
        self.fs = sampling_freq
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self._init_filters()
        
    def _init_filters(self):
        """Initialize bandpass filters"""
        nyquist = self.fs / 2
        
        # Mu band filter
        self.mu_b, self.mu_a = butter(4, 
                                     [BCIConfig.ERD_BAND[0]/nyquist, 
                                      BCIConfig.ERD_BAND[1]/nyquist], 
                                     btype='band')
        
        # Artifact removal filter
        self.artifact_b, self.artifact_a = butter(4, 
                                                 [0.5/nyquist, 
                                                  min(40/nyquist, 0.99)], 
                                                 btype='band')
    
    def preprocess_data(self, data, selected_channels=None):
        """Full preprocessing pipeline"""
        # Step 1: Basic filtering
        filtered_data = self._causal_filter(data)
        
        # Step 2: Artifact rejection
        ar_data = self._reject_artifacts(filtered_data)
        
        # Step 3: CAR
        if BCIConfig.USE_CAR:
            car_data = self._apply_car(ar_data)
        else:
            car_data = ar_data
        
        # Step 4: Spatial filtering
        if BCIConfig.SPATIAL_METHOD == 'laplacian':
            spatial_data = self._laplacian_spatial_filter(car_data)
        else:
            spatial_data = car_data
        
        # Step 5: Extract ERD channels
        if selected_channels is not None:
            erd_data = spatial_data[selected_channels, :]
            return erd_data
        
        return spatial_data
    
    def _causal_filter(self, data):
        """Apply basic frequency filter"""
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered[ch, :] = filtfilt(self.artifact_b, self.artifact_a, data[ch, :])
        return filtered
    
    def _apply_car(self, data):
        """Apply Common Average Reference"""
        car_signal = np.mean(data, axis=0)
        return data - car_signal[np.newaxis, :]
    
    def _reject_artifacts(self, data):
        """Simple threshold-based artifact rejection"""
        if BCIConfig.ARTIFACT_METHOD != 'threshold':
            return data
            
        clean_data = data.copy()
        artifact_mask = np.abs(data) > BCIConfig.ARTIFACT_THRESHOLD
        
        for ch in range(data.shape[0]):
            if np.any(artifact_mask[ch, :]):
                artifact_indices = np.where(artifact_mask[ch, :])[0]
                for idx in artifact_indices:
                    if 0 < idx < len(data[ch]) - 1:
                        clean_data[ch, idx] = (data[ch, idx-1] + data[ch, idx+1]) / 2
        
        return clean_data
    
    def _laplacian_spatial_filter(self, data):
        """Apply Laplacian spatial filter"""
        laplacian_data = data.copy()
        
        # Simplified Laplacian for motor channels
        for i, ch_name in enumerate(self.channel_names):
            if 'C3' in ch_name:
                neighbors = ['FC3', 'CP3', 'C1', 'C5', 'FC5', 'FC1', 'CP5', 'CP1']
            else:
                continue
            
            if neighbors:
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
    
    def calculate_band_power(self, data):
        """Calculate power in mu band"""
        powers = []
        
        for ch in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch, :], 
                                     self.fs, 
                                     nperseg=min(len(data[ch, :]), int(self.fs)))
            
            mu_mask = (freqs >= BCIConfig.ERD_BAND[0]) & (freqs <= BCIConfig.ERD_BAND[1])
            mu_power = np.mean(psd[mu_mask])
            powers.append(mu_power)
        
        return np.array(powers)


# =============================================================
# ERD DETECTION
# =============================================================

class ERDDetectionSystem:
    """ERD Detection with integrated preprocessing, robust baseline, and moving average"""
    
    def __init__(self, sampling_freq, channel_names):
        self.fs = sampling_freq
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        
        # Initialize components
        self.preprocessor = Preprocessor(sampling_freq, channel_names)
        self.erd_channel_indices = self._find_erd_channels()
        
        # Baseline management
        self.baseline_buffer = deque(maxlen=int(BCIConfig.BASELINE_BUFFER_DURATION * self.fs))
        self.baseline_power = None
        self.baseline_calculated = False

        # ERD Calc
        self.erd_threshold = BCIConfig.ERD_THRESHOLD
        
        # Moving average buffers
        self.baseline_ma_buffer = deque(maxlen=BCIConfig.BASELINE_MA_WINDOWS)
        self.erd_ma_buffer = deque(maxlen=BCIConfig.ERD_MA_WINDOWS)
        self.smoothed_baseline_power = None
        
        # Statistics
        self.erd_history = deque(maxlen=100)
        self.last_erd_values = {}
        
        # Consecutive window tracking for robust detection
        self.consecutive_window_buffer = deque(maxlen=BCIConfig.CONSECUTIVE_WINDOWS if hasattr(BCIConfig, 'CONSECUTIVE_WINDOWS') else 3)
        self.power_decrease_count = 0
        
        # CSP+SVM detector (if enabled)
        self.csp_svm_detector = None
        if BCIConfig.USE_CSP_SVM:
            self._init_csp_svm()
    
    def _find_erd_channels(self):
        """Find indices of ERD channels"""
        indices = []
        for target in BCIConfig.ERD_CHANNELS:
            for i, ch_name in enumerate(self.channel_names):
                if target == ch_name:
                    indices.append(i)
                    break
        
        if not indices:
            print("Warning: No ERD channels found, using first 3")
            indices = list(range(min(3, self.n_channels)))
        
        return indices
    
    def _init_csp_svm(self):
        """Initialize CSP+SVM detector"""
        self.csp_svm_detector = CSPSVMDetector(
            fs=self.fs,
            n_channels=len(self.erd_channel_indices),
            use_multiband=BCIConfig.CSP_MULTIBAND,
            autotrain = BCIConfig.AUTO_TRAIN_CSP
        )
    
    def add_to_baseline(self, data):
        """Add data to baseline buffer"""
        for i in range(data.shape[1]):
            self.baseline_buffer.append(data[:, i])
        
        if len(self.baseline_buffer) >= self.baseline_buffer.maxlen:
            if not self.baseline_calculated:
                self.calculate_baseline()
    
    def calculate_baseline(self, main_buffer=None, force=False):
        """Calculate baseline power with multiple methods"""
        if len(self.baseline_buffer) < self.baseline_buffer.maxlen:
            return False
        
        # Prepare baseline data
        if main_buffer is not None:
            baseline_data = np.array(main_buffer)
        else:
            if len(self.baseline_buffer) == 0:
                return False
            baseline_data = np.array(list(self.baseline_buffer)).T

        if baseline_data.shape[0] != self.n_channels:
            return False
        
        try:
            # Preprocess baseline data
            baseline_data_processed = self.preprocessor.preprocess_data(baseline_data, self.erd_channel_indices)
            
            # Calculate baseline power based on method - default to 'standard' which uses mean
            baseline_method = getattr(BCIConfig, 'BASELINE_CALCULATION_METHOD', 'standard')
            
            if baseline_method == 'robust':
                current_baseline_power = self._calculate_robust_baseline(baseline_data_processed)
            elif baseline_method == 'welch':
                current_baseline_power = self._calculate_welch_baseline(baseline_data_processed)
            else:
                # Standard baseline calculation using mean
                current_baseline_power = self.preprocessor.calculate_band_power(baseline_data_processed)
            
            self.baseline_power = current_baseline_power
            self.baseline_calculated = True
            
            if BCIConfig.VERBOSE:
                print(f"Baseline calculated using {baseline_method} method")
                for i, ch_idx in enumerate(self.erd_channel_indices):
                    print(f"  {self.channel_names[ch_idx]}: {self.baseline_power[i]:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Error calculating baseline: {e}")
            return False
    
    def _calculate_robust_baseline(self, data):
        """Calculate robust baseline power"""
        segment_size = int(self.fs * 0.5)  # 0.5-second segments
        n_segments = data.shape[1] // segment_size
        
        if n_segments < 2:
            return self.preprocessor.calculate_band_power(data)
        
        segment_powers = []
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size
            segment = data[:, start:end]
            power = self.preprocessor.calculate_band_power(segment)
            segment_powers.append(power)
        
        segment_powers = np.array(segment_powers)
        
        # Apply robust method - default to mean
        if BCIConfig.ROBUST_METHOD == 'trimmed_mean':
            # Remove top and bottom 10%
            return self._trimmed_mean(segment_powers.T, trim_percent=0.1)
        elif BCIConfig.ROBUST_METHOD == 'median':
            return np.median(segment_powers, axis=0)
        else:
            return np.mean(segment_powers, axis=0)
    
    def _calculate_welch_baseline(self, data):
        """Calculate baseline power using Welch's method"""
        powers = []
        
        for ch in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch, :], 
                              fs=self.fs, 
                              nperseg=min(len(data[ch, :]), int(self.fs)))
            
            # Find frequency band
            band_mask = (freqs >= BCIConfig.ERD_BAND[0]) & (freqs <= BCIConfig.ERD_BAND[1])
            band_power = np.mean(psd[band_mask])
            powers.append(band_power)
        
        return np.array(powers)
    
    def _trimmed_mean(self, data, trim_percent=0.1):
        """Calculate trimmed mean"""
        sorted_data = np.sort(data, axis=0)
        n_trim = int(data.shape[0] * trim_percent)
        if n_trim > 0:
            trimmed = sorted_data[n_trim:-n_trim, :]
        else:
            trimmed = sorted_data
        return np.mean(trimmed, axis=0)
    
    def detect_erd(self, data, main_buffer=None):
        """Detect ERD with multiple calculation methods"""
        # Use sliding baseline if enabled
        if BCIConfig.SLIDING_BASELINE and main_buffer is not None:
            sliding_samples = int(BCIConfig.SLIDING_BASELINE_DURATION * self.fs)
            
            if len(main_buffer) >= sliding_samples:
                window_size = data.shape[1]
                if len(main_buffer) > sliding_samples + window_size:
                    baseline_start = -(sliding_samples + window_size)
                    baseline_end = -window_size
                    recent_samples = list(main_buffer)[baseline_start:baseline_end]
                else:
                    recent_samples = list(main_buffer)[:-window_size]
                
                baseline_data = np.array(recent_samples).T
                self.calculate_baseline(force=True, main_buffer=deque(baseline_data))        
        # Check if we have a valid baseline
        if not self.baseline_calculated or self.baseline_power is None:
            return False, {}, 0.0
        
        try:
            # Preprocess data
            erd_data = self.preprocessor.preprocess_data(data, self.erd_channel_indices)
            
            # Choose ERD calculation method
            erd_method = getattr(BCIConfig, 'ERD_CALCULATION_METHOD', 'percentage')
            
            if erd_method == 'db_correction':
                current_erd_values = self._calculate_erd_db_correction(erd_data)
            elif erd_method == 'welch':
                current_erd_values = self._calculate_erd_welch(erd_data)
            elif erd_method == 'consecutive':
                current_erd_values = self._calculate_erd_consecutive(erd_data)
            else:
                # Standard percentage method
                current_erd_values = self._calculate_erd_percentage(erd_data)
            
            # Calculate average ERD across channels
            current_avg_erd = np.mean(list(current_erd_values.values())) if current_erd_values else 0.0
            current_erd_values['avg'] = current_avg_erd
            
            # Apply moving average if enabled
            if BCIConfig.USE_MOVING_AVERAGE:
                self.erd_ma_buffer.append(current_erd_values)
                
                # Calculate smoothed ERD values
                smoothed_erd_values = {}
                
                if len(self.erd_ma_buffer) > 0:
                    for ch_name in current_erd_values.keys():
                        values = [window.get(ch_name, 0) for window in self.erd_ma_buffer]
                        smoothed_erd_values[ch_name] = np.mean(values)
                    
                    erd_values = smoothed_erd_values
                    avg_erd = smoothed_erd_values.get('avg', 0)
                else:
                    erd_values = current_erd_values
                    avg_erd = current_avg_erd
            else:
                erd_values = current_erd_values
                avg_erd = current_avg_erd
            
            # Store in history
            self.erd_history.append(avg_erd)
            self.last_erd_values = erd_values
            
            # Determine detection based on configuration
            # ERD is negative for desynchronization, so check if below threshold
            detected_channels = [ch for ch, erd in erd_values.items() 
                               if ch != 'avg' and erd < self.erd_threshold]
            
            if BCIConfig.AVERAGE_ERDS:
                detected = avg_erd < self.erd_threshold
            else:
                detected = len(detected_channels) > 0
            
            # CSP+SVM detection if available
            if self.csp_svm_detector and self.csp_svm_detector.is_trained:
                csp_pred, csp_conf = self.csp_svm_detector.predict(erd_data)
                if csp_pred is not None:
                    combined_conf = csp_conf
                    detected = combined_conf > 0.5
                    erd_values['csp_conf'] = csp_conf * 100
                    erd_values['combined_conf'] = combined_conf * 100
                    
                    return detected, erd_values, combined_conf * 100
            
            # Return confidence as absolute value of avg_erd
            confidence = abs(avg_erd)
            
            return detected, erd_values, confidence
            
        except Exception as e:
            if BCIConfig.VERBOSE:
                print(f"Error in ERD detection: {e}")
            return False, {}, 0.0
    
    def _calculate_erd_percentage(self, data):
        """Standard percentage-based ERD calculation with corrected formula"""
        current_power = self.preprocessor.calculate_band_power(data)
        
        erd_values = {}
        for i, ch_idx in enumerate(self.erd_channel_indices):
            if self.baseline_power[i] > 0:
                # Corrected formula: (activity - baseline) / baseline * 100
                # ERD will be negative for desynchronization
                erd_percent = ((current_power[i] - self.baseline_power[i]) / 
                              self.baseline_power[i]) * 100
                ch_name = self.channel_names[ch_idx]
                erd_values[ch_name] = erd_percent
        
        return erd_values
    
    def _calculate_erd_db_correction(self, data):
        """Calculate ERD using dB correction method"""
        current_power = self.preprocessor.calculate_band_power(data)
        
        erd_values = {}
        for i, ch_idx in enumerate(self.erd_channel_indices):
            if self.baseline_power[i] > 0 and current_power[i] > 0:
                # ERD in dB: 10 * log10(current/baseline)
                # Negative values indicate desynchronization
                erd_db = 10 * np.log10(current_power[i] / self.baseline_power[i])
                # Convert to percentage scale
                erd_percent = erd_db * BCIConfig.DB_SCALE_FACTOR
                ch_name = self.channel_names[ch_idx]
                erd_values[ch_name] = erd_percent
            else:
                ch_name = self.channel_names[ch_idx]
                erd_values[ch_name] = 0.0
        
        return erd_values
    
    def _calculate_erd_welch(self, data):
        """Calculate ERD using Welch's method for current window"""
        erd_values = {}
        
        for i, ch_idx in enumerate(self.erd_channel_indices):
            # Calculate PSD for current window
            freqs, psd = signal.welch(data[i, :], 
                              fs=self.fs, 
                              nperseg=min(len(data[i, :]), int(self.fs/2)))
            
            # Get power in ERD band
            band_mask = (freqs >= BCIConfig.ERD_BAND[0]) & (freqs <= BCIConfig.ERD_BAND[1])
            current_power = np.mean(psd[band_mask])
            
            if self.baseline_power[i] > 0:
                # Corrected formula
                erd_percent = ((current_power - self.baseline_power[i]) / 
                              self.baseline_power[i]) * 100
                ch_name = self.channel_names[ch_idx]
                erd_values[ch_name] = erd_percent
            else:
                ch_name = self.channel_names[ch_idx]
                erd_values[ch_name] = 0.0
        
        return erd_values
    
    def _calculate_erd_consecutive(self, data):
        """Calculate ERD with consecutive window validation"""
        # Calculate current ERD
        current_erd_values = self._calculate_erd_percentage(data)
        
        # Add to consecutive window buffer
        self.consecutive_window_buffer.append(current_erd_values)
        
        # Check if we have enough windows
        min_consecutive = getattr(BCIConfig, 'MIN_CONSECUTIVE_WINDOWS', 2)
        
        if len(self.consecutive_window_buffer) < min_consecutive:
            # Not enough windows yet, return zero ERD
            return {ch: 0.0 for ch in current_erd_values.keys()}
        
        # Check for consecutive ERD detections
        validated_erd = {}
        
        for ch_name in current_erd_values.keys():
            if ch_name == 'avg':
                continue
                
            # Get ERD values for this channel across all windows
            ch_erds = [window.get(ch_name, 0) for window in self.consecutive_window_buffer]
            
            # Count how many consecutive windows show ERD (negative values below threshold)
            consecutive_count = 0
            for erd in ch_erds[-min_consecutive:]:
                if erd < BCIConfig.ERD_THRESHOLD:
                    consecutive_count += 1
            
            # Only report ERD if minimum consecutive windows show it
            if consecutive_count >= min_consecutive:
                validated_erd[ch_name] = current_erd_values[ch_name]
            else:
                validated_erd[ch_name] = 0.0
        
        return validated_erd
    
    def reset_baseline(self):
        """Reset baseline buffer and calculations"""
        self.baseline_buffer.clear()
        self.baseline_power = None
        self.baseline_calculated = False
        self.baseline_ma_buffer.clear()
        self.erd_ma_buffer.clear()
        self.smoothed_baseline_power = None
        self.consecutive_window_buffer.clear()
        self.power_decrease_count = 0


# =============================================================
# MAIN BCI SYSTEM
# =============================================================

class BCISystem:
    """Main BCI ERD System with overlap-based updates"""
    
    def __init__(self):
        # Components
        self.receiver = None
        self.erd_detector = None
        
        # Buffers
        self.main_buffer = None
        self.samples_since_update = 0
        self.update_threshold = 0
        
        # State
        self.running = False
        self.baseline_ready = False
        self.session_start_time = None
        
        # Statistics
        self.detection_count = 0
        self.sample_count = 0
        self.update_count = 0
        
        # Annotation tracking
        self.annotations = []
        self.annotation_times = []
        self.last_annotation_check = 0
        self.erd_detection_times = []

        # ERD parameters
        self.erd_threshold = BCIConfig.ERD_THRESHOLD
        
        # Communication
        self.gui_queue = queue.Queue() if BCIConfig.GUI_ENABLED else None
        
        # Keyboard control
        self.keyboard_enabled = not BCIConfig.GUI_ENABLED
        
    def initialize(self):
        """Initialize all components"""
        print("Initializing BCI ERD System...")
        
        # Initialize receiver
        self._init_receiver()
        
        # Initialize connection
        print("Establishing connection...")
        self.fs, self.ch_names, self.n_channels, _ = self.receiver.initialize_connection()
        print(f"Connected: {self.n_channels} channels at {self.fs} Hz")
        
        # Calculate buffer and update parameters
        self.main_buffer_size = int(BCIConfig.MAIN_BUFFER_DURATION * self.fs)
        self.operating_window_size = int(BCIConfig.OPERATING_WINDOW_DURATION * self.fs)
        
        # Calculate update threshold based on overlap
        self.update_threshold = int(self.operating_window_size * (1 - BCIConfig.WINDOW_OVERLAP))
        print(f"Update threshold: {self.update_threshold} samples (overlap: {BCIConfig.WINDOW_OVERLAP*100}%)")
        
        # Initialize buffers
        self.main_buffer = deque(maxlen=self.main_buffer_size)
        
        # Initialize ERD detector
        self.erd_detector = ERDDetectionSystem(self.fs, self.ch_names)
        
        # Auto-training setup
        if BCIConfig.AUTO_TRAIN_ANNOTATIONS and BCIConfig.USE_CSP_SVM:
            print("\nAuto-training from annotations enabled:")
            print(f"  REST: {', '.join(BCIConfig.TRAINING_ANNOTATIONS['rest'])}")
            print(f"  MI: {', '.join(BCIConfig.TRAINING_ANNOTATIONS['mi'])}")
        
        print("Initialization complete!")
        
        # Print keyboard controls if enabled
        if self.keyboard_enabled:
            self._print_keyboard_controls()
    
    def _update_erd_threshold(self, new_threshold):
        """Update ERD threshold"""
        self.erd_threshold = new_threshold
        self.erd_detector.erd_threshold = new_threshold
        if BCIConfig.VERBOSE:
            print(f"ERD threshold updated to: {self.erd_threshold}")

    def _init_receiver(self):
        """Initialize appropriate receiver"""
        if BCIConfig.VIRTUAL:
            self.receiver = virtual_receiver.Emulator(verbose=BCIConfig.VERBOSE, broadcast=BCIConfig.BROADCAST_ENABLED)
        else:
            self.receiver = livestream_receiver.LivestreamReceiver(
                address=BCIConfig.LIVESTREAM_IP,
                port=BCIConfig.LIVESTREAM_PORT,
                broadcast=BCIConfig.BROADCAST_ENABLED
            )
    
    def _print_keyboard_controls(self):
        """Print available keyboard controls"""
        print("\n" + "="*60)
        print("Keyboard Controls:")
        print("  b - Manual baseline calculation")
        print("  r - Reset baseline")
        print("  s - Print system status")
        print("  t - Increase threshold magnitude (-5)")
        print("  g - Decrease threshold magnitude (+5)")
        print("  a - Toggle moving average")
        print("  d - Toggle sliding baseline")
        print("  + - Increase ERD MA windows")
        print("  - - Decrease ERD MA windows")
        print("  e - Cycle ERD calculation method")
        print("  w - Cycle baseline calculation method")
        print("  0 - Start collecting REST data (CSP+SVM)")
        print("  1 - Start collecting MI data (CSP+SVM)")
        print("  9 - Stop collecting training data")
        print("  u - Toggle auto-training from annotations")
        print("  p - Print training data status")
        print("  m - Save CSP+SVM model")
        print("  l - Load CSP+SVM model")
        print("  q - Quit")
        print("="*60 + "\n")
    
    def run(self):
        """Main processing loop with overlap-based updates"""
        print("\n" + "="*60)
        print("Starting ERD Detection")
        print("="*60)
        
        self.running = True
        self.session_start_time = time.time()
        
        # Phase tracking
        phase = "COLLECTING_BASELINE"
        collection_mode = None
        
        try:
            while self.running:
                
                # Handle keyboard input
                if self.keyboard_enabled:
                    self._handle_keyboard_input(collection_mode)
                
                # Get data from receiver
                data = self.receiver.get_data()
                
                if data is None:
                    continue
                
                self.sample_count += data.shape[1]
                self.samples_since_update += data.shape[1]
                
                # Add data to main buffer
                for i in range(data.shape[1]):
                    self.main_buffer.append(data[:, i])
                
                # Phase 1: Baseline collection
                if phase == "COLLECTING_BASELINE":
                    self.erd_detector.add_to_baseline(data)
                    
                    if self.erd_detector.baseline_calculated:
                        phase = "DETECTING"
                        print("\n✓ Baseline established! Starting ERD detection...\n")
                        self.baseline_ready = True
                        self._send_gui_update({'baseline_ready': True})
                    else:
                        # Show progress
                        progress = len(self.erd_detector.baseline_buffer) / self.erd_detector.baseline_buffer.maxlen
                        if self.samples_since_update >= self.update_threshold:
                            print(f"\rCollecting baseline: {progress*100:.1f}%", end='', flush=True)
                
                # Phase 2: ERD Detection with overlap-based updates
                elif phase == "DETECTING" and len(self.main_buffer) >= self.operating_window_size:
                    
                    # Check if we should update based on overlap
                    if self.samples_since_update >= self.update_threshold:
                        
                        # Extract operating window
                        window_data = np.array(list(self.main_buffer))[-self.operating_window_size:].T
                        
                        # Collect training data if in collection mode
                        if collection_mode is not None and BCIConfig.USE_CSP_SVM:
                            erd_window = window_data[self.erd_detector.erd_channel_indices, :]
                            label = 0 if collection_mode == 'rest' else 1
                            self.erd_detector.csp_svm_detector.collect_training_data(erd_window, label)
                        
                        # Auto-train from annotations if enabled
                        elif BCIConfig.AUTO_TRAIN_ANNOTATIONS and BCIConfig.USE_CSP_SVM:
                            if self.annotations and self.erd_detector.csp_svm_detector:
                                # Check recent annotations
                                current_time = self.receiver.current_index / self.fs
                                for ann in self.annotations[-3:]:  # Check last 3 annotations
                                    ann_time = ann['time']
                                    # If annotation is 0.5-1.5 seconds before current window
                                    if current_time - 4 < ann_time < current_time:
                                        desc = ann['description']
                                        erd_window = window_data[self.erd_detector.erd_channel_indices, :]
                                        
                                        # Check if MI annotation
                                        for mi_ann in BCIConfig.TRAINING_ANNOTATIONS['mi']:
                                            if mi_ann in desc:
                                                self.erd_detector.csp_svm_detector.collect_training_data(erd_window, 1)
                                                if BCIConfig.VERBOSE:
                                                    print(f"AUTO-TRAIN: MI sample from '{desc}'")
                                                break
                                        
                                        # Check if REST annotation
                                        for rest_ann in BCIConfig.TRAINING_ANNOTATIONS['rest']:
                                            if rest_ann in desc:
                                                self.erd_detector.csp_svm_detector.collect_training_data(erd_window, 0)
                                                if BCIConfig.VERBOSE:
                                                    print(f"AUTO-TRAIN: REST sample from '{desc}'")
                                                break
                                        
                                # Check if ready to auto-train
                                if BCIConfig.AUTO_TRAIN_CSP and not self.erd_detector.csp_svm_detector.is_trained:
                                    detector = self.erd_detector.csp_svm_detector
                                    if (len(detector.training_data['rest']) >= 20 and 
                                        len(detector.training_data['mi']) >= 20):
                                        print("\nAuto-training CSP+SVM model...")
                                        if detector.train():
                                            print("Auto-training successful!")
                                            detector.save_model(f"{time.time() - self.session_start_time}.pkl")
                                            # Re-integrate with detection
                                            self._integrate_csp_svm_detection()
                        
                        # Detect ERD
                        detected, erd_values, confidence = self.erd_detector.detect_erd(window_data, self.main_buffer)
                        
                        if detected:
                            self.detection_count += 1
                            current_time = self.receiver.current_index / self.fs
                            self.erd_detection_times.append(current_time)
                            
                            if BCIConfig.BROADCAST_ENABLED:
                                self.receiver.use_classification(1)
                        
                        # Check for annotations
                        self._check_for_annotations()
                        
                        # Update display
                        self._update_display(detected, erd_values, confidence)
                        
                        # Reset sample counter
                        self.samples_since_update = 0
                        self.update_count += 1
                        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            if BCIConfig.VERBOSE:
                import traceback
                traceback.print_exc()
        finally:
            pass
            # self.cleanup()
    
    def _handle_keyboard_input(self, collection_mode):
        """Handle keyboard input for control"""
        try:
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
            else:
                # Unix/Linux
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                else:
                    return collection_mode
            
            if key == 'b':
                self.manual_baseline_calculation()
            elif key == 'r':
                self.erd_detector.reset_baseline()
                print("Baseline reset")
            elif key == 's':
                self.print_system_status()
            elif key == 't':
                # Increase threshold magnitude (more negative)
                BCIConfig.ERD_THRESHOLD = max(-100, BCIConfig.ERD_THRESHOLD - 5)
                print(f"Threshold increased to {BCIConfig.ERD_THRESHOLD}%")
            elif key == 'g':
                # Decrease threshold magnitude (less negative)
                BCIConfig.ERD_THRESHOLD = min(-10, BCIConfig.ERD_THRESHOLD + 5)
                print(f"Threshold decreased to {BCIConfig.ERD_THRESHOLD}%")
            elif key == 'a':
                BCIConfig.USE_MOVING_AVERAGE = not BCIConfig.USE_MOVING_AVERAGE
                print(f"Moving average {'enabled' if BCIConfig.USE_MOVING_AVERAGE else 'disabled'}")
            elif key == 'd':
                BCIConfig.SLIDING_BASELINE = not BCIConfig.SLIDING_BASELINE
                print(f"Sliding baseline {'enabled' if BCIConfig.SLIDING_BASELINE else 'disabled'}")
            elif key == '+':
                BCIConfig.ERD_MA_WINDOWS = min(10, BCIConfig.ERD_MA_WINDOWS + 1)
                print(f"ERD MA windows increased to {BCIConfig.ERD_MA_WINDOWS}")
            elif key == '-':
                BCIConfig.ERD_MA_WINDOWS = max(1, BCIConfig.ERD_MA_WINDOWS - 1)
                print(f"ERD MA windows decreased to {BCIConfig.ERD_MA_WINDOWS}")
            elif key == '0' and BCIConfig.USE_CSP_SVM:
                collection_mode = 'rest'
                print("Collecting REST data...")
            elif key == '1' and BCIConfig.USE_CSP_SVM:
                collection_mode = 'mi'
                print("Collecting MOTOR IMAGERY data...")
            elif key == '9':
                collection_mode = None
                print("Stopped collecting")
            elif key == 'u' and BCIConfig.USE_CSP_SVM:
                BCIConfig.AUTO_TRAIN_ANNOTATIONS = not BCIConfig.AUTO_TRAIN_ANNOTATIONS
                status = "enabled" if BCIConfig.AUTO_TRAIN_ANNOTATIONS else "disabled"
                print(f"Auto-training from annotations {status}")
            elif key == 'p' and BCIConfig.USE_CSP_SVM:
                self._print_training_status()
            elif key == 'm' and BCIConfig.USE_CSP_SVM:
                self._save_csp_model()
            elif key == 'l' and BCIConfig.USE_CSP_SVM:
                self._load_csp_model()
            elif key == 'q':
                self.running = False
            elif key == 'e':
                # Cycle through ERD calculation methods
                methods = ['percentage', 'db_correction', 'welch', 'consecutive']
                current_idx = methods.index(BCIConfig.ERD_CALCULATION_METHOD)
                BCIConfig.ERD_CALCULATION_METHOD = methods[(current_idx + 1) % len(methods)]
                print(f"ERD calculation method: {BCIConfig.ERD_CALCULATION_METHOD}")
            
            elif key == 'w':
                # Cycle through baseline calculation methods  
                methods = ['standard', 'robust', 'welch']
                current_idx = methods.index(BCIConfig.BASELINE_CALCULATION_METHOD)
                BCIConfig.BASELINE_CALCULATION_METHOD = methods[(current_idx + 1) % len(methods)]
                print(f"Baseline calculation method: {BCIConfig.BASELINE_CALCULATION_METHOD}")
                
            return collection_mode
            
        except:
            return collection_mode
    
    def _check_for_annotations(self):
        """Check for annotations from both virtual and livestream receivers"""
        if not BCIConfig.SHOW_ANNOTATIONS:
            return
        
        # Virtual receiver annotations
        if hasattr(self.receiver, 'annotation_onsets') and hasattr(self.receiver, 'current_index'):
            current_time = self.receiver.current_index / self.fs
            
            if self.receiver.annotation_onsets is not None:
                for i, onset in enumerate(self.receiver.annotation_onsets):
                    if self.last_annotation_check <= onset < current_time:
                        runtime = onset
                        desc = self.receiver.annotation_descriptions[i]
                        
                        if desc != 'Stimulus/S  1':
                            self.annotations.append({
                                'time': runtime,
                                'description': desc,
                                'sample': int(onset * self.fs)
                            })
                            self.annotation_times.append(runtime)
                            
                            print(f"\n📍 ANNOTATION: '{desc}' at t={runtime:.3f}s")
            
            self.last_annotation_check = current_time
        
        # Livestream receiver markers - handled in receiver's get_data() method
        # The markers are printed when detected in the livestream
    
    def _update_display(self, detected, erd_values, confidence):
        """Update console and GUI displays"""
        runtime = time.time() - self.session_start_time
        if BCIConfig.VIRTUAL:
            runtime = self.receiver.current_index / self.fs if hasattr(self.receiver, 'current_index') else runtime
        
        detection_rate = (self.detection_count / runtime) * 60 if runtime > 0 else 0
        update_rate = self.update_count / runtime if runtime > 0 else 0
        
        # Console output
        # if not BCIConfig.GUI_ENABLED or BCIConfig.VERBOSE:
        if True:
            avg_erd = erd_values.get('avg', 0)
            status = "DETECTED" if detected else "--------"
            
            if BCIConfig.MINIMAL_GUI or not BCIConfig.GUI_ENABLED:
                # Minimal display - single line update
                print(f"\r[{runtime:6.1f}s] Avg ERD: {avg_erd:+6.1f}% | {status} | "
                      f"Count: {self.detection_count} | Rate: {detection_rate:4.1f}/min | "
                      f"Updates: {update_rate:3.1f}/s", end='', flush=True)
            else:
                # Verbose display
                erd_str = " | ".join([f"{ch}:{erd:+6.1f}%" for ch, erd in erd_values.items() if ch != 'avg'])
                if detected:
                    print(f"[{runtime:6.1f}s] {erd_str} | {status}")
        
        # GUI update
        if self.gui_queue:
            self._send_gui_update({
                'detected': detected,
                'erd_values': erd_values,
                'runtime': runtime,
                'count': self.detection_count,
                'confidence': confidence,
                'annotations': self.annotations,
                'erd_detection_times': list(self.erd_detection_times),
                'update_rate': update_rate
            })
    
    def _send_gui_update(self, data):
        """Send update to GUI queue"""
        if self.gui_queue:
            try:
                self.gui_queue.put_nowait(data)
            except queue.Full:
                # Drop old data
                try:
                    self.gui_queue.get_nowait()
                    self.gui_queue.put_nowait(data)
                except:
                    pass
    
    def manual_baseline_calculation(self):
        """Manually trigger baseline calculation"""
        if self.erd_detector and len(self.main_buffer) > 0:
            success = self.erd_detector.calculate_baseline(main_buffer=self.main_buffer, force=True)
            if success:
                self.baseline_ready = True
                print("✓ Manual baseline calculation successful")
                self._send_gui_update({'baseline_ready': True})
            else:
                print("✗ Manual baseline calculation failed")
        else:
            print("Insufficient data for baseline calculation")
    
    def print_system_status(self):
        """Print detailed system status"""
        print("\n" + "="*50)
        print("BCI SYSTEM STATUS")
        print("="*50)
        
        runtime = time.time() - self.session_start_time if self.session_start_time else 0
        
        print(f"Runtime: {runtime:.1f}s")
        print(f"Main buffer: {len(self.main_buffer) if self.main_buffer else 0} samples")
        print(f"Baseline ready: {self.baseline_ready}")
        print(f"Total samples: {self.sample_count}")
        print(f"Updates: {self.update_count} ({self.update_count/runtime:.1f}/s)")
        print(f"ERD threshold: {BCIConfig.ERD_THRESHOLD}%")
        print(f"Detection count: {self.detection_count}")
        
        # Moving average status
        print(f"\nMoving Average:")
        print(f"  Enabled: {BCIConfig.USE_MOVING_AVERAGE}")
        if BCIConfig.USE_MOVING_AVERAGE:
            print(f"  Baseline MA windows: {BCIConfig.BASELINE_MA_WINDOWS}")
            print(f"  ERD MA windows: {BCIConfig.ERD_MA_WINDOWS}")
            if self.erd_detector:
                print(f"  Current baseline MA buffer: {len(self.erd_detector.baseline_ma_buffer)}")
                print(f"  Current ERD MA buffer: {len(self.erd_detector.erd_ma_buffer)}")
        
        print(f"\nSliding Baseline:")
        print(f"  Enabled: {BCIConfig.SLIDING_BASELINE}")
        if BCIConfig.SLIDING_BASELINE:
            print(f"  Window duration: {BCIConfig.SLIDING_BASELINE_DURATION}s")
        
        if self.erd_detector:
            print(f"\nERD channels ({len(self.erd_detector.erd_channel_indices)}):")
            for idx in self.erd_detector.erd_channel_indices:
                print(f"  {self.ch_names[idx]}")
            
            if self.erd_detector.baseline_power is not None:
                print("\nBaseline powers:")
                for i, ch_idx in enumerate(self.erd_detector.erd_channel_indices):
                    print(f"  {self.ch_names[ch_idx]}: {self.erd_detector.baseline_power[i]:.6f}")
            
            # Show last ERD values if available
            if hasattr(self.erd_detector, 'last_erd_values') and self.erd_detector.last_erd_values:
                print("\nLast ERD values:")
                for ch, erd in self.erd_detector.last_erd_values.items():
                    if ch not in ['csp_conf', 'combined_conf']:
                        print(f"  {ch}: {erd:+6.1f}%")
        
        print("="*50 + "\n")
    
    def _print_training_status(self):
        """Print CSP+SVM training status"""
        if self.erd_detector.csp_svm_detector:
            detector = self.erd_detector.csp_svm_detector
            print(f"\nTraining data collected:")
            print(f"  Rest: {len(detector.training_data['rest'])} windows")
            print(f"  MI: {len(detector.training_data['mi'])} windows")
            print(f"  Model trained: {detector.is_trained}")
            
            if detector.is_trained and hasattr(detector, 'band_performances'):
                print("\nBand performances:")
                for band, score in detector.band_performances.items():
                    print(f"  {band}: {score:.3f}")
            
            if BCIConfig.AUTO_TRAIN_CSP and not detector.is_trained:
                if (len(detector.training_data['rest']) >= 20 and 
                    len(detector.training_data['mi']) >= 20):
                    print("Auto-training CSP+SVM...")
                    detector.train()
    
    def _integrate_csp_svm_detection(self):
        """Integrate CSP+SVM detection with ERD system"""
        if not self.erd_detector.csp_svm_detector:
            return
        
        # The detect_erd method already handles CSP+SVM integration
        # This method is called after training to ensure integration is active
        print("CSP+SVM detection integrated with ERD system")
    
    def _save_csp_model(self):
        """Save CSP+SVM model"""
        if self.erd_detector.csp_svm_detector and self.erd_detector.csp_svm_detector.is_trained:
            filename = input("Enter filename to save model: ")
            if filename:
                if not filename.endswith('.pkl'):
                    filename += '.pkl'
                self.erd_detector.csp_svm_detector.save_model(filename)
        else:
            print("No trained model to save")
    
    def _load_csp_model(self):
        """Load CSP+SVM model"""
        if self.erd_detector.csp_svm_detector:
            filename = input("Enter filename to load model: ")
            if filename and os.path.exists(filename):
                self.erd_detector.csp_svm_detector.load_model(filename)
            else:
                print("File not found")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.receiver:
            self.receiver.disconnect()
        
        # Print summary
        if self.session_start_time:
            total_runtime = time.time() - self.session_start_time
            print("\n\n" + "="*60)
            print("Session Summary")
            print("="*60)
            print(f"Total Runtime:     {total_runtime:.1f} seconds")
            print(f"Samples Processed: {self.sample_count}")
            print(f"Updates:           {self.update_count} ({self.update_count/total_runtime:.1f}/s)")
            print(f"ERD Detections:    {self.detection_count}")
            print(f"Detection Rate:    {self.detection_count/total_runtime*60:.1f} per minute")
            
            # Annotation summary
            if self.annotations and BCIConfig.SHOW_ANNOTATIONS:
                print(f"\nAnnotations Found: {len(self.annotations)}")
                print("\nAnnotation-Detection Correlation:")
                print("-" * 40)
                
                for ann in self.annotations:
                    if self.erd_detection_times:
                        closest_det = min(self.erd_detection_times, 
                                        key=lambda x: abs(x - ann['time']))
                        time_diff = closest_det - ann['time']
                        
                        if abs(time_diff) < 2.0:
                            print(f"  {ann['description']:20s} at {ann['time']:6.2f}s → "
                                  f"ERD at {closest_det:6.2f}s (Δ={time_diff:+.2f}s)")
                        else:
                            print(f"  {ann['description']:20s} at {ann['time']:6.2f}s → "
                                  f"No ERD within 2s")
            
            print("="*60)


# =============================================================
# MAIN ENTRY POINT
# =============================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete BCI ERD Detection System")
    
    # Connection options
    parser.add_argument('--virtual', action='store_true', default=BCIConfig.VIRTUAL,
                       help="Use virtual receiver")
    parser.add_argument('--broadcast', action='store_true', default=BCIConfig.BROADCAST_ENABLED,
                       help="Broadcast classification")
    parser.add_argument('--ip', default=BCIConfig.LIVESTREAM_IP,
                       help="Livestream IP address")
    parser.add_argument('--port', type=int, default=BCIConfig.LIVESTREAM_PORT,
                       help="Livestream port")
    
    # GUI options
    parser.add_argument('--no-gui', action='store_true',
                       help="Run without GUI (keyboard control)")
    parser.add_argument('--minimal-gui', action='store_true',
                       help="Use minimal GUI without plotting")
    
    # Detection options
    parser.add_argument('--threshold', type=float, default=BCIConfig.ERD_THRESHOLD,
                       help="ERD detection threshold (negative for desynchronization)")
    parser.add_argument('--overlap', type=float, default=BCIConfig.WINDOW_OVERLAP,
                       help="Window overlap (0.0-0.9)")
    parser.add_argument('--window', type=float, default=BCIConfig.OPERATING_WINDOW_DURATION,
                       help="Operating window duration (seconds)")
    
    # Moving average options
    parser.add_argument('--no-ma', action='store_true',
                       help="Disable moving average")
    parser.add_argument('--baseline-ma', type=int, default=BCIConfig.BASELINE_MA_WINDOWS,
                       help="Baseline moving average windows")
    parser.add_argument('--erd-ma', type=int, default=BCIConfig.ERD_MA_WINDOWS,
                       help="ERD moving average windows")
    
    # Advanced options
    parser.add_argument('--csp-svm', action='store_true',
                       help="Enable CSP+SVM detection")
    parser.add_argument('--auto-train', action='store_true',
                       help="Enable automatic training from annotations")
    parser.add_argument('--multiband', action='store_true', default=BCIConfig.CSP_MULTIBAND,
                       help="Use multiband CSP analysis")
    parser.add_argument('--robust-baseline', action='store_true',
                       help="Use robust baseline calculation")
    parser.add_argument('--sliding-baseline', action='store_true', default = BCIConfig.SLIDING_BASELINE,
                       help="Use sliding baseline that updates continuously")
    parser.add_argument('--sliding-duration', type=float, default=BCIConfig.SLIDING_BASELINE_DURATION,
                       help="Duration for sliding baseline (seconds)")
    parser.add_argument('--verbose', action='store_true',
                       help="Verbose output")
    parser.add_argument('--erd-method', choices=['percentage', 'db_correction', 'welch', 'consecutive'],
                   default='percentage', help="ERD calculation method")
    parser.add_argument('--baseline-method', choices=['standard', 'robust', 'welch'],
                    default='standard', help="Baseline calculation method")
    parser.add_argument('--consecutive-windows', type=int, default=2,
                    help="Minimum consecutive windows for ERD detection")
    
    args = parser.parse_args()
    
    # Update configuration
    BCIConfig.VIRTUAL = args.virtual
    BCIConfig.LIVESTREAM_IP = args.ip
    BCIConfig.LIVESTREAM_PORT = args.port
    BCIConfig.GUI_ENABLED = not args.no_gui
    BCIConfig.MINIMAL_GUI = args.minimal_gui
    BCIConfig.ERD_THRESHOLD = args.threshold
    BCIConfig.WINDOW_OVERLAP = args.overlap
    BCIConfig.OPERATING_WINDOW_DURATION = args.window
    BCIConfig.USE_CSP_SVM = args.csp_svm
    BCIConfig.AUTO_TRAIN_ANNOTATIONS = args.auto_train
    BCIConfig.CSP_MULTIBAND = args.multiband
    BCIConfig.BASELINE_METHOD = 'robust' if args.robust_baseline else 'standard'
    BCIConfig.VERBOSE = args.verbose
    BCIConfig.USE_MOVING_AVERAGE = not args.no_ma
    BCIConfig.BASELINE_MA_WINDOWS = args.baseline_ma
    BCIConfig.ERD_MA_WINDOWS = args.erd_ma
    BCIConfig.SLIDING_BASELINE = args.sliding_baseline
    BCIConfig.SLIDING_BASELINE_DURATION = args.sliding_duration
    BCIConfig.BROADCAST_ENABLED = args.broadcast
    BCIConfig.ERD_CALCULATION_METHOD = args.erd_method
    BCIConfig.BASELINE_CALCULATION_METHOD = args.baseline_method
    BCIConfig.MIN_CONSECUTIVE_WINDOWS = args.consecutive_windows
    
    # Validate overlap
    if not 0 <= BCIConfig.WINDOW_OVERLAP < 1:
        print("Error: Overlap must be between 0 and 0.99")
        return
    
    # Print configuration
    print("BCI ERD System Configuration")
    print("="*60)
    print(f"Mode:              {'Virtual' if args.virtual else 'Livestream'}")
    print(f"GUI:               {'Disabled' if args.no_gui else ('Minimal' if args.minimal_gui else 'Full')}")
    print(f"Threshold:         {BCIConfig.ERD_THRESHOLD}%")
    print(f"Window:            {BCIConfig.OPERATING_WINDOW_DURATION}s")
    print(f"Overlap:           {BCIConfig.WINDOW_OVERLAP*100}%")
    print(f"Baseline:          {BCIConfig.BASELINE_METHOD}")
    print(f"Broadcast:         {BCIConfig.BROADCAST_ENABLED}")
    if BCIConfig.SLIDING_BASELINE:
        print(f"  Sliding:         Yes ({BCIConfig.SLIDING_BASELINE_DURATION}s window)")
    print(f"Moving Average:    {'Enabled' if BCIConfig.USE_MOVING_AVERAGE else 'Disabled'}")
    if BCIConfig.USE_MOVING_AVERAGE:
        print(f"  Baseline MA:     {BCIConfig.BASELINE_MA_WINDOWS} windows")
        print(f"  ERD MA:          {BCIConfig.ERD_MA_WINDOWS} windows")
    print(f"CSP+SVM:           {'Enabled' if BCIConfig.USE_CSP_SVM else 'Disabled'}")
    if BCIConfig.USE_CSP_SVM:
        print(f"  Auto-train:      {'Yes' if BCIConfig.AUTO_TRAIN_ANNOTATIONS else 'No'}")
        print(f"  Multiband:       {'Yes' if BCIConfig.CSP_MULTIBAND else 'No'}")
    print("="*60 + "\n")
    
    # Create and run system
    system = BCISystem()
    system.initialize()
    
    if BCIConfig.GUI_ENABLED:
        # Start GUI in separate thread
        from bci_erd_gui import BCIGUI
        gui = BCIGUI(system, minimal=BCIConfig.MINIMAL_GUI)
        gui.run()
    else:
        # Run in console mode
        system.run()


if __name__ == "__main__":
    main()