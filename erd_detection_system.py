'''
Unified ERD Detection System for BrainVision ActiChamp
Supports both real hardware and virtual emulation

Joseph Hong
'''

# =============================================================
# IMPORTS

import numpy as np
import mne
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from datetime import datetime
import os
import sys


#==========================================================================================
# RECEIVER FACTORY

class ReceiverFactory:
    """Factory to create appropriate receiver based on mode"""
    
    @staticmethod
    def create_receiver(mode='real', **kwargs):
        """
        Create receiver instance
        
        Args:
            mode: 'real' for actual hardware, 'virtual' for emulation
            **kwargs: Additional parameters for receiver initialization
        """
        if mode == 'real':
            try:
                from receivers.livestream_receiver import LivestreamReceiver
                return LivestreamReceiver(
                    address=kwargs.get('address', '169.254.1.147'),
                    port=kwargs.get('port', 51244),
                    broadcast=kwargs.get('broadcast', True)
                )
            except ImportError:
                print("Warning: livestream_receiver.py not found, switching to virtual mode")
                mode = 'virtual'
        
        if mode == 'virtual':
            try:
                from receivers.virtual_receiver import Emulator
                return Emulator(fileName=kwargs.get('fileName', 'MIT33'))
            except ImportError:
                raise ImportError("Neither livestream_receiver.py nor virtual_receiver.py found!")


#==========================================================================================
# ADAPTIVE ERD DETECTOR

class AdaptiveERDDetector:
    """
    Event-Related Desynchronization (ERD) detector with a robust hybrid baseline adaptation method.
    Works with both real and virtual receivers.
    """
    
    def __init__(self, sampling_freq=1000, buffer_size=2000):
        # Sampling parameters
        self.fs = sampling_freq
        self.buffer_size = buffer_size
        
        # Frequency bands for ERD detection
        self.freq_bands = {
            'mu': (8, 12),
            'beta': (13, 30),
            'alpha': (8, 13),
            'low_beta': (13, 20),
            'high_beta': (20, 30)
        }
        
        # Default parameters
        self.selected_channels = ['C3']
        self.selected_band = 'mu'
        self.erd_threshold = 80.0
        self.baseline_duration = 2.0
        self.window_size = 0.5
        self.overlap = 0.8
        
        # Baseline adaptation parameters
        self.adaptation_method = 'hybrid' # Hybrid is now the default and only active method
        self.adaptation_rate = 0.01
        self.sliding_baseline_window = 30.0
        self.rest_detection_threshold = 5.0
        self.min_valid_power = 1e-9
        self.max_valid_erd = 95.0
        
        # Buffers and state
        self.channel_buffers = {}
        self.baseline_power = {}
        self.baseline_history = {}
        self.is_baseline_set = False
        self.baseline_samples = []
        
        # Rest detection
        self.rest_detector_buffer = deque(maxlen=int(2 * sampling_freq))
        self.is_resting = True
        self.rest_confidence = 0.0
        
        # Error tracking
        self.error_count = {}
        self.last_valid_erd = {}
        self.channel_status = {}
        
        # Kalman filter states (kept for potential future use)
        self.kalman_states = {}
        self.kalman_covariances = {}
        
        # Detection state
        self.erd_detected = False
        self.erd_values = {}

        # Baseline power
        self.unified_baseline_power = 0.0
        
        # Initialize filters
        self._init_filters()

    def reset_baseline_collection(self):
        """
        Resets the baseline state, forcing a new collection.
        """
        print("\n*** Baseline reset triggered. Recalibrating... ***")
        self.is_baseline_set = False
        self.baseline_power = {} # Clear old baseline values
        for idx in self.selected_indices:
            # Also clear the history to start fresh
            if idx in self.baseline_history:
                self.baseline_history[idx].clear()
        
    def _init_filters(self):
        """Initialize bandpass filters for each frequency band"""
        self.filters = {}
        nyquist = self.fs / 2
        
        for band_name, (low, high) in self.freq_bands.items():
            if high < nyquist:
                b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
                self.filters[band_name] = (b, a)
            else:
                print(f"Warning: {band_name} band exceeds Nyquist frequency")
    
    def set_channels(self, channel_names, selected_indices):
        """Set the channels to monitor for ERD"""
        self.channel_names = channel_names
        self.selected_indices = selected_indices
        
        # Initialize buffers and states for selected channels
        for idx in selected_indices:
            self.channel_buffers[idx] = deque(maxlen=self.buffer_size)
            self.baseline_power[idx] = None
            self.baseline_history[idx] = deque(maxlen=int(self.sliding_baseline_window * self.fs / self.window_size))
            self.error_count[idx] = 0
            self.last_valid_erd[idx] = 0.0
            self.channel_status[idx] = "INITIALIZING"
            self.kalman_states[idx] = None
            self.kalman_covariances[idx] = 1.0
            
    def update_parameters(self, band='mu', threshold=20.0, baseline_duration=2.0, 
                         adaptation_method='hybrid', adaptation_rate=0.01):
        """Update detection parameters"""
        self.selected_band = band
        self.erd_threshold = threshold
        self.baseline_duration = baseline_duration
        self.adaptation_method = adaptation_method
        self.adaptation_rate = adaptation_rate
        
    def calculate_band_power(self, data, band='mu'):
        """Calculate band power using Welch's method with safety checks"""
        if np.var(data) < 1e-10: return self.min_valid_power
        data = data - np.mean(data)
        if np.any(np.abs(data) > 500): return self.min_valid_power
        
        try:
            b, a = self.filters[band]
            filtered = filtfilt(b, a, data)
            nperseg = min(len(data), int(self.fs * 0.5))
            freqs, psd = signal.welch(filtered, self.fs, nperseg=nperseg)
            low, high = self.freq_bands[band]
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[band_mask])
            return max(band_power, self.min_valid_power)
        except Exception as e:
            print(f"Error calculating band power: {e}")
            return self.min_valid_power
    
    def detect_rest_periods(self, erd_values):
        """Sophisticated rest detection based on ERD stability and magnitude"""
        if not erd_values: return True
        valid_erds = [v for v in erd_values.values() if not np.isnan(v)]
        if not valid_erds: return True
        avg_erd = np.mean(valid_erds)
        self.rest_detector_buffer.append(abs(avg_erd))
        
        if len(self.rest_detector_buffer) >= self.fs:
            recent_erds = list(self.rest_detector_buffer)[-self.fs:]
            mean_erd = np.mean(recent_erds)
            std_erd = np.std(recent_erds)
            
            is_low_erd = mean_erd < self.rest_detection_threshold
            is_stable = std_erd < 3.0
            
            erd_score = max(0, 1 - mean_erd / self.rest_detection_threshold)
            stability_score = max(0, 1 - std_erd / 3.0)
            self.rest_confidence = (erd_score + stability_score) / 2
            self.is_resting = is_low_erd and is_stable
        
        return self.is_resting
    
    def update_baseline_sliding(self, idx, current_power):
        """Sliding window baseline with outlier rejection"""
        self.baseline_history[idx].append(current_power)
        if len(self.baseline_history[idx]) > 10:
            self.baseline_power[idx] = np.median(list(self.baseline_history[idx]))
    
    def update_baseline_exponential(self, idx, current_power):
        """Exponential moving average with rest-gated updates"""
        if self.baseline_power[idx] is None or self.baseline_power[idx] < self.min_valid_power:
            self.baseline_power[idx] = current_power
        elif self.is_resting and self.rest_confidence > 0.7:
            adaptive_rate = self.adaptation_rate * self.rest_confidence
            self.baseline_power[idx] = ((1 - adaptive_rate) * self.baseline_power[idx] + 
                                       adaptive_rate * current_power)
    
    def calculate_erd_safe(self, reference, active, channel_idx):
        """Safe ERD calculation with comprehensive error handling"""
        if reference is None or reference <= self.min_valid_power:
            self.channel_status[channel_idx] = "INVALID_BASELINE"
            return self._get_fallback_erd(channel_idx)
        
        if reference < 0 or active < 0:
            self.channel_status[channel_idx] = "NEGATIVE_POWER"
            return self._get_fallback_erd(channel_idx)
        
        erd = ((reference - active) / reference) * 100
        
        if erd > 100: erd = 100.0
        elif erd < -200: self.channel_status[channel_idx] = "ARTIFACT_WARNING"
        else:
            self.channel_status[channel_idx] = "OK"
            self.error_count[channel_idx] = 0
            self.last_valid_erd[channel_idx] = erd
        
        return erd
    
    def _get_fallback_erd(self, channel_idx):
        """Fallback strategy for invalid ERD calculations."""
        self.error_count[channel_idx] += 1
        return 0.0
    
    def set_baseline(self):
        """Manually set baseline from current buffer data"""
        if not all(len(self.channel_buffers[idx]) >= self.baseline_duration * self.fs 
                  for idx in self.selected_indices):
            return False
        
        for idx in self.selected_indices:
            baseline_data = list(self.channel_buffers[idx])[-int(self.baseline_duration * self.fs):]
            baseline_power = self.calculate_band_power(baseline_data, self.selected_band)
            
            self.baseline_power[idx] = baseline_power
            self.baseline_history[idx].clear()
            self.baseline_history[idx].extend([baseline_power] * 10)
            self.kalman_states[idx] = baseline_power # Also reset kalman state
            self.kalman_covariances[idx] = 1.0
            self.error_count[idx] = 0
            self.channel_status[idx] = "OK"
        
        self.is_baseline_set = True
        return True
    
    def detect_erd(self, new_data):
        """
        Main ERD detection with MULTI-CHANNEL AVERAGING for improved SNR.
        """
        if new_data.shape[0] < max(self.selected_indices) + 1:
            return False, {}
        
        for idx in self.selected_indices:
            self.channel_buffers[idx].extend(new_data[idx, :])
        
        if not all(len(self.channel_buffers[idx]) >= self.window_size * self.fs 
                  for idx in self.selected_indices):
            return False, {}
        
        if not self.is_baseline_set:
            # This logic now handles both initial and manual recalibration
            if len(list(self.channel_buffers.values())[0]) >= self.baseline_duration * self.fs:
                self.set_baseline()
            else:
                # Still collecting data for the new baseline
                return False, {'ERD_Avg': 0.0, 'status': 'CALIBRATING'}
        
        # --- START OF MODIFIED LOGIC FOR AVERAGING ---
        
        active_powers = []
        baseline_powers = []
        
        for idx in self.selected_indices:
            window_data = list(self.channel_buffers[idx])[-int(self.window_size * self.fs):]
            current_power = self.calculate_band_power(window_data, self.selected_band)
            
            self.update_baseline_sliding(idx, current_power)
            if self.is_resting:
                self.update_baseline_exponential(idx, current_power)
            
            active_powers.append(current_power)
            baseline_powers.append(self.baseline_power[idx])

        if not active_powers:
            return False, {}
            
        avg_active_power = np.mean(active_powers)
        avg_baseline_power = np.mean(baseline_powers)
        
        self.unified_baseline_power = avg_baseline_power
        
        first_channel_idx = self.selected_indices[0]
        unified_erd = self.calculate_erd_safe(avg_baseline_power, avg_active_power, first_channel_idx)
        
        erd_values = {'ERD_Avg': unified_erd}
        
        self.detect_rest_periods({'unified': unified_erd})
        
        self.erd_detected = unified_erd > self.erd_threshold
        self.erd_values = erd_values
        
        return self.erd_detected, self.erd_values
    
    def get_status_info(self):
        """Get comprehensive system status, including unified baseline power."""
        status = {
            'baseline_method': self.adaptation_method,
            'is_resting': self.is_resting,
            'rest_confidence': self.rest_confidence,
            'channel_status': {
                self.channel_names[idx]: self.channel_status.get(idx, 'UNKNOWN')
                for idx in self.selected_indices if idx < len(self.channel_names)
            },
            # MODIFIED: Add the unified baseline for easy access
            'unified_baseline_power': self.unified_baseline_power
        }
        # For backward compatibility with any part of the GUI that might use this
        status['baseline_values'] = {
                self.channel_names[idx]: self.baseline_power.get(idx, 0)
                for idx in self.selected_indices if idx < len(self.channel_names)
        }
        return status

class UnifiedERDGUI:
    """
    GUI for ERD Detection System supporting both real and virtual receivers
    """
    
    def __init__(self, mode='real', **kwargs):
        self.mode = mode
        self.receiver_kwargs = kwargs
        self.receiver = None
        self.detector = None
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title(f"ERD Detection System - {mode.capitalize()} Mode")
        self.root.geometry("1400x900")
        
        # Threading
        self.update_queue = queue.Queue()
        self.running = False
        
        # Data for plotting
        self.erd_history = {ch: deque(maxlen=200) for ch in ['C3', 'C4', 'Cz']}
        self.baseline_history = {ch: deque(maxlen=200) for ch in ['C3', 'C4', 'Cz']}
        self.time_history = deque(maxlen=200)
        
        # Performance metrics
        self.detection_count = 0
        self.session_start_time = None
        
        # Virtual mode specific
        self.total_samples = 0
        self.current_sample = 0
        
        self._setup_gui()
        
    def _setup_gui(self):
        """Setup comprehensive GUI components"""
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Main Control
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main Control")
        self._setup_main_tab(main_tab)
        
        # Tab 2: Advanced Settings
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced Settings")
        self._setup_advanced_tab(advanced_tab)
        
        # Tab 3: Mode Settings
        mode_tab = ttk.Frame(notebook)
        notebook.add(mode_tab, text="Mode Settings")
        self._setup_mode_tab(mode_tab)
        
    def _setup_main_tab(self, parent):
        """Setup main control tab"""
        # Connection Panel
        conn_frame = ttk.LabelFrame(parent, text="Connection", padding="10")
        conn_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Mode indicator
        mode_label = ttk.Label(conn_frame, text=f"Mode: {self.mode.upper()}", 
                              font=("Arial", 10, "bold"))
        mode_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(conn_frame, text="Status:").grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=2, sticky=tk.W, padx=10)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect)
        self.connect_btn.grid(row=0, column=3, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", 
                                       command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=4, padx=5)
        
        # Progress bar for virtual mode
        if self.mode == 'virtual':
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(conn_frame, variable=self.progress_var,
                                              length=200, mode='determinate')
            self.progress_bar.grid(row=1, column=0, columnspan=5, pady=5, sticky=(tk.W, tk.E))
        
        # Parameters Frame
        param_frame = ttk.LabelFrame(parent, text="ERD Parameters", padding="10")
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        # Channel selection
        ttk.Label(param_frame, text="Channels:").grid(row=0, column=0, sticky=tk.W)
        self.channel_frame = ttk.Frame(param_frame)
        self.channel_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W, pady=5)
        
        # Frequency band
        ttk.Label(param_frame, text="Frequency Band:").grid(row=1, column=0, sticky=tk.W)
        self.band_var = tk.StringVar(value="mu")
        band_menu = ttk.Combobox(param_frame, textvariable=self.band_var, 
                                values=['mu', 'beta', 'alpha', 'low_beta', 'high_beta'], width=15)
        band_menu.grid(row=1, column=1, padx=5)
        band_menu.bind('<<ComboboxSelected>>', self.update_parameters)
        
        # ERD Threshold
        ttk.Label(param_frame, text="ERD Threshold (%):").grid(row=2, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=20.0)
        threshold_scale = ttk.Scale(param_frame, from_=5, to=50, orient=tk.HORIZONTAL,
                                  variable=self.threshold_var, length=200)
        threshold_scale.grid(row=2, column=1, padx=5)
        self.threshold_label = ttk.Label(param_frame, text="20.0%")
        self.threshold_label.grid(row=2, column=2)
        threshold_scale.configure(command=lambda x: self.update_threshold())
        
        # Baseline controls
        baseline_frame = ttk.LabelFrame(param_frame, text="Baseline Settings", padding="5")
        baseline_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(baseline_frame, text="Duration (s):").grid(row=0, column=0, sticky=tk.W)
        self.baseline_var = tk.DoubleVar(value=2.0)
        baseline_spin = ttk.Spinbox(baseline_frame, from_=1, to=5, increment=0.5,
                                   textvariable=self.baseline_var, width=10)
        baseline_spin.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        self.baseline_btn = ttk.Button(baseline_frame, text="Reset Baseline", 
                                     command=self.set_baseline, state=tk.DISABLED)
        self.baseline_btn.grid(row=0, column=2, padx=5)
        
        # Real-time displays
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        # ERD values display
        values_frame = ttk.LabelFrame(display_frame, text="Real-Time ERD Values", padding="10")
        values_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.erd_display = tk.Text(values_frame, width=35, height=10, font=("Courier", 10))
        self.erd_display.pack()
        
        # Detection indicator
        indicator_frame = ttk.LabelFrame(display_frame, text="Detection Status", padding="10")
        indicator_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.detection_canvas = tk.Canvas(indicator_frame, width=100, height=100)
        self.detection_canvas.pack()
        
        self.detection_label = ttk.Label(indicator_frame, text="ERD DETECTED", 
                                       font=("Arial", 14, "bold"))
        self.detection_label.pack(pady=5)
        
        self.detection_count_label = ttk.Label(indicator_frame, text="Detections: 0")
        self.detection_count_label.pack()
        
        # Classification display for virtual mode
        if self.mode == 'virtual':
            self.classification_label = ttk.Label(indicator_frame, text="Output: ---", 
                                                font=("Courier", 10))
            self.classification_label.pack(pady=5)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(parent, text="ERD Trend", padding="10")
        plot_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        
        self.fig = Figure(figsize=(12, 3), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('ERD (%)')
        self.ax.set_ylim(-30, 60)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_advanced_tab(self, parent):
        """Setup advanced settings tab"""
        # Baseline adaptation settings
        adapt_frame = ttk.LabelFrame(parent, text="Baseline Adaptation", padding="10")
        adapt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        ttk.Label(adapt_frame, text="Adaptation Method:").grid(row=0, column=0, sticky=tk.W)
        self.adapt_method_var = tk.StringVar(value="hybrid")
        methods = ["static", "sliding", "exponential", "kalman", "hybrid"]
        adapt_menu = ttk.Combobox(adapt_frame, textvariable=self.adapt_method_var,
                                values=methods, width=15)
        adapt_menu.grid(row=0, column=1, padx=5)
        adapt_menu.bind('<<ComboboxSelected>>', self.update_adaptation)
        
        ttk.Label(adapt_frame, text="Update Rate:").grid(row=1, column=0, sticky=tk.W)
        self.alpha_var = tk.DoubleVar(value=0.01)
        alpha_scale = ttk.Scale(adapt_frame, from_=0.001, to=0.05, 
                               orient=tk.HORIZONTAL, variable=self.alpha_var,
                               length=200)
        alpha_scale.grid(row=1, column=1, padx=5)
        self.alpha_label = ttk.Label(adapt_frame, text="1.0%")
        self.alpha_label.grid(row=1, column=2)
        alpha_scale.configure(command=lambda x: self.update_alpha())
        
        ttk.Label(adapt_frame, text="Rest Threshold (%):").grid(row=2, column=0, sticky=tk.W)
        self.rest_thresh_var = tk.DoubleVar(value=5.0)
        rest_scale = ttk.Scale(adapt_frame, from_=1, to=10, 
                              orient=tk.HORIZONTAL, variable=self.rest_thresh_var,
                              length=200)
        rest_scale.grid(row=2, column=1, padx=5)
        self.rest_label = ttk.Label(adapt_frame, text="5.0%")
        self.rest_label.grid(row=2, column=2)
        rest_scale.configure(command=lambda x: self.update_rest_threshold())
        
    def _setup_mode_tab(self, parent):
        """Setup mode-specific settings tab"""
        
        if self.mode == 'real':
            # Real mode settings
            real_frame = ttk.LabelFrame(parent, text="BrainVision Settings", padding="10")
            real_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
            
            ttk.Label(real_frame, text="IP Address:").grid(row=0, column=0, sticky=tk.W)
            self.ip_var = tk.StringVar(value=self.receiver_kwargs.get('address', '169.254.1.147'))
            ip_entry = ttk.Entry(real_frame, textvariable=self.ip_var, width=20)
            ip_entry.grid(row=0, column=1, padx=5)
            
            ttk.Label(real_frame, text="Port:").grid(row=1, column=0, sticky=tk.W)
            self.port_var = tk.IntVar(value=self.receiver_kwargs.get('port', 51244))
            port_entry = ttk.Entry(real_frame, textvariable=self.port_var, width=20)
            port_entry.grid(row=1, column=1, padx=5)
            
        else:
            # Virtual mode settings
            virtual_frame = ttk.LabelFrame(parent, text="Emulator Settings", padding="10")
            virtual_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
            
            ttk.Label(virtual_frame, text="Data File:").grid(row=0, column=0, sticky=tk.W)
            self.file_var = tk.StringVar(value=self.receiver_kwargs.get('fileName', 'MIT33'))
            file_entry = ttk.Entry(virtual_frame, textvariable=self.file_var, width=30)
            file_entry.grid(row=0, column=1, padx=5)
            
            ttk.Label(virtual_frame, text="Playback Speed:").grid(row=1, column=0, sticky=tk.W)
            self.speed_var = tk.DoubleVar(value=1.0)
            speed_scale = ttk.Scale(virtual_frame, from_=0.5, to=5.0, 
                                  orient=tk.HORIZONTAL, variable=self.speed_var,
                                  length=200)
            speed_scale.grid(row=1, column=1, padx=5)
            self.speed_label = ttk.Label(virtual_frame, text="1.0x")
            self.speed_label.grid(row=1, column=2)
            speed_scale.configure(command=lambda x: self.update_speed())
            
            # Loop option
            self.loop_var = tk.BooleanVar(value=True)
            loop_check = ttk.Checkbutton(virtual_frame, text="Loop playback", 
                                       variable=self.loop_var)
            loop_check.grid(row=2, column=0, columnspan=2, pady=5)
    
    def connect(self):
        """Connect to receiver (real or virtual)"""
        try:
            # Update receiver kwargs if in real mode
            if self.mode == 'real':
                self.receiver_kwargs['address'] = self.ip_var.get()
                self.receiver_kwargs['port'] = self.port_var.get()
            else:
                self.receiver_kwargs['fileName'] = self.file_var.get()
            
            # Create receiver
            self.receiver = ReceiverFactory.create_receiver(self.mode, **self.receiver_kwargs)
            
            # Initialize connection
            fs, ch_names, n_channels, _ = self.receiver.initialize_connection()
            
            # Create detector
            self.detector = AdaptiveERDDetector(sampling_freq=fs, buffer_size=int(2*fs))
            
            # Update detector
            self.detector.fs = fs
            
            # Create channel checkboxes
            self.channel_vars = []
            self.channel_checks = []
            
            # Clear previous
            for widget in self.channel_frame.winfo_children():
                widget.destroy()
            
            # Motor-related channels
            motor_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'FCz', 
                            'CP3', 'CP4', 'CPz', 'F3', 'F4', 'Fz']
            
            row, col = 0, 0
            for i, ch in enumerate(ch_names):
                var = tk.BooleanVar(value=(ch in motor_channels))
                self.channel_vars.append(var)
                
                cb = ttk.Checkbutton(self.channel_frame, text=ch, variable=var,
                                   command=self.update_selected_channels)
                cb.grid(row=row, column=col, sticky=tk.W)
                self.channel_checks.append(cb)
                
                col += 1
                if col > 6:
                    col = 0
                    row += 1
            
            # Update GUI state
            self.status_label.config(text="Connected", foreground="green")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.baseline_btn.config(state=tk.NORMAL)
            
            # Start session
            self.session_start_time = time.time()
            self.detection_count = 0
            
            # Get total samples for virtual mode
            if self.mode == 'virtual' and hasattr(self.receiver, 'raw_data'):
                self.total_samples = self.receiver.raw_data._data.shape[1]
            
            # Start data thread
            self.running = True
            self.data_thread = threading.Thread(target=self.data_acquisition_loop)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # Start GUI updates
            self.update_gui()
            
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
    
    def disconnect(self):
        """Disconnect from receiver"""
        self.running = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=1)
        
        if self.receiver:
            self.receiver.disconnect()
        
        self.status_label.config(text="Disconnected", foreground="red")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.baseline_btn.config(state=tk.DISABLED)
        
        # Reset progress bar for virtual mode
        if self.mode == 'virtual':
            self.progress_var.set(0)
    
    def data_acquisition_loop(self):
        """Background thread for data acquisition"""
        while self.running:
            try:
                # Get data
                data = self.receiver.get_data()
                
                # Handle end of file for virtual mode
                if self.mode == 'virtual' and (data is None or data.shape[1] == 0):
                    if self.loop_var.get():
                        # Reset to beginning
                        self.receiver.current_index = 0
                        continue
                    else:
                        # Stop
                        self.running = False
                        break
                
                if data is not None and data.shape[1] > 0:
                    # Update progress for virtual mode
                    if self.mode == 'virtual':
                        self.current_sample = self.receiver.current_index
                        progress = (self.current_sample / self.total_samples) * 100
                        self.progress_var.set(progress)
                    
                    # Get selected channels
                    selected_indices = [i for i, var in enumerate(self.channel_vars) 
                                      if var.get()]
                    
                    if selected_indices:
                        # Update detector
                        ch_names = [self.receiver.channel_names[i] for i in selected_indices]
                        self.detector.set_channels(self.receiver.channel_names, selected_indices)
                        
                        # Detect ERD
                        detected, erd_values = self.detector.detect_erd(data)
                        
                        # Get system status
                        status_info = self.detector.get_status_info()
                        
                        # Queue update
                        self.update_queue.put({
                            'detected': detected,
                            'erd_values': erd_values,
                            'status': status_info,
                            'timestamp': time.time()
                        })
                        
                        # Send classification if detected
                        if detected:
                            self.receiver.use_classification(1)
                            self.detection_count += 1
                
                # Sleep adjustment for virtual mode speed
                if self.mode == 'virtual' and hasattr(self, 'speed_var'):
                    time.sleep(0.02 / self.speed_var.get())  # Adjust delay based on speed
                    
            except Exception as e:
                if self.running:
                    print(f"Data acquisition error: {e}")
                    
        # Update GUI when stopped
        if self.mode == 'virtual':
            self.root.after(0, lambda: self.status_label.config(text="Playback Complete", foreground="orange"))
    
    def update_gui(self):
        """Update GUI with latest data"""
        try:
            while not self.update_queue.empty():
                data = self.update_queue.get_nowait()
                
                # Update ERD display
                self.update_erd_display(data['erd_values'], data['status'])
                
                # Update detection indicator
                self.update_indicator(data['detected'])
                
                # Update plots
                self.update_plots(data['erd_values'], data['status'], data['timestamp'])
                
                # Update classification label for virtual mode
                if self.mode == 'virtual' and hasattr(self, 'classification_label'):
                    if data['detected']:
                        self.classification_label.config(text="Output: FLEX")
                    else:
                        self.classification_label.config(text="Output: REST")
                
        except queue.Empty:
            pass
        
        # Schedule next update
        if self.running:
            self.root.after(50, self.update_gui)
    
    def update_erd_display(self, erd_values, status):
        """Update ERD values display"""
        self.erd_display.delete(1.0, tk.END)
        self.erd_display.insert(tk.END, "Channel  ERD%   Status\n")
        self.erd_display.insert(tk.END, "-" * 30 + "\n")
        
        for ch, erd in erd_values.items():
            status_ch = status['channel_status'].get(ch, 'OK')
            marker = "*" if erd > self.threshold_var.get() else " "
            
            if np.isnan(erd):
                self.erd_display.insert(tk.END, f"{marker} {ch:6s}  ---    {status_ch}\n")
            else:
                self.erd_display.insert(tk.END, f"{marker} {ch:6s} {erd:6.1f}  {status_ch}\n")
        
        # Rest status
        self.erd_display.insert(tk.END, "\n" + "-" * 30 + "\n")
        rest_status = "RESTING" if status['is_resting'] else "ACTIVE"
        confidence = status['rest_confidence'] * 100
        self.erd_display.insert(tk.END, f"State: {rest_status} ({confidence:.0f}%)\n")
        self.erd_display.insert(tk.END, f"Method: {status['baseline_method']}\n")
    
    def update_indicator(self, detected):
        """Update detection indicator"""
        self.detection_canvas.delete("all")
        color = "#00FF00" if detected else "#FF0000"
        self.detection_canvas.create_oval(10, 10, 90, 90, fill=color, outline="")
        
        if detected:
            self.detection_label.config(foreground="green")
        else:
            self.detection_label.config(foreground="gray")
        
        self.detection_count_label.config(text=f"Detections: {self.detection_count}")
    
    def update_plots(self, erd_values, status, timestamp):
        """Update trend plots"""
        # Update data
        if self.time_history:
            time_offset = timestamp - self.time_history[0]
        else:
            time_offset = 0
        
        self.time_history.append(timestamp)
        
        # Update ERD history for each channel
        for ch in ['C3', 'C4', 'Cz']:
            if ch in erd_values and not np.isnan(erd_values[ch]):
                self.erd_history[ch].append(erd_values[ch])
                if ch in status['baseline_values']:
                    self.baseline_history[ch].append(status['baseline_values'][ch])
            else:
                if self.erd_history[ch]:
                    self.erd_history[ch].append(self.erd_history[ch][-1])
                else:
                    self.erd_history[ch].append(0)
        
        # Plot ERD trends
        if len(self.time_history) > 1:
            times = np.array(self.time_history) - self.time_history[0]
            
            self.ax.clear()
            colors = {'C3': 'blue', 'C4': 'red', 'Cz': 'green'}
            
            for ch, color in colors.items():
                if self.erd_history[ch]:
                    self.ax.plot(times, list(self.erd_history[ch]), 
                               color=color, linewidth=2, label=ch)
            
            # Threshold line
            self.ax.axhline(y=self.threshold_var.get(), color='k', 
                          linestyle='--', alpha=0.5, label='Threshold')
            self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Rest periods
            if status['is_resting']:
                self.ax.axvspan(times[-1]-0.5, times[-1], alpha=0.2, color='green')
            
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('ERD (%)')
            self.ax.set_ylim(-30, 60)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc='upper right')
            
            self.canvas.draw()
    
    def update_parameters(self, event=None):
        """Update detector parameters"""
        if self.detector:
            self.detector.update_parameters(
                band=self.band_var.get(),
                threshold=self.threshold_var.get(),
                baseline_duration=self.baseline_var.get(),
                adaptation_method=self.adapt_method_var.get(),
                adaptation_rate=self.alpha_var.get()
            )
    
    def update_threshold(self):
        """Update threshold display"""
        self.threshold_label.config(text=f"{self.threshold_var.get():.1f}%")
        self.update_parameters()
    
    def update_adaptation(self, event=None):
        """Update adaptation method"""
        if self.detector:
            self.detector.adaptation_method = self.adapt_method_var.get()
    
    def update_alpha(self):
        """Update adaptation rate"""
        if self.detector:
            self.detector.adaptation_rate = self.alpha_var.get()
        self.alpha_label.config(text=f"{self.alpha_var.get()*100:.1f}%")
    
    def update_rest_threshold(self):
        """Update rest detection threshold"""
        if self.detector:
            self.detector.rest_detection_threshold = self.rest_thresh_var.get()
        self.rest_label.config(text=f"{self.rest_thresh_var.get():.1f}%")
    
    def update_speed(self):
        """Update playback speed for virtual mode"""
        self.speed_label.config(text=f"{self.speed_var.get():.1f}x")
    
    def update_selected_channels(self):
        """Update selected channels in detector"""
        if self.detector:
            selected_indices = [i for i, var in enumerate(self.channel_vars) if var.get()]
            if selected_indices:
                self.detector.set_channels(self.receiver.channel_names, selected_indices)
    
    def set_baseline(self):
        """Manually set baseline"""
        if self.detector and self.detector.set_baseline():
            messagebox.showinfo("Baseline Set", "Baseline successfully reset")
            # Clear history
            for ch in self.erd_history:
                self.erd_history[ch].clear()
                self.baseline_history[ch].clear()
            self.time_history.clear()
        else:
            messagebox.showwarning("Baseline Error", 
                                 f"Need at least {self.baseline_var.get()} seconds of data")
    
    def run(self):
        """Start GUI main loop"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Update indicator initially
        self.update_indicator(False)
        
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.disconnect()
        self.root.destroy()


#==========================================================================================
# MAIN EXECUTION

def main():
    """Main entry point with mode selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ERD Detection System')
    parser.add_argument('--mode', choices=['real', 'virtual'], default='real',
                      help='Operation mode: real hardware or virtual emulation')
    parser.add_argument('--address', default='169.254.1.147', 
                      help='BrainVision IP address (real mode)')
    parser.add_argument('--port', type=int, default=51244,
                      help='Port number (real mode)')
    parser.add_argument('--file', default='MIT33',
                      help='Data file name (virtual mode)')
    parser.add_argument('--broadcast', action='store_true', default=True,
                      help='Enable TCP broadcasting (real mode)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"ERD Detection System - {args.mode.upper()} Mode")
    print("=" * 60)
    
    # Prepare kwargs based on mode
    if args.mode == 'real':
        kwargs = {
            'address': args.address,
            'port': args.port,
            'broadcast': args.broadcast
        }
        print(f"Real Mode Configuration:")
        print(f"  Address: {args.address}:{args.port}")
        print(f"  Broadcasting: {'Enabled' if args.broadcast else 'Disabled'}")
    else:
        kwargs = {
            'fileName': args.file
        }
        print(f"Virtual Mode Configuration:")
        print(f"  Data File: {args.file}")
        print(f"  Note: Using emulated EEG data")
    
    print("=" * 60)
    
    # Create and run GUI
    gui = UnifiedERDGUI(mode=args.mode, **kwargs)
    
    print("Starting GUI...")
    gui.run()
    
    print("\nSystem shutdown complete.")


if __name__ == "__main__":
    main()