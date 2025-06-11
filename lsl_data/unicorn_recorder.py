import numpy as np
import mne
import threading
import time
from pylsl import StreamInfo, StreamOutlet, local_clock
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from scipy.io import savemat
import random
import unicornpy.UnicornPy as UnicornPy

class EEGMarkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unicorn EEG Marker GUI")
        self.root.geometry("900x700")
        self.root.minsize(900, 600)

        # Adjustable durations (in seconds) - updated defaults
        self.blank1_duration = tk.IntVar(value=0)
        self.baseline1_duration = tk.IntVar(value=3)  # Default 3 seconds
        self.motor_duration = tk.IntVar(value=2)      # Default 2 seconds
        self.blank2_duration = tk.IntVar(value=0)
        self.imagery_duration = tk.IntVar(value=2)    # Default 2 seconds
        self.rest_duration = tk.IntVar(value=2)       # Default 2 seconds

        self.total_trials = tk.IntVar(value=80)
        self.file_format = tk.StringVar(value="fif")

        # Activity selection checkboxes - all three activities equally selectable
        self.select_motor_exec = tk.BooleanVar(value=True)
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        # Randomization for all three activities
        self.randomize_activities = tk.BooleanVar(value=True)

        self.running = False
        self._acquisition_running = False
        self.current_trial = 0
        self.eeg_data = []
        self.timestamps = []
        self.markers = []
        self.start_timestamp = None
        self.recording_start_time = None

        # Unicorn device variables
        self.unicorn_device = None
        self.device_serial = None
        self.sfreq = UnicornPy.SamplingRate  # Use API constant
        self.num_channels = UnicornPy.TotalChannelsCount  # Use API constant
        self.num_eeg_channels = UnicornPy.EEGChannelsCount  # Use API constant
        self.eeg_channels = []
        self.eeg_indices = []
        
        # Marker outlet for LSL markers
        self.marker_outlet = None

        # Minimum baseline duration (in seconds) for randomization
        self.min_baseline_duration = 1.5

        # Activity sequence for trials (includes all three activities)
        self.trial_activity_sequence = []

        self.setup_ui()
        self.setup_unicorn()

    def setup_ui(self):
        # Create a canvas and a vertical scrollbar so all content is accessible
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        # Create a frame inside the canvas which will hold all other widgets
        scrollable_frame = tk.Frame(canvas, padx=20, pady=20)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure column weights for responsive layout
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        self.dynamic_labels = []
        self.dynamic_buttons = []

        # LEFT COLUMN - Device and Timing Settings
        left_row = 0
        
        # Device selection
        lbl_device = tk.Label(scrollable_frame, text="Unicorn Device")
        lbl_device.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
        self.dynamic_labels.append(lbl_device)
        
        self.device_var = tk.StringVar()
        self.device_dropdown = ttk.Combobox(scrollable_frame, textvariable=self.device_var, state="readonly")
        self.device_dropdown.grid(row=left_row, column=0, sticky="ew", pady=(0, 5), padx=(0, 10)); left_row += 1
        
        self.refresh_button = tk.Button(scrollable_frame, text="Refresh Devices", command=self.refresh_devices)
        self.refresh_button.grid(row=left_row, column=0, sticky="ew", pady=(0, 15), padx=(0, 10)); left_row += 1
        self.dynamic_buttons.append(self.refresh_button)


        # Baseline Duration
        lbl_baseline1 = tk.Label(scrollable_frame, text="Pre-Activity Baseline Duration (s)")
        lbl_baseline1.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
        self.dynamic_labels.append(lbl_baseline1)
        baseline1_scale = ttk.Scale(scrollable_frame, from_=self.min_baseline_duration, to=10, orient="horizontal",
                                    variable=self.baseline1_duration,
                                    command=lambda val: self.update_label(self.baseline1_val_label, val))
        baseline1_scale.grid(row=left_row, column=0, sticky="ew", padx=(0, 10)); left_row += 1
        self.baseline1_val_label = tk.Label(scrollable_frame, text=f"{self.baseline1_duration.get()} s")
        self.baseline1_val_label.grid(row=left_row, column=0, sticky="e", pady=(0, 10), padx=(0, 10)); left_row += 1

        # Activity Durations Section
        lbl_activity_durations = tk.Label(scrollable_frame, text="Activity Durations", font=("Arial", 10, "bold"))
        lbl_activity_durations.grid(row=left_row, column=0, sticky="w", pady=(0, 10), padx=(0, 10)); left_row += 1
        self.dynamic_labels.append(lbl_activity_durations)

        # Motor Execution Duration
        lbl_motor = tk.Label(scrollable_frame, text="Motor Execution Duration (s)")
        lbl_motor.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
        self.dynamic_labels.append(lbl_motor)
        motor_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                variable=self.motor_duration,
                                command=lambda val: self.update_label(self.motor_val_label, val))
        motor_scale.grid(row=left_row, column=0, sticky="ew", padx=(0, 10)); left_row += 1
        self.motor_val_label = tk.Label(scrollable_frame, text=f"{self.motor_duration.get()} s")
        self.motor_val_label.grid(row=left_row, column=0, sticky="e", pady=(0, 10), padx=(0, 10)); left_row += 1

        # Motor Imagery Duration
        lbl_imagery = tk.Label(scrollable_frame, text="Motor Imagery Duration (s)")
        lbl_imagery.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
        self.dynamic_labels.append(lbl_imagery)
        imagery_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                  variable=self.imagery_duration,
                                  command=lambda val: self.update_label(self.imagery_val_label, val))
        imagery_scale.grid(row=left_row, column=0, sticky="ew", padx=(0, 10)); left_row += 1
        self.imagery_val_label = tk.Label(scrollable_frame, text=f"{self.imagery_duration.get()} s")
        self.imagery_val_label.grid(row=left_row, column=0, sticky="e", pady=(0, 10), padx=(0, 10)); left_row += 1
        
        # Rest Duration
        lbl_rest = tk.Label(scrollable_frame, text="Rest Duration (s)")
        lbl_rest.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
        self.dynamic_labels.append(lbl_rest)
        rest_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                               variable=self.rest_duration,
                               command=lambda val: self.update_label(self.rest_val_label, val))
        rest_scale.grid(row=left_row, column=0, sticky="ew", padx=(0, 10)); left_row += 1
        self.rest_val_label = tk.Label(scrollable_frame, text=f"{self.rest_duration.get()} s")
        self.rest_val_label.grid(row=left_row, column=0, sticky="e", pady=(0, 10), padx=(0, 10)); left_row += 1

        # RIGHT COLUMN - Activity Selection and Session Settings
        right_row = 0

        # Activity Selection Section
        lbl_activities = tk.Label(scrollable_frame, text="Activity Selection", font=("Arial", 10, "bold"))
        lbl_activities.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_activities)

        # Activity Checkboxes
        cb_motor = tk.Checkbutton(scrollable_frame, text="Enable Motor Execution", variable=self.select_motor_exec)
        cb_motor.grid(row=right_row, column=1, sticky="w", pady=(0, 5), padx=(10, 0)); right_row += 1
        self.dynamic_buttons.append(cb_motor)

        cb_imagery = tk.Checkbutton(scrollable_frame, text="Enable Motor Imagery", variable=self.select_motor_imagery)
        cb_imagery.grid(row=right_row, column=1, sticky="w", pady=(0, 5), padx=(10, 0)); right_row += 1
        self.dynamic_buttons.append(cb_imagery)
        
        cb_rest = tk.Checkbutton(scrollable_frame, text="Enable Rest", variable=self.select_rest)
        cb_rest.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1
        self.dynamic_buttons.append(cb_rest)

        # Randomize Activities Checkbox
        cb_random_activities = tk.Checkbutton(scrollable_frame, text="Randomize Activity Order",
                                             variable=self.randomize_activities)
        cb_random_activities.grid(row=right_row, column=1, sticky="w", pady=(0, 15), padx=(10, 0)); right_row += 1
        self.dynamic_buttons.append(cb_random_activities)

        # Session Settings Section
        lbl_session = tk.Label(scrollable_frame, text="Session Settings", font=("Arial", 10, "bold"))
        lbl_session.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_session)

        # Total Trials
        lbl_trials = tk.Label(scrollable_frame, text="Total Trials")
        lbl_trials.grid(row=right_row, column=1, sticky="w", pady=(0, 2), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_trials)
        t_spinbox = ttk.Spinbox(scrollable_frame, from_=1, to=100, textvariable=self.total_trials, width=5)
        t_spinbox.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        # File Format
        lbl_format = tk.Label(scrollable_frame, text="Save Format")
        lbl_format.grid(row=right_row, column=1, sticky="w", pady=(0, 2), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_format)
        format_menu = ttk.OptionMenu(scrollable_frame, self.file_format, "fif", "fif", "mat")
        format_menu.grid(row=right_row, column=1, sticky="ew", pady=(0, 15), padx=(10, 0)); right_row += 1

        # Session Progress Section
        lbl_progress = tk.Label(scrollable_frame, text="Session Progress", font=("Arial", 10, "bold"))
        lbl_progress.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_progress)

        # Trial Label
        self.trial_label = tk.Label(scrollable_frame, text="Trial: 0 / 0")
        self.trial_label.grid(row=right_row, column=1, pady=(0, 15), padx=(10, 0)); right_row += 1

        # Control Buttons
        lbl_controls = tk.Label(scrollable_frame, text="Session Control", font=("Arial", 10, "bold"))
        lbl_controls.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_controls)

        self.start_button = tk.Button(scrollable_frame, text="Start Session", command=self.start_session)
        self.start_button.grid(row=right_row, column=1, sticky="ew", pady=(0, 5), padx=(10, 0)); right_row += 1
        self.dynamic_buttons.append(self.start_button)
        
        self.stop_button = tk.Button(scrollable_frame, text="Stop and Save", command=self.stop_session, state="disabled")
        self.stop_button.grid(row=right_row, column=1, sticky="ew", pady=(0, 15), padx=(10, 0)); right_row += 1
        self.dynamic_buttons.append(self.stop_button)

        # Logs - Right column only
        lbl_logs = tk.Label(scrollable_frame, text="Logs", font=("Arial", 10, "bold"))
        lbl_logs.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1
        self.dynamic_labels.append(lbl_logs)
        
        self.log_box = tk.Text(scrollable_frame, height=8, state="disabled", bg="#f5f5f5")
        self.log_box.grid(row=right_row, column=1, sticky="nsew", pady=(0, 10), padx=(10, 0)); right_row += 1
        scrollable_frame.grid_rowconfigure(right_row-1, weight=1)

    def update_label(self, label_widget, val):
        if label_widget in [self.baseline1_val_label]:
            fval = max(float(val), self.min_baseline_duration)
            label_widget.config(text=f"{fval:.1f} s")
        else:
            label_widget.config(text=f"{int(float(val))} s")

    def log(self, message):
        def _log_update():
            if self.log_box.winfo_exists():
                self.log_box.config(state="normal")
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.log_box.insert("end", f"[{timestamp}] {message}\n")
                self.log_box.config(state="disabled")
                self.log_box.see("end")
        self.root.after(0, _log_update)

    def update_cue(self, symbol, title):
        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.title(title)
            self.cue_label.config(text=symbol)
            self.cue_win.update()

    def setup_unicorn(self):
        """Initialize Unicorn device connection and marker stream."""
        try:
            # Log API version
            api_version = UnicornPy.GetApiVersion()
            self.log(f"Unicorn API Version: {api_version}")
            
            # Create LSL marker stream
            marker_info = StreamInfo('ExperimentMarkers', 'Markers', 1, 0, 'string', 'myuidw43536')
            self.marker_outlet = StreamOutlet(marker_info)
            self.log("Marker stream created.")
            
            # Get available devices
            self.refresh_devices()
            
        except Exception as e:
            self.log(f"Error during Unicorn setup: {e}")
            messagebox.showerror("Unicorn Error", f"Could not initialize Unicorn API:\n{e}")

    def refresh_devices(self):
        """Refresh the list of available Unicorn devices."""
        try:
            self.log("Scanning for paired Unicorn devices...")
            available_devices = UnicornPy.GetAvailableDevices(True)  # True for paired devices only
            
            if available_devices:
                # Convert device names to strings properly
                device_list = []
                for dev in available_devices:
                    if isinstance(dev, bytes):
                        device_list.append(dev.decode('utf-8'))
                    else:
                        device_list.append(str(dev))
                
                self.device_dropdown['values'] = device_list
                if device_list:
                    self.device_dropdown.current(0)
                    self.device_serial = device_list[0]
                self.log(f"Found {len(device_list)} Unicorn device(s): {', '.join(device_list)}")
            else:
                self.device_dropdown['values'] = []
                self.log("No Unicorn devices found. Make sure the device is:")
                self.log("  - Turned on")
                self.log("  - Paired via Bluetooth")
                self.log("  - Not connected to another application")
                
        except UnicornPy.DeviceException as e:
            self.log(f"Device error while scanning: {e}")
            self.device_dropdown['values'] = []
            messagebox.showwarning("Scan Error", f"Could not scan for devices: {e}")
        except Exception as e:
            self.log(f"Error refreshing devices: {e}")
            self.device_dropdown['values'] = []

    def connect_unicorn(self):
        """Connect to the selected Unicorn device."""
        try:
            if not self.device_var.get():
                raise Exception("No device selected")
                
            self.device_serial = self.device_var.get()
            self.log(f"Connecting to Unicorn device: {self.device_serial}")
            
            # Create Unicorn instance
            self.unicorn_device = UnicornPy.Unicorn(self.device_serial)
            self.log("Connected to Unicorn device successfully")
            
            # Get device information
            try:
                device_info = self.unicorn_device.GetDeviceInformation()
                self.log(f"Device Serial: {device_info.Serial}")
                self.log(f"Firmware Version: {device_info.FwVersion}")
                self.log(f"Device Version: {device_info.DeviceVersion}")
                self.log(f"Number of EEG Channels: {device_info.NumberOfEegChannels}")
                
                # Check if device version is supported
                if hasattr(UnicornPy, 'SupportedDeviceVersion'):
                    if device_info.DeviceVersion != UnicornPy.SupportedDeviceVersion:
                        self.log(f"Warning: Device version {device_info.DeviceVersion} may not be fully supported.")
                        self.log(f"Supported version: {UnicornPy.SupportedDeviceVersion}")
            except Exception as e:
                self.log(f"Warning: Could not retrieve device information: {e}")
            
            # Get and log configuration
            config = self.unicorn_device.GetConfiguration()
            self.log(f"Sampling Rate: {UnicornPy.SamplingRate} Hz")
            self.log(f"Number of channels: {len(config.Channels)}")
            
            # Configure channels - enable only EEG channels
            for i in range(len(config.Channels)):
                if i >= UnicornPy.EEGConfigIndex and i < UnicornPy.EEGConfigIndex + UnicornPy.EEGChannelsCount:
                    config.Channels[i].Enabled = True
                else:
                    config.Channels[i].Enabled = False
            
            # Apply configuration
            self.unicorn_device.SetConfiguration(config)
            
            # Update our channel information after configuration
            self.eeg_channels = []
            self.eeg_indices = []
            enabled_count = 0
            for i in range(UnicornPy.EEGChannelsCount):
                channel_idx = UnicornPy.EEGConfigIndex + i
                channel = config.Channels[channel_idx]
                if channel.Enabled:
                    self.eeg_channels.append(channel.Name)
                    self.eeg_indices.append(i)
                    enabled_count += 1
            
            self.num_acquired_channels = self.unicorn_device.GetNumberOfAcquiredChannels()
            self.log(f"Number of acquired channels: {self.num_acquired_channels}")
            self.log(f"Enabled EEG channels: {', '.join(self.eeg_channels)}")
            self.log(f"EEG channel indices: {self.eeg_indices}")
            
            return True
            
        except UnicornPy.DeviceException as e:
            self.log(f"Unicorn Device Error: {e}")
            if hasattr(e, 'code'):
                if e.code == UnicornPy.ErrorOpenDeviceFailed:
                    self.log("Could not open device. Is it already connected elsewhere?")
                elif e.code == UnicornPy.ErrorBluetoothSocketFailed:
                    self.log("Bluetooth connection failed. Check if device is on and in range.")
                elif e.code == UnicornPy.ErrorInvalidParameter:
                    self.log("Invalid parameter. Check device serial number.")
            if self.unicorn_device:
                del self.unicorn_device
                self.unicorn_device = None
            return False
        except Exception as e:
            self.log(f"Error connecting to Unicorn device: {e}")
            if self.unicorn_device:
                del self.unicorn_device
                self.unicorn_device = None
            return False

    def disconnect_unicorn(self):
        """Disconnect from the Unicorn device."""
        if self.unicorn_device:
            try:
                if hasattr(self, '_acquisition_running') and self._acquisition_running:
                    self.unicorn_device.StopAcquisition()
                del self.unicorn_device
                self.unicorn_device = None
                self.log("Disconnected from Unicorn device")
            except Exception as e:
                self.log(f"Error disconnecting from Unicorn: {e}")

    def _send_marker(self, marker_label):
        """Helper function to send marker and log it."""
        try:
            marker_timestamp = local_clock()
            self.marker_outlet.push_sample([marker_label], timestamp=marker_timestamp)
            
            # Optionally send digital output trigger for hardware synchronization
            # This sends a brief pulse on digital output 1 for each marker
            if self.unicorn_device and hasattr(self, '_acquisition_running') and self._acquisition_running:
                try:
                    # Set digital output 1 high (bit 0)
                    self.unicorn_device.SetDigitalOutputs(0x01)
                    # Brief delay
                    time.sleep(0.01)
                    # Set back to low
                    self.unicorn_device.SetDigitalOutputs(0x00)
                except:
                    pass  # Don't fail if digital outputs aren't available
            
            if self.recording_start_time is not None:
                self.markers.append((marker_label, marker_timestamp))
                relative_time = marker_timestamp - self.start_timestamp
                self.log(f"Marker '{marker_label}' sent at LSL: {marker_timestamp:.4f} (Relative: {relative_time:.4f})")
            else:
                self.markers.append((marker_label, marker_timestamp))
                self.log(f"Marker '{marker_label}' sent at LSL: {marker_timestamp:.4f}")
        except Exception as e:
            self.log(f"Error sending marker '{marker_label}': {e}")

    def start_session(self):
        # Input Validation
        if not self.device_var.get():
            self.log("Please select a Unicorn device.")
            messagebox.showerror("Error", "Please select a Unicorn device.")
            return

        # Connect to device if not already connected
        if not self.unicorn_device:
            if not self.connect_unicorn():
                messagebox.showerror("Error", "Failed to connect to Unicorn device.")
                return

        if not self.marker_outlet:
            self.log("Cannot start: Marker stream not available.")
            messagebox.showerror("Error", "Marker stream could not be created.")
            return

        n_trials = self.total_trials.get()
        if n_trials <= 0:
            messagebox.showerror("Error", "Total trials must be greater than 0.")
            return

        # Check which activities are selected
        motor_selected = self.select_motor_exec.get()
        imagery_selected = self.select_motor_imagery.get()
        rest_selected = self.select_rest.get()

        if not (motor_selected or imagery_selected or rest_selected):
            messagebox.showerror("Error", "At least one activity type must be selected.")
            return

        # Generate Activity Sequence for all trials
        self.trial_activity_sequence = []
        selected_activities = []
        
        if motor_selected:
            selected_activities.append('motor_execution')
        if imagery_selected:
            selected_activities.append('motor_imagery')
        if rest_selected:
            selected_activities.append('rest')

        if self.randomize_activities.get() and len(selected_activities) > 1:
            # Balanced randomization across all selected activities
            activities_per_type = n_trials // len(selected_activities)
            remainder = n_trials % len(selected_activities)
            
            for i, activity in enumerate(selected_activities):
                count = activities_per_type + (1 if i < remainder else 0)
                self.trial_activity_sequence.extend([activity] * count)
            
            random.shuffle(self.trial_activity_sequence)
            self.log(f"Generated balanced randomized sequence for {n_trials} trials across {len(selected_activities)} activities.")
            self.log(f"Activity distribution: {dict(zip(*np.unique(self.trial_activity_sequence, return_counts=True)))}")
        else:
            # Sequential or single activity
            if len(selected_activities) == 1:
                self.trial_activity_sequence = selected_activities * n_trials
            else:
                # Sequential cycling through activities
                for i in range(n_trials):
                    self.trial_activity_sequence.append(selected_activities[i % len(selected_activities)])
            self.log(f"Generated sequential activity sequence for {n_trials} trials.")

        # Start Recording
        self.running = True
        self._acquisition_running = False  # Initialize as False
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.eeg_data = []
        self.timestamps = []
        self.markers = []
        self.current_trial = 0
        self.trial_label.config(text=f"Trial: 0 / {n_trials}")

        # Setup Cue Window
        self.cue_win = tk.Toplevel(self.root)
        self.cue_win.attributes("-fullscreen", True)
        self.cue_win.configure(bg="black")
        self.cue_win.focus_force()
        self.cue_win.bind("<Escape>", lambda e: self.stop_session())

        self.cue_label = tk.Label(self.cue_win, text="", fg="white", bg="black", font=("Arial", 100, "bold"))
        self.cue_label.pack(expand=True)
        self.cue_win.update()

        self.root.bind("<Escape>", lambda e: self.stop_session())

        # Start Unicorn data acquisition
        try:
            self.start_timestamp = local_clock()
            self.recording_start_time = time.time()
            self.log(f"Recording started. LSL Start Time: {self.start_timestamp:.4f}")
            
            # Start acquisition on device
            self.log("Starting Unicorn data acquisition...")
            self.unicorn_device.StartAcquisition(False)  # False for non-test signal
            self._acquisition_running = True
            self.log("Unicorn acquisition started successfully")
            
            self._send_marker("session_start")

            # Start the recording thread
            self.log("Starting recording thread...")
            self.recording_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.recording_thread.start()
            self.log("Recording thread started")

            # Start the first trial after a short delay
            self.root.after(500, self.start_trial)

        except Exception as e:
            self.log(f"Error starting recording: {e}")
            import traceback
            self.log(f"Full traceback: {traceback.format_exc()}")
            self.stop_session()
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def start_trial(self):
        """Start a single trial with the appropriate activity."""
        if not self.running or self.current_trial >= len(self.trial_activity_sequence):
            self.complete_session()
            return

        current_activity = self.trial_activity_sequence[self.current_trial]
        self.log(f"Starting Trial {self.current_trial + 1}: {current_activity}")
        self.trial_label.config(text=f"Trial: {self.current_trial + 1} / {len(self.trial_activity_sequence)} ({current_activity})")

        # Pre-block blank
        self.update_cue("", "Blank")
        self._send_marker(f"trial_{self.current_trial + 1}_blank_start")
        self.root.after(int(self.blank1_duration.get() * 1000), 
                       lambda: self.start_baseline(current_activity))

    def start_baseline(self, activity_type):
        """Start the pre-activity baseline."""
        if not self.running:
            return
        
        self.update_cue("+", "Baseline")
        self._send_marker(f"trial_{self.current_trial + 1}_baseline_start")
        
        max_duration = self.baseline1_duration.get()
        duration_randomized = random.uniform(self.min_baseline_duration, max_duration)
        self.log(f"Phase: Baseline ({duration_randomized:.2f} s)")
        
        self.root.after(int(duration_randomized * 1000), 
                       lambda: self.start_activity(activity_type))

    def start_activity(self, activity_type):
        """Start the main activity (motor execution, motor imagery, or rest)."""
        if not self.running:
            return

        if activity_type == 'motor_execution':
            self.update_cue("M", "Motor Execution")
            duration = self.motor_duration.get()
            self._send_marker(f"trial_{self.current_trial + 1}_motor_execution_start")
            self.log(f"Phase: Motor Execution ({duration} s)")
        elif activity_type == 'motor_imagery':
            self.update_cue("I", "Motor Imagery")
            duration = self.imagery_duration.get()
            self._send_marker(f"trial_{self.current_trial + 1}_motor_imagery_start")
            self.log(f"Phase: Motor Imagery ({duration} s)")
        elif activity_type == 'rest':
            self.update_cue("", "Rest")
            duration = self.rest_duration.get()
            self._send_marker(f"trial_{self.current_trial + 1}_rest_start")
            self.log(f"Phase: Rest ({duration} s)")
        else:
            self.log(f"Unknown activity type: {activity_type}")
            return

        self.root.after(int(duration * 1000), self.start_inter_block_blank)

    def start_inter_block_blank(self):
        """Start the inter-block blank period."""
        if not self.running:
            return
        
        self.update_cue("", "Blank")
        self._send_marker(f"trial_{self.current_trial + 1}_blank_end")
        duration = self.blank2_duration.get()
        self.log(f"Phase: Inter-block blank ({duration} s)")
        self.root.after(int(duration * 1000), self.end_trial)

    def end_trial(self):
        """End the current trial and move to the next."""
        if not self.running:
            return
        
        self._send_marker(f"trial_{self.current_trial + 1}_end")
        self.current_trial += 1
        
        # Continue to next trial or complete session
        if self.current_trial < len(self.trial_activity_sequence):
            self.root.after(500, self.start_trial)
        else:
            self.complete_session()

    def complete_session(self):
        """Complete the entire session."""
        self.log("Session completed!")
        self.update_cue("✓", "Session Complete")
        self._send_marker("session_end")
        self.root.after(2000, self.stop_session)

    # Data Recording
    def record_loop(self):
        """Recording loop for Unicorn device."""
        self.log("EEG recording thread started.")
        
        try:
            # Get the exact number of acquired channels
            self.num_acquired_channels = self.unicorn_device.GetNumberOfAcquiredChannels()
            
            # Calculate buffer size for approximately 50ms chunks
            buffer_size = int(self.sfreq * 0.05)  # 50ms worth of samples
            
            # Allocate bytearray for receiving data as specified in API
            # Each scan has num_acquired_channels floats, each float is 4 bytes
            bytes_per_scan = self.num_acquired_channels * 4
            total_bytes = buffer_size * bytes_per_scan
            receive_buffer = bytearray(total_bytes)
            
            # Also create a numpy view for easier data manipulation
            float_buffer = np.frombuffer(receive_buffer, dtype=np.float32)
            
            self.log(f"Recording parameters:")
            self.log(f"  - Buffer size: {buffer_size} samples (50ms)")
            self.log(f"  - Acquired channels: {self.num_acquired_channels}")
            self.log(f"  - EEG channels: {self.num_eeg_channels}")
            self.log(f"  - Bytes per scan: {bytes_per_scan}")
            self.log(f"  - Total buffer bytes: {total_bytes}")
            
            data_chunks_collected = 0
            
            while self.running and self._acquisition_running:
                try:
                    # Get data from Unicorn device
                    self.unicorn_device.GetData(buffer_size, receive_buffer, len(float_buffer))
                    
                    # Create timestamps for this chunk
                    current_time = local_clock()
                    chunk_timestamps = np.linspace(
                        current_time - (buffer_size / self.sfreq),
                        current_time,
                        buffer_size,
                        endpoint=False
                    )
                    
                    # Reshape the flat buffer into scans x channels
                    data_array = float_buffer.reshape(buffer_size, self.num_acquired_channels)
                    
                    # Extract only EEG channels (first 8 channels when all are acquired)
                    eeg_data = data_array[:, :self.num_eeg_channels]
                    
                    # Convert to list format and append
                    self.eeg_data.extend(eeg_data.tolist())
                    self.timestamps.extend(chunk_timestamps)
                    
                    data_chunks_collected += 1
                    
                    # Log progress every 200 chunks (10 seconds at 50ms chunks)
                    if data_chunks_collected % 200 == 0:
                        total_samples = len(self.timestamps)
                        duration = total_samples / self.sfreq
                        self.log(f"Collected {data_chunks_collected} chunks, "
                               f"{total_samples} samples, {duration:.1f}s duration")
                    
                except UnicornPy.DeviceException as e:
                    if self.running:
                        self.log(f"Device error in recording loop: {e}")
                        if hasattr(e, 'code'):
                            if e.code == UnicornPy.ErrorConnectionProblem:
                                self.log("Connection lost. Stopping recording...")
                                self.root.after(0, self.stop_session)
                                break
                            elif e.code == UnicornPy.ErrorBufferOverflow:
                                self.log("Buffer overflow - data loss may have occurred")
                            elif e.code == UnicornPy.ErrorBufferUnderflow:
                                self.log("Buffer underflow - waiting for data")
                        time.sleep(0.5)
                except Exception as e:
                    if self.running:
                        self.log(f"Error in recording loop: {e}")
                        import traceback
                        self.log(f"Traceback: {traceback.format_exc()}")
                        time.sleep(0.5)
                        
        except Exception as e:
            self.log(f"Fatal error in recording loop setup: {e}")
            import traceback
            self.log(f"Full traceback: {traceback.format_exc()}")
        finally:
            self._acquisition_running = False
            total_chunks = len(self.eeg_data) if hasattr(self, 'eeg_data') else 0
            total_samples = len(self.timestamps) if hasattr(self, 'timestamps') else 0
            duration = total_samples / self.sfreq if total_samples > 0 else 0
            self.log(f"EEG recording thread finished. Collected {total_chunks} samples, "
                   f"{duration:.2f} seconds of data.")

    def stop_session(self):
        """Stop the recording session and save data."""
        if not self.running:
            return
        
        self.log("Stopping recording session...")
        self.running = False
        
        # Close cue window
        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.destroy()
        
        # Stop Unicorn acquisition
        try:
            if self.unicorn_device and hasattr(self, '_acquisition_running') and self._acquisition_running:
                self.log("Stopping Unicorn data acquisition...")
                self.unicorn_device.StopAcquisition()
                self._acquisition_running = False
                self.log("Unicorn acquisition stopped")
        except Exception as e:
            self.log(f"Error stopping acquisition: {e}")
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.log("Waiting for recording thread to finish...")
            self.recording_thread.join(timeout=5.0)
            if self.recording_thread.is_alive():
                self.log("Warning: Recording thread did not finish in time")
            else:
                self.log("Recording thread finished successfully")
        
        # Send final marker
        if self.marker_outlet:
            self._send_marker("session_end")
        
        # Save data
        self.save_data()
        
        # Reset UI
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.trial_label.config(text="Trial: 0 / 0")
        
        # Disconnect from device
        self.disconnect_unicorn()
        
        self.log("Session stopped and data saved.")

    def save_data(self):
        """Save the recorded EEG data and markers."""
        self.log(f"Attempting to save data...")
        self.log(f"EEG data samples collected: {len(self.eeg_data) if self.eeg_data else 0}")
        self.log(f"Timestamps collected: {len(self.timestamps) if self.timestamps else 0}")
        self.log(f"Markers collected: {len(self.markers)}")
        
        if not self.eeg_data:
            self.log("No EEG data to save.")
            messagebox.showwarning("No Data", "No EEG data was collected during the session.")
            return
        
        try:
            # Convert list data to numpy array
            eeg_array = np.array(self.eeg_data).T  # Transpose for MNE (channels x samples)
            times_lsl = np.array(self.timestamps)

            if eeg_array.shape[1] != len(times_lsl):
                self.log(f"Warning: Mismatch between EEG samples ({eeg_array.shape[1]}) and timestamps ({len(times_lsl)}).")
                min_len = min(eeg_array.shape[1], len(times_lsl))
                eeg_array = eeg_array[:, :min_len]
                times_lsl = times_lsl[:min_len]

            # Create MNE Info object
            if eeg_array.shape[0] == 8:
                ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
            else:
                ch_names = [f"EEG_{i+1}" for i in range(eeg_array.shape[0])]
                
            ch_types = ['eeg'] * len(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types=ch_types)
            info['description'] = f"Unicorn recording started at {datetime.fromtimestamp(self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')}"

            # Process Markers for MNE Annotations
            marker_labels = [m[0] for m in self.markers]
            marker_times_lsl = np.array([m[1] for m in self.markers])

            if times_lsl.size > 0:
                first_sample_time_lsl = times_lsl[0]
                sort_idx = np.argsort(marker_times_lsl)
                marker_times_lsl = marker_times_lsl[sort_idx]
                marker_labels = [marker_labels[i] for i in sort_idx]

                marker_onsets_relative = marker_times_lsl - first_sample_time_lsl
                valid_marker_indices = marker_onsets_relative >= 0
                marker_onsets_final = marker_onsets_relative[valid_marker_indices]
                marker_labels_final = [marker_labels[i] for i, valid in enumerate(valid_marker_indices) if valid]

                if len(marker_labels_final) != len(marker_labels):
                    self.log(f"Warning: {len(marker_labels) - len(marker_labels_final)} markers occurred before the first EEG sample.")

                annotations = mne.Annotations(onset=marker_onsets_final,
                                            duration=np.zeros(len(marker_labels_final)),
                                            description=marker_labels_final,
                                            orig_time=None)
                self.log(f"Created {len(marker_labels_final)} annotations for MNE.")
            else:
                self.log("Warning: No EEG timestamps recorded.")
                annotations = None

            # Create MNE Raw object
            raw = mne.io.RawArray(eeg_array, info)

            if annotations:
                try:
                    raw.set_annotations(annotations)
                except Exception as e:
                    self.log(f"Error setting annotations: {e}")

            # Save File
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"unicorn_recording_{timestamp_str}"
            save_format = self.file_format.get()
            filename = f"{filename_base}.{save_format}"

            if save_format == "fif":
                raw.save(filename, overwrite=True)
                self.log(f"Data saved to {filename}")
                
                # Verification
                try:
                    raw_check = mne.io.read_raw_fif(filename, preload=False)
                    self.log(f"Successfully verified {filename}: {raw_check.info['nchan']} channels, {raw_check.n_times} samples.")
                    if raw_check.annotations:
                        self.log(f"Verified {len(raw_check.annotations)} annotations in saved file.")
                except Exception as e:
                    self.log(f"Error verifying saved FIF file: {e}")
                    
                # Show success message
                messagebox.showinfo("Save Successful", 
                                  f"Data saved successfully!\n"
                                  f"File: {filename}\n"
                                  f"Shape: {eeg_array.shape}\n"
                                  f"Duration: {eeg_array.shape[1] / self.sfreq:.2f} seconds\n"
                                  f"Markers: {len(self.markers)}")

            elif save_format == "mat":
                mat_dict = {
                    'eeg_data': eeg_array,
                    'eeg_timestamps_lsl': times_lsl,
                    'channels': ch_names,
                    'sfreq': self.sfreq,
                    'markers': marker_labels,
                    'marker_timestamps_lsl': marker_times_lsl,
                    'info': str(info),
                    'lsl_start_time': self.start_timestamp,
                    'recording_start_time_unix': self.recording_start_time,
                    'device_serial': self.device_serial
                }
                if 'marker_onsets_relative' in locals():
                    mat_dict['marker_onsets_relative_to_first_sample'] = marker_onsets_relative

                savemat(filename, mat_dict)
                self.log(f"Data saved to {filename}")
                
                # Show success message
                messagebox.showinfo("Save Successful", 
                                  f"Data saved successfully!\n"
                                  f"File: {filename}\n"
                                  f"Shape: {eeg_array.shape}\n"
                                  f"Duration: {eeg_array.shape[1] / self.sfreq:.2f} seconds\n"
                                  f"Markers: {len(self.markers)}")

            else:
                self.log(f"Error: Unknown file format '{save_format}'")

        except Exception as e:
            self.log(f"Error during data saving: {e}")
            import traceback
            self.log(f"Full traceback: {traceback.format_exc()}")
            messagebox.showerror("Save Error", f"An error occurred while saving data:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EEGMarkerGUI(root)
    root.mainloop()