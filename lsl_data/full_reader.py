import numpy as np
import mne
import threading
import time
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet, local_clock
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from scipy.io import savemat
import random

class MultiStreamEEGRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Stream EEG Recorder")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)

        # Adjustable durations (in seconds)
        self.blank1_duration = tk.IntVar(value=0)
        self.baseline1_duration = tk.IntVar(value=3)
        self.motor_duration = tk.IntVar(value=2)
        self.blank2_duration = tk.IntVar(value=0)
        self.imagery_duration = tk.IntVar(value=2)
        self.rest_duration = tk.IntVar(value=2)

        self.total_trials = tk.IntVar(value=80)
        self.file_format = tk.StringVar(value="fif")

        # Activity selection checkboxes
        self.select_motor_exec = tk.BooleanVar(value=True)
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        # Randomization options
        self.randomize_activities = tk.BooleanVar(value=True)

        # Recording state
        self.running = False
        self.current_trial = 0
        self.all_stream_data = {}  # Will store data for each stream
        self.all_timestamps = {}   # Will store timestamps for each stream
        self.markers = []
        self.start_timestamp = None
        self.recording_start_time = None

        # Stream management
        self.available_streams = []
        self.selected_streams = {}  # stream_name: {'info': info, 'inlet': inlet, 'selected': bool}
        self.marker_outlet = None

        # Minimum baseline duration for randomization
        self.min_baseline_duration = 1.5

        # Activity sequence for trials
        self.trial_activity_sequence = []

        # Recording threads
        self.recording_threads = {}

        self.setup_ui()
        self.setup_lsl()

    def setup_ui(self):
        # Create main container with scrolling
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, padx=20, pady=20)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure grid weights
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        # LEFT COLUMN - Stream Selection and Settings
        left_row = 0
        
        # Stream Discovery and Selection
        lbl_streams = tk.Label(scrollable_frame, text="Available LSL Streams", font=("Arial", 12, "bold"))
        lbl_streams.grid(row=left_row, column=0, sticky="w", pady=(0, 10), padx=(0, 10)); left_row += 1

        self.refresh_button = tk.Button(scrollable_frame, text="Refresh Streams", command=self.refresh_streams)
        self.refresh_button.grid(row=left_row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); left_row += 1

        # Stream list with checkboxes
        self.stream_frame = tk.LabelFrame(scrollable_frame, text="Select Streams to Record", padx=10, pady=10)
        self.stream_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 15), padx=(0, 10)); left_row += 1
        
        # Stream info display
        self.stream_info_frame = tk.Frame(self.stream_frame)
        self.stream_info_frame.pack(fill="both", expand=True)

        # Timing Settings Section
        lbl_timing = tk.Label(scrollable_frame, text="Timing Settings", font=("Arial", 12, "bold"))
        lbl_timing.grid(row=left_row, column=0, sticky="w", pady=(0, 10), padx=(0, 10)); left_row += 1

        # Pre-Activity Baseline Duration
        lbl_baseline1 = tk.Label(scrollable_frame, text="Pre-Activity Baseline Duration (s)")
        lbl_baseline1.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
        baseline1_scale = ttk.Scale(scrollable_frame, from_=self.min_baseline_duration, to=10, orient="horizontal",
                                    variable=self.baseline1_duration,
                                    command=lambda val: self.update_label(self.baseline1_val_label, val))
        baseline1_scale.grid(row=left_row, column=0, sticky="ew", padx=(0, 10)); left_row += 1
        self.baseline1_val_label = tk.Label(scrollable_frame, text=f"{self.baseline1_duration.get()} s")
        self.baseline1_val_label.grid(row=left_row, column=0, sticky="e", pady=(0, 10), padx=(0, 10)); left_row += 1

        # Activity Durations
        activities = [
            ("Motor Execution Duration (s)", self.motor_duration, "motor_val_label"),
            ("Motor Imagery Duration (s)", self.imagery_duration, "imagery_val_label"),
            ("Rest Duration (s)", self.rest_duration, "rest_val_label")
        ]

        for label_text, var, label_attr in activities:
            lbl = tk.Label(scrollable_frame, text=label_text)
            lbl.grid(row=left_row, column=0, sticky="w", pady=(0, 2), padx=(0, 10)); left_row += 1
            scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                             variable=var,
                             command=lambda val, attr=label_attr: self.update_label(getattr(self, attr), val))
            scale.grid(row=left_row, column=0, sticky="ew", padx=(0, 10)); left_row += 1
            val_label = tk.Label(scrollable_frame, text=f"{var.get()} s")
            setattr(self, label_attr, val_label)
            val_label.grid(row=left_row, column=0, sticky="e", pady=(0, 10), padx=(0, 10)); left_row += 1

        # RIGHT COLUMN - Activity Selection and Session Settings
        right_row = 0

        # Activity Selection
        lbl_activities = tk.Label(scrollable_frame, text="Activity Selection", font=("Arial", 12, "bold"))
        lbl_activities.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        activities_frame = tk.Frame(scrollable_frame)
        activities_frame.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        cb_motor = tk.Checkbutton(activities_frame, text="Enable Motor Execution", variable=self.select_motor_exec)
        cb_motor.pack(anchor="w", pady=2)

        cb_imagery = tk.Checkbutton(activities_frame, text="Enable Motor Imagery", variable=self.select_motor_imagery)
        cb_imagery.pack(anchor="w", pady=2)
        
        cb_rest = tk.Checkbutton(activities_frame, text="Enable Rest", variable=self.select_rest)
        cb_rest.pack(anchor="w", pady=2)

        # Randomization
        cb_random = tk.Checkbutton(scrollable_frame, text="Randomize Activity Order",
                                  variable=self.randomize_activities)
        cb_random.grid(row=right_row, column=1, sticky="w", pady=(0, 15), padx=(10, 0)); right_row += 1

        # Session Settings
        lbl_session = tk.Label(scrollable_frame, text="Session Settings", font=("Arial", 12, "bold"))
        lbl_session.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        # Total Trials
        lbl_trials = tk.Label(scrollable_frame, text="Total Trials")
        lbl_trials.grid(row=right_row, column=1, sticky="w", pady=(0, 2), padx=(10, 0)); right_row += 1
        trials_spinbox = ttk.Spinbox(scrollable_frame, from_=1, to=200, textvariable=self.total_trials, width=10)
        trials_spinbox.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        # File Format
        lbl_format = tk.Label(scrollable_frame, text="Save Format")
        lbl_format.grid(row=right_row, column=1, sticky="w", pady=(0, 2), padx=(10, 0)); right_row += 1
        format_menu = ttk.OptionMenu(scrollable_frame, self.file_format, "fif", "fif", "mat")
        format_menu.grid(row=right_row, column=1, sticky="ew", pady=(0, 15), padx=(10, 0)); right_row += 1

        # Session Progress
        lbl_progress = tk.Label(scrollable_frame, text="Session Progress", font=("Arial", 12, "bold"))
        lbl_progress.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        self.trial_label = tk.Label(scrollable_frame, text="Trial: 0 / 0")
        self.trial_label.grid(row=right_row, column=1, pady=(0, 15), padx=(10, 0)); right_row += 1

        # Control Buttons
        lbl_controls = tk.Label(scrollable_frame, text="Session Control", font=("Arial", 12, "bold"))
        lbl_controls.grid(row=right_row, column=1, sticky="w", pady=(0, 10), padx=(10, 0)); right_row += 1

        self.start_button = tk.Button(scrollable_frame, text="Start Session", command=self.start_session)
        self.start_button.grid(row=right_row, column=1, sticky="ew", pady=(0, 5), padx=(10, 0)); right_row += 1
        
        self.stop_button = tk.Button(scrollable_frame, text="Stop and Save", command=self.stop_session, state="disabled")
        self.stop_button.grid(row=right_row, column=1, sticky="ew", pady=(0, 15), padx=(10, 0)); right_row += 1

        # Logs - spans both columns
        lbl_logs = tk.Label(scrollable_frame, text="Logs", font=("Arial", 12, "bold"))
        lbl_logs.grid(row=right_row, column=0, columnspan=2, sticky="w", pady=(0, 10)); right_row += 1
        
        self.log_box = tk.Text(scrollable_frame, height=10, state="disabled", bg="#f5f5f5")
        self.log_box.grid(row=right_row, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        scrollable_frame.grid_rowconfigure(right_row, weight=1)

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

    def setup_lsl(self):
        """Initialize LSL marker stream and discover available streams."""
        try:
            # Create LSL marker stream
            marker_info = StreamInfo('ExperimentMarkers', 'Markers', 1, 0, 'string', 'multistream_markers')
            self.marker_outlet = StreamOutlet(marker_info)
            self.log("Marker stream created.")
            
            # Discover available streams
            self.refresh_streams()
            
        except Exception as e:
            self.log(f"Error during LSL setup: {e}")
            messagebox.showerror("LSL Error", f"Could not initialize LSL:\n{e}")

    def refresh_streams(self):
        """Discover and display all available LSL streams."""
        try:
            self.log("Scanning for LSL streams...")
            
            # Resolve all available streams
            streams = resolve_streams(wait_time=2.0)
            
            # Clear previous stream display
            for widget in self.stream_info_frame.winfo_children():
                widget.destroy()
            
            self.available_streams = []
            self.selected_streams = {}
            
            if not streams:
                no_streams_label = tk.Label(self.stream_info_frame, text="No LSL streams found.", fg="red")
                no_streams_label.pack(pady=10)
                self.log("No LSL streams found.")
                return
            
            self.log(f"Found {len(streams)} LSL stream(s):")
            
            # Create checkboxes for each stream
            for i, stream in enumerate(streams):
                stream_name = stream.name()
                stream_type = stream.type()
                channel_count = stream.channel_count()
                nominal_srate = stream.nominal_srate()
                
                # Create a frame for this stream
                stream_frame = tk.Frame(self.stream_info_frame, relief="ridge", borderwidth=1, padx=10, pady=5)
                stream_frame.pack(fill="x", pady=2)
                
                # Checkbox for selection
                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(stream_frame, variable=var, 
                                        command=lambda name=stream_name, v=var: self.toggle_stream_selection(name, v.get()))
                checkbox.pack(side="left")
                
                # Stream info
                info_text = f"{stream_name} ({stream_type}) - {channel_count} channels @ {nominal_srate} Hz"
                info_label = tk.Label(stream_frame, text=info_text, font=("Arial", 9))
                info_label.pack(side="left", padx=(5, 0))
                
                # Store stream info
                self.available_streams.append(stream)
                self.selected_streams[stream_name] = {
                    'info': stream,
                    'inlet': None,
                    'selected': False,
                    'checkbox_var': var,
                    'type': stream_type,
                    'channels': channel_count,
                    'srate': nominal_srate
                }
                
                self.log(f"  {info_text}")
                
        except Exception as e:
            self.log(f"Error refreshing streams: {e}")
            messagebox.showerror("Error", f"Could not refresh streams:\n{e}")

    def toggle_stream_selection(self, stream_name, selected):
        """Handle stream selection/deselection."""
        if stream_name in self.selected_streams:
            self.selected_streams[stream_name]['selected'] = selected
            if selected:
                self.log(f"Selected stream: {stream_name}")
            else:
                self.log(f"Deselected stream: {stream_name}")

    def get_selected_streams(self):
        """Return list of selected stream names."""
        return [name for name, info in self.selected_streams.items() if info['selected']]

    def connect_selected_streams(self):
        """Create inlets for all selected streams."""
        selected = self.get_selected_streams()
        if not selected:
            return False
        
        try:
            for stream_name in selected:
                stream_info = self.selected_streams[stream_name]['info']
                inlet = StreamInlet(stream_info)
                self.selected_streams[stream_name]['inlet'] = inlet
                self.log(f"Connected to stream: {stream_name}")
            return True
        except Exception as e:
            self.log(f"Error connecting to streams: {e}")
            return False

    def disconnect_streams(self):
        """Disconnect from all streams."""
        for stream_name, stream_data in self.selected_streams.items():
            if stream_data['inlet']:
                try:
                    del stream_data['inlet']
                    stream_data['inlet'] = None
                except:
                    pass

    def _send_marker(self, marker_label):
        """Send marker to LSL stream."""
        try:
            marker_timestamp = local_clock()
            self.marker_outlet.push_sample([marker_label], timestamp=marker_timestamp)
            
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
        """Start the recording session."""
        # Validation
        selected_streams = self.get_selected_streams()
        if not selected_streams:
            messagebox.showerror("Error", "Please select at least one stream to record from.")
            return

        if not self.marker_outlet:
            messagebox.showerror("Error", "Marker stream not available.")
            return

        n_trials = self.total_trials.get()
        if n_trials <= 0:
            messagebox.showerror("Error", "Total trials must be greater than 0.")
            return

        # Check activity selection
        motor_selected = self.select_motor_exec.get()
        imagery_selected = self.select_motor_imagery.get()
        rest_selected = self.select_rest.get()

        if not (motor_selected or imagery_selected or rest_selected):
            messagebox.showerror("Error", "At least one activity type must be selected.")
            return

        # Connect to selected streams
        if not self.connect_selected_streams():
            messagebox.showerror("Error", "Failed to connect to selected streams.")
            return

        # Generate activity sequence
        self.generate_activity_sequence(n_trials, motor_selected, imagery_selected, rest_selected)

        # Initialize recording state
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Initialize data storage for each selected stream
        self.all_stream_data = {}
        self.all_timestamps = {}
        self.recording_threads = {}
        
        for stream_name in selected_streams:
            self.all_stream_data[stream_name] = []
            self.all_timestamps[stream_name] = []

        self.markers = []
        self.current_trial = 0
        self.trial_label.config(text=f"Trial: 0 / {n_trials}")

        # Setup cue window
        self.setup_cue_window()

        # Start recording
        try:
            self.start_timestamp = local_clock()
            self.recording_start_time = time.time()
            self.log(f"Recording started. LSL Start Time: {self.start_timestamp:.4f}")
            
            self._send_marker("session_start")

            # Start recording threads for each selected stream
            for stream_name in selected_streams:
                thread = threading.Thread(target=self.record_stream_loop, args=(stream_name,), daemon=True)
                self.recording_threads[stream_name] = thread
                thread.start()
                self.log(f"Started recording thread for {stream_name}")

            # Start the first trial
            self.root.after(500, self.start_trial)

        except Exception as e:
            self.log(f"Error starting recording: {e}")
            self.stop_session()
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def generate_activity_sequence(self, n_trials, motor_selected, imagery_selected, rest_selected):
        """Generate sequence of activities for trials."""
        self.trial_activity_sequence = []
        selected_activities = []
        
        if motor_selected:
            selected_activities.append('motor_execution')
        if imagery_selected:
            selected_activities.append('motor_imagery')
        if rest_selected:
            selected_activities.append('rest')

        if self.randomize_activities.get() and len(selected_activities) > 1:
            # Balanced randomization
            activities_per_type = n_trials // len(selected_activities)
            remainder = n_trials % len(selected_activities)
            
            for i, activity in enumerate(selected_activities):
                count = activities_per_type + (1 if i < remainder else 0)
                self.trial_activity_sequence.extend([activity] * count)
            
            random.shuffle(self.trial_activity_sequence)
            self.log(f"Generated balanced randomized sequence for {n_trials} trials across {len(selected_activities)} activities.")
        else:
            # Sequential or single activity
            if len(selected_activities) == 1:
                self.trial_activity_sequence = selected_activities * n_trials
            else:
                for i in range(n_trials):
                    self.trial_activity_sequence.append(selected_activities[i % len(selected_activities)])
            self.log(f"Generated sequential activity sequence for {n_trials} trials.")

    def setup_cue_window(self):
        """Setup fullscreen cue window."""
        self.cue_win = tk.Toplevel(self.root)
        self.cue_win.attributes("-fullscreen", True)
        self.cue_win.configure(bg="black")
        self.cue_win.focus_force()
        self.cue_win.bind("<Escape>", lambda e: self.stop_session())

        self.cue_label = tk.Label(self.cue_win, text="", fg="white", bg="black", font=("Arial", 100, "bold"))
        self.cue_label.pack(expand=True)
        self.cue_win.update()

        self.root.bind("<Escape>", lambda e: self.stop_session())

    def update_cue(self, symbol, title):
        """Update cue window display."""
        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.title(title)
            self.cue_label.config(text=symbol)
            self.cue_win.update()

    def record_stream_loop(self, stream_name):
        """Recording loop for a specific stream."""
        self.log(f"Recording thread started for {stream_name}")
        
        try:
            inlet = self.selected_streams[stream_name]['inlet']
            stream_type = self.selected_streams[stream_name]['type']
            
            while self.running:
                try:
                    # Pull data chunks
                    chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=100)
                    
                    if timestamps:
                        # Store the data
                        self.all_stream_data[stream_name].extend(chunk)
                        self.all_timestamps[stream_name].extend(timestamps)
                        
                except Exception as e:
                    if self.running:
                        self.log(f"Error in recording loop for {stream_name}: {e}")
                        time.sleep(0.5)
                        
        except Exception as e:
            self.log(f"Fatal error in recording loop for {stream_name}: {e}")
        finally:
            total_samples = len(self.all_stream_data.get(stream_name, []))
            duration = len(self.all_timestamps.get(stream_name, [])) / self.selected_streams[stream_name]['srate'] if self.selected_streams[stream_name]['srate'] > 0 else 0
            self.log(f"Recording thread finished for {stream_name}. Collected {total_samples} samples, {duration:.2f} seconds.")

    def start_trial(self):
        """Start a single trial."""
        if not self.running or self.current_trial >= len(self.trial_activity_sequence):
            self.complete_session()
            return

        current_activity = self.trial_activity_sequence[self.current_trial]
        self.log(f"Starting Trial {self.current_trial + 1}: {current_activity}")
        self.trial_label.config(text=f"Trial: {self.current_trial + 1} / {len(self.trial_activity_sequence)} ({current_activity})")

        # Pre-block blank
        self.update_cue("", "Blank")
        self.root.after(int(self.blank1_duration.get() * 1000), 
                       lambda: self.start_baseline(current_activity))

    def start_baseline(self, activity_type):
        """Start baseline period."""
        if not self.running:
            return
        
        self.update_cue("+", "Baseline")
        self._send_marker("baseline")
        
        max_duration = self.baseline1_duration.get()
        duration_randomized = random.uniform(self.min_baseline_duration, max_duration)
        self.log(f"Phase: Baseline ({duration_randomized:.2f} s)")
        
        self.root.after(int(duration_randomized * 1000), 
                       lambda: self.start_activity(activity_type))

    def start_activity(self, activity_type):
        """Start main activity phase."""
        if not self.running:
            return

        if activity_type == 'motor_execution':
            self.update_cue("M", "Motor Execution")
            duration = self.motor_duration.get()
            self._send_marker("motor_execution")
            self.log(f"Phase: Motor Execution ({duration} s)")
        elif activity_type == 'motor_imagery':
            self.update_cue("I", "Motor Imagery")
            duration = self.imagery_duration.get()
            self._send_marker("motor_imagery")
            self.log(f"Phase: Motor Imagery ({duration} s)")
        elif activity_type == 'rest':
            self.update_cue("", "Rest")
            duration = self.rest_duration.get()
            self._send_marker("rest")
            self.log(f"Phase: Rest ({duration} s)")
        else:
            self.log(f"Unknown activity type: {activity_type}")
            return

        self.root.after(int(duration * 1000), self.start_inter_block_blank)

    def start_inter_block_blank(self):
        """Start inter-block blank period."""
        if not self.running:
            return
        
        self.update_cue("", "Blank")
        duration = self.blank2_duration.get()
        self.log(f"Phase: Inter-block blank ({duration} s)")
        self.root.after(int(duration * 1000), self.end_trial)

    def end_trial(self):
        """End current trial."""
        if not self.running:
            return
        
        self.current_trial += 1
        
        if self.current_trial < len(self.trial_activity_sequence):
            self.root.after(500, self.start_trial)
        else:
            self.complete_session()

    def complete_session(self):
        """Complete the session."""
        self.log("Session completed!")
        self.update_cue("✓", "Session Complete")
        self._send_marker("session_end")
        self.root.after(2000, self.stop_session)

    def stop_session(self):
        """Stop recording and save data."""
        if not self.running:
            return
        
        self.log("Stopping recording session...")
        self.running = False
        
        # Close cue window
        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.destroy()
        
        # Wait for recording threads
        for stream_name, thread in self.recording_threads.items():
            if thread.is_alive():
                self.log(f"Waiting for recording thread {stream_name} to finish...")
                thread.join(timeout=3.0)
        
        # Send final marker
        if self.marker_outlet:
            self._send_marker("session_end")
        
        # Save data
        self.save_data()
        
        # Reset UI
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.trial_label.config(text="Trial: 0 / 0")
        
        # Disconnect streams
        self.disconnect_streams()
        
        self.log("Session stopped and data saved.")

    def save_data(self):
        """Save recorded data from all streams."""
        self.log("Saving data...")
        
        if not any(self.all_stream_data.values()):
            self.log("No data to save.")
            messagebox.showwarning("No Data", "No data was collected during the session.")
            return
        
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_format = self.file_format.get()
            
            selected_streams = self.get_selected_streams()
            
            if len(selected_streams) == 1:
                # Single stream - save as before
                self.save_single_stream_data(selected_streams[0], timestamp_str, save_format)
            else:
                # Multiple streams - save separately or combined
                self.save_multi_stream_data(selected_streams, timestamp_str, save_format)
                
        except Exception as e:
            self.log(f"Error saving data: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Save Error", f"Error saving data:\n{e}")

    def save_single_stream_data(self, stream_name, timestamp_str, save_format):
        """Save data from a single stream."""
        try:
            stream_data = np.array(self.all_stream_data[stream_name])
            stream_timestamps = np.array(self.all_timestamps[stream_name])
            
            if len(stream_data) == 0:
                self.log(f"No data for stream {stream_name}")
                return
            
            # Transpose for MNE (channels x samples)
            eeg_array = stream_data.T
            
            # Get stream info
            stream_info = self.selected_streams[stream_name]
            sfreq = stream_info['srate']
            
            # Create channel names
            ch_names = [f"{stream_name}_CH{i+1}" for i in range(eeg_array.shape[0])]
            ch_types = ['eeg'] * len(ch_names)
            
            # Create MNE info
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            info['description'] = f"Multi-stream recording from {stream_name} started at {datetime.fromtimestamp(self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Process markers
            marker_labels = [m[0] for m in self.markers]
            marker_times_lsl = np.array([m[1] for m in self.markers])
            
            if len(stream_timestamps) > 0:
                first_sample_time = stream_timestamps[0]
                marker_onsets = marker_times_lsl - first_sample_time
                valid_markers = marker_onsets >= 0
                
                annotations = mne.Annotations(
                    onset=marker_onsets[valid_markers],
                    duration=np.zeros(np.sum(valid_markers)),
                    description=[marker_labels[i] for i, valid in enumerate(valid_markers) if valid]
                )
            else:
                annotations = None
            
            # Create Raw object
            raw = mne.io.RawArray(eeg_array, info)
            if annotations:
                raw.set_annotations(annotations)
            
            # Save file
            filename = f"multistream_{stream_name}_{timestamp_str}.{save_format}"
            
            if save_format == "fif":
                raw.save(filename, overwrite=True)
                self.log(f"Data saved to {filename}")
                
                # Show success message
                messagebox.showinfo("Save Successful", 
                                  f"Data saved successfully!\n"
                                  f"File: {filename}\n"
                                  f"Shape: {eeg_array.shape}\n"
                                  f"Duration: {eeg_array.shape[1] / sfreq:.2f} seconds\n"
                                  f"Markers: {len(self.markers)}")
            
            elif save_format == "mat":
                mat_dict = {
                    'eeg_data': eeg_array,
                    'timestamps': stream_timestamps,
                    'channels': ch_names,
                    'sfreq': sfreq,
                    'markers': marker_labels,
                    'marker_timestamps': marker_times_lsl,
                    'stream_name': stream_name,
                    'info': str(info)
                }
                savemat(filename, mat_dict)
                self.log(f"Data saved to {filename}")
                
                messagebox.showinfo("Save Successful", 
                                  f"Data saved successfully!\n"
                                  f"File: {filename}\n"
                                  f"Shape: {eeg_array.shape}\n"
                                  f"Duration: {eeg_array.shape[1] / sfreq:.2f} seconds\n"
                                  f"Markers: {len(self.markers)}")
                
        except Exception as e:
            self.log(f"Error saving single stream data: {e}")
            raise

    def save_multi_stream_data(self, selected_streams, timestamp_str, save_format):
        """Save data from multiple streams."""
        try:
            # Save each stream separately
            for stream_name in selected_streams:
                self.save_single_stream_data(stream_name, timestamp_str, save_format)
            
            # Also save a combined file with all stream info
            if save_format == "mat":
                combined_dict = {
                    'recording_info': {
                        'start_time': self.recording_start_time,
                        'lsl_start_time': self.start_timestamp,
                        'streams_recorded': selected_streams,
                        'total_trials': len(self.trial_activity_sequence),
                        'activity_sequence': self.trial_activity_sequence
                    },
                    'markers': [m[0] for m in self.markers],
                    'marker_timestamps': [m[1] for m in self.markers]
                }
                
                # Add data from each stream
                for stream_name in selected_streams:
                    if self.all_stream_data[stream_name]:
                        stream_data = np.array(self.all_stream_data[stream_name])
                        combined_dict[f'{stream_name}_data'] = stream_data.T
                        combined_dict[f'{stream_name}_timestamps'] = np.array(self.all_timestamps[stream_name])
                        combined_dict[f'{stream_name}_info'] = {
                            'srate': self.selected_streams[stream_name]['srate'],
                            'channels': self.selected_streams[stream_name]['channels'],
                            'type': self.selected_streams[stream_name]['type']
                        }
                
                combined_filename = f"multistream_combined_{timestamp_str}.mat"
                savemat(combined_filename, combined_dict)
                self.log(f"Combined data saved to {combined_filename}")
            
            self.log(f"Successfully saved data from {len(selected_streams)} streams")
            
        except Exception as e:
            self.log(f"Error saving multi-stream data: {e}")
            raise


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiStreamEEGRecorder(root)
    root.mainloop()