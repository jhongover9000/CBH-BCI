import numpy as np
import mne
import threading
import time
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, local_clock
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from scipy.io import savemat
import random
import math # Added for ceiling division

class EEGMarkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Marker GUI")
        # Increased initial height slightly for the new checkbox
        self.root.geometry("550x900")
        self.root.minsize(600, 600)

        # Adjustable durations (in seconds)
        self.blank1_duration = tk.IntVar(value=2)      # Pre-trial blank (Block 1 or 2)
        self.baseline1_duration = tk.IntVar(value=5)     # First baseline (Block 1 or 2)
        self.motor_duration = tk.IntVar(value=5)         # Motor execution (Block 1 if enabled)
        self.blank2_duration = tk.IntVar(value=2)        # Blank between blocks
        self.baseline2_duration = tk.IntVar(value=5)     # Second baseline (Block 2)
        self.imagery_duration = tk.IntVar(value=5)       # Motor imagery (Activity Block)
        self.rest_duration = tk.IntVar(value=5)          # Rest (Activity Block)

        self.total_trials = tk.IntVar(value=10)
        self.file_format = tk.StringVar(value="fif")

        # Activity selection checkboxes
        self.select_motor_exec = tk.BooleanVar(value=True)
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        # --- NEW: Randomization Option ---
        self.randomize_block_order = tk.BooleanVar(value=False) # Option to randomize Execution vs Activity block order

        self.running = False
        self.current_trial = 0
        self.eeg_data = []
        self.timestamps = []
        self.markers = []
        self.start_timestamp_lsl = None
        self.recording_start_time = None # Wall clock time when recording started

        self.eeg_inlet = None
        self.marker_outlet = None
        self.eeg_channels = []
        self.eeg_indices = []
        self.sfreq = 0

        # Minimum baseline duration (in seconds) for randomization
        self.min_baseline_duration = 1.5

        # --- NEW: Sequences for balanced randomization ---
        self.trial_activity_sequence = [] # Holds 'imagery' or 'rest' for each trial if balanced
        self.trial_block_order_sequence = [] # Holds 'motor_first' or 'activity_first' if randomized

        self.setup_ui()
        self.setup_lsl()

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

        self.dynamic_labels = []
        self.dynamic_buttons = []

        # --- Font handling (remains the same) ---
        def responsive_font(base_size):
            width = self.root.winfo_width()
            scale = max(min(width / 600, 1.0), 0.75)
            return int(base_size * scale)

        def update_fonts():
            available_fonts = ["Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial", "TkDefaultFont"]
            font_name = available_fonts[0]
            # ... (rest of font finding logic)
            font_small = (font_name, responsive_font(9))
            font_normal = (font_name, responsive_font(10))
            font_bold = (font_name, responsive_font(12), 'bold')
            # ... (rest of font application logic)


        # --- UI Elements (Mostly the same, added block randomization checkbox) ---
        row_idx = 0

        # Pre-trial Blank Duration (Used before the FIRST block)
        lbl_blank1 = tk.Label(scrollable_frame, text="Pre-Block Blank Duration (s)")
        lbl_blank1.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_blank1)
        blank1_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                 variable=self.blank1_duration,
                                 command=lambda val: self.update_label(self.blank1_val_label, val))
        blank1_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.blank1_val_label = tk.Label(scrollable_frame, text=f"{self.blank1_duration.get()} s")
        self.blank1_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1

        # Baseline 1 Duration (Used before the FIRST block's activity)
        lbl_baseline1 = tk.Label(scrollable_frame, text="Pre-Activity Baseline Duration (s)")
        lbl_baseline1.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_baseline1)
        baseline1_scale = ttk.Scale(scrollable_frame, from_=self.min_baseline_duration, to=10, orient="horizontal",
                                    variable=self.baseline1_duration,
                                    command=lambda val: self.update_label(self.baseline1_val_label, val))
        baseline1_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.baseline1_val_label = tk.Label(scrollable_frame, text=f"{self.baseline1_duration.get()} s")
        self.baseline1_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1

        # --- Motor Execution Duration & Checkbox ---
        lbl_motor = tk.Label(scrollable_frame, text="Motor Execution Duration (s)")
        lbl_motor.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_motor)
        motor_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                variable=self.motor_duration,
                                command=lambda val: self.update_label(self.motor_val_label, val))
        motor_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.motor_val_label = tk.Label(scrollable_frame, text=f"{self.motor_duration.get()} s")
        self.motor_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1
        cb_motor = tk.Checkbutton(scrollable_frame, text="Enable Motor Execution Block", variable=self.select_motor_exec)
        cb_motor.grid(row=row_idx, column=0, sticky="w"); row_idx += 1
        self.dynamic_buttons.append(cb_motor)

        # --- Inter-Block Blank Duration ---
        lbl_blank2 = tk.Label(scrollable_frame, text="Inter-Block Blank Duration (s)")
        lbl_blank2.grid(row=row_idx, column=0, sticky="w", pady=(10, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_blank2)
        blank2_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                 variable=self.blank2_duration,
                                 command=lambda val: self.update_label(self.blank2_val_label, val))
        blank2_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.blank2_val_label = tk.Label(scrollable_frame, text=f"{self.blank2_duration.get()} s")
        self.blank2_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1

        # --- Baseline 2 Duration (Used before the SECOND block's activity) ---
        lbl_baseline2 = tk.Label(scrollable_frame, text="Inter-Block Baseline Duration (s)")
        lbl_baseline2.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_baseline2)
        baseline2_scale = ttk.Scale(scrollable_frame, from_=self.min_baseline_duration, to=10, orient="horizontal",
                                    variable=self.baseline2_duration,
                                    command=lambda val: self.update_label(self.baseline2_val_label, val))
        baseline2_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.baseline2_val_label = tk.Label(scrollable_frame, text=f"{self.baseline2_duration.get()} s")
        self.baseline2_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1

        # --- Activity Phase (Block 2) Selection & Durations ---
        lbl_activity = tk.Label(scrollable_frame, text="Activity Block Options (choose one or both):")
        lbl_activity.grid(row=row_idx, column=0, sticky="w", pady=(10, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_activity)
        act_frame = tk.Frame(scrollable_frame)
        act_frame.grid(row=row_idx, column=0, sticky="w", pady=(0, 10)); row_idx += 1
        cb_imagery = tk.Checkbutton(act_frame, text="Motor Imagery", variable=self.select_motor_imagery)
        cb_imagery.pack(side="left", padx=(0, 10))
        self.dynamic_buttons.append(cb_imagery)
        cb_rest = tk.Checkbutton(act_frame, text="Rest", variable=self.select_rest)
        cb_rest.pack(side="left")
        self.dynamic_buttons.append(cb_rest)
        # Motor Imagery Duration slider
        lbl_imagery = tk.Label(scrollable_frame, text="Motor Imagery Duration (s)")
        lbl_imagery.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_imagery)
        imagery_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                  variable=self.imagery_duration,
                                  command=lambda val: self.update_label(self.imagery_val_label, val))
        imagery_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.imagery_val_label = tk.Label(scrollable_frame, text=f"{self.imagery_duration.get()} s")
        self.imagery_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1
        # Rest Duration slider
        lbl_rest = tk.Label(scrollable_frame, text="Rest Duration (s)")
        lbl_rest.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_rest)
        rest_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                               variable=self.rest_duration,
                               command=lambda val: self.update_label(self.rest_val_label, val))
        rest_scale.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.rest_val_label = tk.Label(scrollable_frame, text=f"{self.rest_duration.get()} s")
        self.rest_val_label.grid(row=row_idx, column=0, sticky="e", pady=(0, 10)); row_idx += 1

        # --- NEW: Randomize Block Order Checkbox ---
        cb_random_block = tk.Checkbutton(scrollable_frame, text="Randomize Block Order (Execution vs. Activity)",
                                         variable=self.randomize_block_order)
        cb_random_block.grid(row=row_idx, column=0, sticky="w", pady=(5, 10)); row_idx += 1
        self.dynamic_buttons.append(cb_random_block)

        # --- Total Trials ---
        lbl_trials = tk.Label(scrollable_frame, text="Total Trials")
        lbl_trials.grid(row=row_idx, column=0, sticky="w", pady=(10, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_trials)
        t_spinbox = ttk.Spinbox(scrollable_frame, from_=1, to=100, textvariable=self.total_trials, width=5)
        t_spinbox.grid(row=row_idx, column=0, sticky="w", pady=(0, 10)); row_idx += 1

        # --- File Format ---
        lbl_format = tk.Label(scrollable_frame, text="Save Format")
        lbl_format.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_format)
        format_menu = ttk.OptionMenu(scrollable_frame, self.file_format, "fif", "fif", "mat")
        format_menu.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1

        # --- Trial Label ---
        self.trial_label = tk.Label(scrollable_frame, text="Trial: 0 / 0")
        self.trial_label.grid(row=row_idx, column=0, pady=(10, 5)); row_idx += 1

        # --- Logs ---
        lbl_logs = tk.Label(scrollable_frame, text="Logs")
        lbl_logs.grid(row=row_idx, column=0, sticky="w"); row_idx += 1
        self.dynamic_labels.append(lbl_logs)
        self.log_box = tk.Text(scrollable_frame, height=8, state="disabled", bg="#f5f5f5") # Increased height slightly
        self.log_box.grid(row=row_idx, column=0, sticky="nsew", pady=(0, 10)); row_idx += 1
        scrollable_frame.grid_rowconfigure(row_idx -1, weight=2) # Make log box expand vertically

        # --- Start and Stop Buttons ---
        self.start_button = tk.Button(scrollable_frame, text="Start Session", command=self.start_session) # Changed command
        self.start_button.grid(row=row_idx, column=0, sticky="ew", pady=(0, 5)); row_idx += 1
        self.dynamic_buttons.append(self.start_button)
        self.stop_button = tk.Button(scrollable_frame, text="Stop and Save", command=self.stop_session, state="disabled") # Changed command
        self.stop_button.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.dynamic_buttons.append(self.stop_button)

        # Font update binding (remains the same)
        # update_fonts()
        # self.root.bind('<Configure>', lambda e: update_fonts())
        # self.update_fonts = update_fonts

    def update_label(self, label_widget, val):
        # Ensure minimum value for baseline scales is respected in label
        if label_widget in [self.baseline1_val_label, self.baseline2_val_label]:
            fval = max(float(val), self.min_baseline_duration)
            label_widget.config(text=f"{fval:.1f} s")
        else:
            label_widget.config(text=f"{int(float(val))} s")


    def update_cue(self, symbol, title):
        # Ensure cue window exists before updating
        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.title(title)
            self.cue_label.config(text=symbol)
            self.cue_win.update()
            # self.update_fonts() # Font update might be too slow here

    def setup_lsl(self):
        # (LSL setup remains the same)
        self.log("Looking for EEG stream...")
        try:
            streams = resolve_byprop('type', 'EEG', timeout=2)
            if not streams:
                self.log("Error: No EEG stream found.")
                # messagebox.showerror("Error", "No EEG stream found.") # Avoid blocking GUI thread
                # self.root.destroy()
                return

            self.eeg_inlet = StreamInlet(streams[0])
            self.log(f"Connected to EEG stream: {streams[0].name()}")

            marker_info = StreamInfo('ExperimentMarkers', 'Markers', 1, 0, 'string', 'myuidw43536')
            self.marker_outlet = StreamOutlet(marker_info)
            self.log("Marker stream created.")

            info = self.eeg_inlet.info()
            self.sfreq = info.nominal_srate()
            if self.sfreq == 0: # If nominal_srate is not reliable, try to estimate
                self.log("Warning: Nominal sampling rate is 0. Trying to estimate...")
                time.sleep(1.2) # Wait for some samples
                _, ts = self.eeg_inlet.pull_chunk(max_samples=100)
                if len(ts) > 1:
                    self.sfreq = len(ts) / (ts[-1] - ts[0])
                    self.log(f"Estimated sampling rate: {self.sfreq:.2f} Hz")
                else:
                    self.sfreq = 250 # Default fallback
                    self.log(f"Warning: Could not estimate sampling rate. Using default: {self.sfreq} Hz")


            ch_list = info.desc().child("channels").first_child()
            print(info.desc().child("channels").first_child())
            channel_names = []
            while ch_list.name() == "channel":
                channel_names.append(ch_list.child_value("label"))
                ch_list = ch_list.next_sibling()
            print(ch_list)
            # Exclude non-EEG channels more robustly if needed (e.g., AUX, ECG)
            self.eeg_channels = [ch for ch in channel_names if ch.upper().startswith('EEG') or ch.upper() in ['CZ', 'C3', 'C4', 'PZ', 'P3', 'P4', 'O1', 'O2', 'FZ', 'F3', 'F4', 'T7', 'T8']] # Add common EEG names if prefix fails
            # Fallback if no 'EEG' prefix found
            if not self.eeg_channels:
                self.eeg_channels = [ch for ch in channel_names if not ch.lower().startswith(('aux', 'ecg', 'emg', 'acc', 'misc', 'stim'))]
                self.log("Warning: No channels start with 'EEG'. Using common EEG names or excluding known non-EEG prefixes.")


            self.eeg_indices = [i for i, ch in enumerate(channel_names) if ch in self.eeg_channels]

            if not self.eeg_indices:
                 self.log("Error: Could not identify any EEG channels.")
                 messagebox.showerror("Error", "No EEG channels found in the stream.")
                 self.eeg_inlet = None # Prevent starting session
                 return

            self.log(f"Selected EEG Channels ({len(self.eeg_channels)}): {', '.join(self.eeg_channels)}")
            excluded = [ch for ch in channel_names if ch not in self.eeg_channels]
            if excluded:
                self.log(f"Excluded Channels: {', '.join(excluded)}")
            self.log(f"Sampling Frequency: {self.sfreq} Hz")

        except Exception as e:
            self.log(f"Error during LSL setup: {e}")
            messagebox.showerror("LSL Error", f"Could not connect or setup LSL streams:\n{e}")


    def _send_marker(self, marker_label):
        """Helper function to send marker and log it."""
        try:
            marker_timestamp = local_clock() # Capture LSL timestamp at the moment of sending
            self.marker_outlet.push_sample([marker_label], timestamp=marker_timestamp)
            # Calculate time relative to the start of the recording
            if self.recording_start_time is not None:
                 # MNE uses time relative to the *first sample*, LSL clock relative to its own epoch
                 # Store both absolute LSL time and the label for later alignment
                self.markers.append((marker_label, marker_timestamp))
                relative_time_lsl = marker_timestamp - self.start_timestamp_lsl # Relative to LSL clock start
                self.log(f"Marker '{marker_label}' sent at LSL: {marker_timestamp:.4f} (Relative LSL: {relative_time_lsl:.4f})")
            else:
                self.markers.append((marker_label, marker_timestamp)) # Fallback
                self.log(f"Marker '{marker_label}' sent at LSL: {marker_timestamp:.4f} (Start time not yet set)")
        except Exception as e:
            self.log(f"Error sending marker '{marker_label}': {e}")


    def start_session(self):
        # --- Input Validation ---
        if not self.eeg_inlet:
             self.log("Cannot start: EEG stream not available.")
             messagebox.showerror("Error", "EEG stream not connected. Please restart the application or check the LSL stream.")
             return

        if not self.marker_outlet:
            self.log("Cannot start: Marker stream not available.")
            messagebox.showerror("Error", "Marker stream could not be created.")
            return

        n_trials = self.total_trials.get()
        if n_trials <= 0:
            messagebox.showerror("Error", "Total trials must be greater than 0.")
            return

        motor_selected = self.select_motor_exec.get()
        imagery_selected = self.select_motor_imagery.get()
        rest_selected = self.select_rest.get()
        activity_selected = imagery_selected or rest_selected

        if not motor_selected and not activity_selected:
            messagebox.showerror("Error", "At least one block type (Motor Execution or an Activity) must be selected.")
            return

        # Warn if randomization is selected but only one block type is enabled
        if self.randomize_block_order.get() and not (motor_selected and activity_selected):
            messagebox.showwarning("Warning", "Randomize Block Order selected, but only one block type (Execution or Activity) is enabled. Randomization will have no effect.")
            self.randomize_block_order.set(False) # Disable it visually

        # --- Generate Trial Sequences ---
        self.trial_activity_sequence = []
        self.trial_block_order_sequence = []

        # 1. Activity Sequence (Imagery vs. Rest)
        if activity_selected:
            if imagery_selected and rest_selected:
                num_imagery = n_trials // 2
                num_rest = n_trials - num_imagery # Handles odd numbers
                self.trial_activity_sequence = ['imagery'] * num_imagery + ['rest'] * num_rest
                random.shuffle(self.trial_activity_sequence)
                self.log(f"Generated balanced Imagery/Rest sequence for {n_trials} trials.")
            elif imagery_selected:
                self.trial_activity_sequence = ['imagery'] * n_trials
            else: # only rest_selected
                self.trial_activity_sequence = ['rest'] * n_trials

        # 2. Block Order Sequence (Motor vs. Activity)
        if motor_selected and activity_selected and self.randomize_block_order.get():
            num_motor_first = n_trials // 2
            num_activity_first = n_trials - num_motor_first
            self.trial_block_order_sequence = ['motor_first'] * num_motor_first + ['activity_first'] * num_activity_first
            random.shuffle(self.trial_block_order_sequence)
            self.log(f"Generated randomized Block Order sequence for {n_trials} trials.")
        elif motor_selected: # Default order if randomization off or only motor selected
             self.trial_block_order_sequence = ['motor_first'] * n_trials
        else: # Only activity selected
             self.trial_block_order_sequence = ['activity_first'] * n_trials


        # --- Start Recording ---
        self.running = True
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
        # Ensure it grabs focus and handles Esc
        self.cue_win.focus_force()
        self.cue_win.bind("<Escape>", lambda e: self.stop_session())


        self.cue_label = tk.Label(self.cue_win, text="", fg="white", bg="black", font=("Arial", 100, "bold"))
        self.cue_label.pack(expand=True)
        self.cue_win.update() # Make sure it's drawn

        # Bind Escape key to main window as well
        self.root.bind("<Escape>", lambda e: self.stop_session())


        # --- Start LSL data acquisition thread ---
        # Get the first timestamp *before* starting the recording thread
        try:
            # Pull a sample to sync clocks and get initial timestamp
            _, ts = self.eeg_inlet.pull_sample(timeout=1.0)
            if ts is None:
                 self.log("Error: Could not get initial sample from LSL stream.")
                 self.stop_session()
                 messagebox.showerror("LSL Error", "Failed to pull initial sample from EEG stream.")
                 return

            self.start_timestamp_lsl = local_clock() # Get precise start time using LSL's clock
            self.recording_start_time = time.time() # Wall clock start time
            self.log(f"Recording started. LSL Start Time: {self.start_timestamp_lsl:.4f}")
            self._send_marker("session_start") # Mark session start

            # Start the recording thread
            self.recording_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.recording_thread.start()

            # Start the first trial after a short delay
            self.root.after(500, self.start_trial)

        except Exception as e:
            self.log(f"Error starting recording: {e}")
            self.stop_session()
            messagebox.showerror("Error", f"Failed to start recording:\n{e}")


    def stop_session(self):
        if not self.running: # Prevent double stop
            return
        self.running = False
        self.log("Stopping session...")
        self._send_marker("session_end") # Mark session end

        # Wait briefly for the recording thread to finish its current chunk
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            time.sleep(0.1) # Adjust if needed

        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.destroy()

        # Cancel any pending `after` calls to prevent errors
        for after_id in self.root.tk.call('after', 'info'):
            self.root.after_cancel(after_id)

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        # Unbind Escape keys
        self.root.unbind("<Escape>")
        # No need to unbind from cue_win as it's destroyed


        if not self.eeg_data:
            self.log("No EEG data recorded.")
        else:
            self.log("Recording stopped. Saving data...")
            self.save_data()


    def start_trial(self):
        """Initiates a single trial."""
        if not self.running or self.current_trial >= self.total_trials.get():
            if self.running: # If stopped externally, stop_session handles logging/saving
                self.stop_session()
            return

        self.current_trial += 1
        self.trial_label.config(text=f"Trial: {self.current_trial} / {self.total_trials.get()}")
        self.log(f"\n--- Starting Trial {self.current_trial} ---")

        # Determine block order for this trial
        current_block_order = self.trial_block_order_sequence[self.current_trial - 1]
        self.log(f"Trial {self.current_trial} Order: {current_block_order}")

        # Start the first phase: Blank before the first block
        self.start_blank_phase(is_first_block=True, next_phase='baseline1')

    # --- Phase Control Functions ---

    def start_blank_phase(self, is_first_block, next_phase):
        """Handles blank periods."""
        duration = self.blank1_duration.get() if is_first_block else self.blank2_duration.get()
        marker = 'blank_pre' if is_first_block else 'blank_inter'
        self.update_cue("", "Blank")
        self._send_marker(marker)
        self.log(f"Phase: {marker} ({duration} s)")
        self.root.after(duration * 1000, getattr(self, f'start_{next_phase}_phase')) # Calls start_baseline1_phase or start_baseline2_phase

    def start_baseline1_phase(self):
        """Handles the baseline period BEFORE the FIRST activity/execution block."""
        self.update_cue("+", "Baseline")
        self._send_marker('baseline_1')

        # Use max duration from slider, randomize between min and max
        max_duration = self.baseline1_duration.get()
        duration_randomized = random.uniform(self.min_baseline_duration, max_duration)
        self.log(f"Phase: Baseline 1 ({duration_randomized:.2f} s)")

        # Determine what comes next based on the pre-shuffled block order
        first_block_type = self.trial_block_order_sequence[self.current_trial - 1]
        if first_block_type == 'motor_first':
            next_func = self.start_motor_execution_phase
        else: # 'activity_first'
            next_func = self.start_activity_phase

        self.root.after(int(duration_randomized * 1000), next_func, True) # Pass is_first_block=True

    def start_baseline2_phase(self):
        """Handles the baseline period BEFORE the SECOND activity/execution block."""
        self.update_cue("+", "Baseline")
        self._send_marker('baseline_2')

        # Use max duration from slider, randomize between min and max
        max_duration = self.baseline2_duration.get()
        duration_randomized = random.uniform(self.min_baseline_duration, max_duration)
        self.log(f"Phase: Baseline 2 ({duration_randomized:.2f} s)")

        # Determine what comes next (the second block)
        first_block_type = self.trial_block_order_sequence[self.current_trial - 1]
        if first_block_type == 'motor_first':
            # Motor was first, so Activity is second
            next_func = self.start_activity_phase
        else: # 'activity_first'
            # Activity was first, so Motor is second
            next_func = self.start_motor_execution_phase

        self.root.after(int(duration_randomized * 1000), next_func, False) # Pass is_first_block=False


    def start_motor_execution_phase(self, is_first_block):
        """Handles the motor execution block."""
        if not self.select_motor_exec.get(): # Should not happen if logic is correct, but safe check
             self.log("Motor Execution skipped (not selected)")
             # Decide what to do if skipped - depends if it was first or second
             if is_first_block:
                 self.start_blank_phase(is_first_block=False, next_phase='baseline2')
             else:
                 self.start_trial() # End of trial
             return

        self.update_cue("M", "Motor Execution")
        self._send_marker('execution_start')
        duration = self.motor_duration.get()
        self.log(f"Phase: Motor Execution ({duration} s)")

        # Schedule end marker and next phase
        self.root.after(duration * 1000, self.end_motor_execution_phase, is_first_block)

    def end_motor_execution_phase(self, is_first_block):
        """Sends end marker and transitions from motor execution."""
        self._send_marker('execution_end')
        if is_first_block:
            # Finished first block, move to inter-block blank/baseline
             self.start_blank_phase(is_first_block=False, next_phase='baseline2')
        else:
            # Finished second block, end trial
            self.start_trial() # Will check if more trials are left


    def start_activity_phase(self, is_first_block):
        """Handles the activity (Imagery or Rest) block."""
        if not (self.select_motor_imagery.get() or self.select_rest.get()):
             self.log("Activity Phase skipped (none selected)")
             if is_first_block:
                 self.start_blank_phase(is_first_block=False, next_phase='baseline2')
             else:
                 self.start_trial()
             return

        # Get the activity for the current trial from the pre-shuffled list
        activity = self.trial_activity_sequence[self.current_trial - 1]
        marker_start = f"{activity}_start"

        if activity == 'imagery':
            self.update_cue("I", "Motor Imagery")
            duration = self.imagery_duration.get()
            self.log(f"Phase: Motor Imagery ({duration} s)")
        else: # activity == 'rest'
            self.update_cue("", "Rest") # Or maybe 'R'? Changed to blank as per original.
            duration = self.rest_duration.get()
            self.log(f"Phase: Rest ({duration} s)")

        self._send_marker(marker_start)
        self.root.after(duration * 1000, self.end_activity_phase, is_first_block, activity)


    def end_activity_phase(self, is_first_block, activity_type):
        """Sends end marker and transitions from activity phase."""
        marker_end = f"{activity_type}_end"
        self._send_marker(marker_end)
        if is_first_block:
            # Finished first block, move to inter-block blank/baseline
            self.start_blank_phase(is_first_block=False, next_phase='baseline2')
        else:
            # Finished second block, end trial
            self.start_trial()

    # --- Data Recording and Saving ---

    def record_loop(self):
        self.log("EEG recording thread started.")
        while self.running:
            try:
                # Pull chunks for better performance
                chunk, timestamps_lsl = self.eeg_inlet.pull_chunk(timeout=1.0, max_samples=100) # Increased max_samples
                if timestamps_lsl:
                    # Process samples using list comprehension for speed
                    valid_indices = self.eeg_indices
                    processed_chunk = [[sample[i] for i in valid_indices] for sample in chunk]

                    # Extend lists (more efficient than appending one by one)
                    self.eeg_data.extend(processed_chunk)
                    self.timestamps.extend(timestamps_lsl)

                    # Optional: Log chunk size periodically to check timing
                    # if len(self.timestamps) % 1000 < len(timestamps_lsl): # Log roughly every 1000 samples
                    #    self.log(f"Recorded chunk of {len(timestamps_lsl)} samples.")

            except Exception as e:
                # Log error but keep trying unless self.running is False
                if self.running: # Avoid logging error if we stopped intentionally
                    self.log(f"Error in recording loop: {e}")
                    time.sleep(0.5) # Avoid spamming logs if error persists
        self.log("EEG recording thread finished.")


    def log(self, message):
        # Ensure GUI updates happen in the main thread
        def _log_update():
            if self.log_box.winfo_exists(): # Check if widget still exists
                self.log_box.config(state="normal")
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.log_box.insert("end", f"[{timestamp}] {message}\n")
                self.log_box.config(state="disabled")
                self.log_box.see("end")
        # Schedule the update in the main Tkinter thread
        self.root.after(0, _log_update)


    def save_data(self):
        if not self.eeg_data:
            self.log("No data to save.")
            return

        try:
            eeg_array = np.array(self.eeg_data).T # Transpose for MNE (channels x samples)
            times_lsl = np.array(self.timestamps) # LSL timestamps for EEG data

            if eeg_array.shape[1] != len(times_lsl):
                 self.log(f"Warning: Mismatch between EEG samples ({eeg_array.shape[1]}) and timestamps ({len(times_lsl)}). Attempting to save anyway.")
                 # Optional: Truncate to the smaller size
                 min_len = min(eeg_array.shape[1], len(times_lsl))
                 eeg_array = eeg_array[:, :min_len]
                 times_lsl = times_lsl[:min_len]


            # --- Create MNE Info object ---
            # Use selected EEG channels directly
            ch_names = self.eeg_channels
            ch_types = ['eeg'] * len(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types=ch_types)
            info['description'] = f"Experiment recording started at {datetime.fromtimestamp(self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')}"

            # --- Process Markers for MNE Annotations ---
            marker_labels = [m[0] for m in self.markers]
            marker_times_lsl = np.array([m[1] for m in self.markers])

            # MNE Annotations need onset times relative to the *first EEG sample's timestamp*
            if times_lsl.size > 0:
                 first_sample_time_lsl = times_lsl[0]
                 # Ensure markers are sorted by time, just in case
                 sort_idx = np.argsort(marker_times_lsl)
                 marker_times_lsl = marker_times_lsl[sort_idx]
                 marker_labels = [marker_labels[i] for i in sort_idx]

                 # Calculate onsets relative to the first EEG sample's LSL time
                 marker_onsets_relative = marker_times_lsl - first_sample_time_lsl

                 # Filter out markers that occurred before the first EEG sample (e.g., session_start sometimes)
                 valid_marker_indices = marker_onsets_relative >= 0
                 marker_onsets_final = marker_onsets_relative[valid_marker_indices]
                 marker_labels_final = [marker_labels[i] for i, valid in enumerate(valid_marker_indices) if valid]

                 if len(marker_labels_final) != len(marker_labels):
                      self.log(f"Warning: {len(marker_labels) - len(marker_labels_final)} markers occurred before the first EEG sample and were excluded from annotations.")

                 annotations = mne.Annotations(onset=marker_onsets_final,
                                       duration=np.zeros(len(marker_labels_final)), # Markers are events
                                       description=marker_labels_final,
                                       orig_time=None) # Relate onset to data start
                 self.log(f"Created {len(marker_labels_final)} annotations for MNE.")

            else:
                 self.log("Warning: No EEG timestamps recorded, cannot create relative MNE annotations.")
                 annotations = None

            # --- Create MNE Raw object ---
            # Data should be (n_channels, n_times)
            raw = mne.io.RawArray(eeg_array, info)

            if annotations:
                try:
                    raw.set_annotations(annotations)
                except Exception as e:
                    self.log(f"Error setting annotations: {e}. Onsets: {marker_onsets_final[:5]}, Labels: {marker_labels_final[:5]}")


            # --- Save File ---
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"eeg_recording_{timestamp_str}"
            save_format = self.file_format.get()
            filename = f"{filename_base}.{save_format}"

            if save_format == "fif":
                raw.save(filename, overwrite=True)
                self.log(f"Data saved to {filename}")
                # --- Verification (Optional but recommended) ---
                try:
                    raw_check = mne.io.read_raw_fif(filename, preload=False)
                    self.log(f"Successfully re-read {filename}. Info: {raw_check.info['nchan']} channels, {raw_check.n_times} samples.")
                    if raw_check.annotations:
                         self.log(f"Verified {len(raw_check.annotations)} annotations in saved file.")
                         # print("First 5 annotations:", raw_check.annotations[:5])
                    else:
                         self.log("No annotations found during verification (as expected if none were created).")
                except Exception as e:
                    self.log(f"Error verifying saved FIF file: {e}")

            elif save_format == "mat":
                 # For MAT, save useful components separately
                 mat_dict = {
                     'eeg_data': eeg_array, # channels x samples
                     'eeg_timestamps_lsl': times_lsl, # LSL timestamps for each sample
                     'channels': ch_names,
                     'sfreq': self.sfreq,
                     'markers': marker_labels, # Original list of marker labels
                     'marker_timestamps_lsl': marker_times_lsl, # LSL timestamps for each marker
                     'info': str(info), # Basic info as string
                     'lsl_start_time': self.start_timestamp_lsl,
                     'recording_start_time_unix': self.recording_start_time,
                 }
                 if 'marker_onsets_relative' in locals():
                      mat_dict['marker_onsets_relative_to_first_sample'] = marker_onsets_relative

                 savemat(filename, mat_dict)
                 self.log(f"Data saved to {filename}")

            else:
                 self.log(f"Error: Unknown file format '{save_format}'")


        except Exception as e:
            self.log(f"Error during data saving: {e}")
            import traceback
            self.log(traceback.format_exc()) # Log detailed traceback
            messagebox.showerror("Save Error", f"An error occurred while saving data:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EEGMarkerGUI(root)
    root.mainloop()