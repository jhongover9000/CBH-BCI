import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import random
import serial
import serial.tools.list_ports
import math
import os
import sys
import traceback
import csv
import json

# Import PsychoPy for the visual display
try:
    from psychopy import visual, core, event, monitors
    PSYCHOPY_AVAILABLE = True
except ImportError:
    PSYCHOPY_AVAILABLE = False

class EEGMarkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MI Assessment Conductor UI")
        # Wider window for two-column layout
        self.root.geometry("1000x650")
        self.root.minsize(1000, 650)

        # Save Directory
        self.save_dir = "./mi_assessment_results/"

        # Marker Values
        self.session_start = 0
        self.baseline = 2
        self.motor_imagery = 3
        self.rest = 4
        self.eval_start = 6
        self.eval_yes = 7
        self.eval_no = 8
        self.session_end = 10

        # Subject and session information
        self.subject_number = tk.StringVar(value="")
        self.is_post_assessment = tk.BooleanVar(value=False)
        
        # Randomization options
        self.use_randomization_file = tk.BooleanVar(value=False)
        self.randomization_file_path = tk.StringVar(value="")

        # Adjustable durations (in seconds)
        self.baseline_duration = tk.IntVar(value=4)     # Baseline duration
        self.imagery_duration = tk.IntVar(value=3)      # Motor imagery duration
        self.rest_duration = tk.IntVar(value=3)         # Rest duration

        self.total_trials = tk.IntVar(value=40)

        # Minimum baseline duration (in seconds) for randomization
        self.min_baseline_duration = 2.5

        # Activity selection checkboxes
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        # --- Serial Port Variables ---
        self.com_port = tk.StringVar(value="")
        self.available_ports = []
        self.serial_connection = None
        self.baud_rate = tk.IntVar(value=2000000)

        # --- Monitor Selection Variables ---
        self.available_monitors = []
        self.selected_monitor = tk.IntVar(value=0)  # Default to primary monitor

        self.running = False
        self.current_trial = 0
        self.markers = []
        self.recording_start_time = None # Wall clock time when recording started
        
        # For evaluation results storage
        self.eval_results = []
        
        # --- Sequences for activity randomization ---
        self.trial_activity_sequence = [] # Holds 'imagery' or 'rest' for each trial if balanced

        # For video playing
        self.showing_landing = False
        self.landing_text = "Press Y to replay the instruction video\nPress N to begin the experiment"
        self.video_file_path = tk.StringVar(value="")
        self.use_instruction_video = tk.BooleanVar(value=False)
        self.video_stimulus = None
        self.showing_video = False

        # PsychoPy related variables
        self.psychopy_window = None
        self.cue_text = None
        self.question_text = None
        self.evaluation_showing = False
        self.waiting_for_response = False

        # Questions and Instructions
        self.imagery_question = "Did you imagine the motor movement?"
        self.rest_question = "Were you able to maintain a resting state?"
        self.instructions = "Welcome to the Motor Imagery Assessment.\n\nAfter a baseline period (+), you will be asked to either imagine a motor movement when you see the \u2022 cue, or to maintain a resting state when you see a blank screen.\n\nAfter each task, you will evaluate your performance.\n\nPlease let the experimenter know when you are ready to view the motor movement to imagine."
        self.show_instructions = True
        
        self.setup_ui()
        self.get_available_com_ports()
        self.get_available_monitors()

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Configure grid with two columns
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Create a canvas and scrollbar for the left column
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        # Create frames for left and right columns
        left_frame = ttk.Frame(canvas, padding="5")
        right_frame = ttk.Frame(main_frame, padding="5")
        
        # Setup canvas for scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=left_frame, anchor="nw")
        left_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Place canvas, scrollbar, and right frame in the grid
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=0, sticky="nse")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Make the canvas expand to fill the frame
        main_frame.rowconfigure(0, weight=1)
        
        self.dynamic_labels = []
        self.dynamic_buttons = []
        
        # Store indices for monitor selection
        self.monitor_indices = [0]  # Default to primary monitor

        # === LEFT COLUMN - SETUP & HARDWARE ===
        left_row = 0

        # --- Subject Information ---
        subj_frame = ttk.LabelFrame(left_frame, text="Subject Information")
        subj_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        left_row += 1
        
        # Subject number
        ttk.Label(subj_frame, text="Subject Number:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(subj_frame, textvariable=self.subject_number, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Pre/Post assessment
        ttk.Label(subj_frame, text="Assessment Type:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        assessment_frame = ttk.Frame(subj_frame)
        assessment_frame.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(assessment_frame, text="Pre", variable=self.is_post_assessment, value=False).pack(side="left")
        ttk.Radiobutton(assessment_frame, text="Post", variable=self.is_post_assessment, value=True).pack(side="left")

        # --- Serial Port Selection ---
        serial_frame = ttk.LabelFrame(left_frame, text="Serial Port Settings")
        serial_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        left_row += 1
        
        # COM Port dropdown
        ttk.Label(serial_frame, text="COM Port:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        port_frame = ttk.Frame(serial_frame)
        port_frame.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        self.port_dropdown = ttk.Combobox(port_frame, textvariable=self.com_port, state="readonly", width=10)
        self.port_dropdown.pack(side="left", padx=(0, 5))
        
        refresh_button = ttk.Button(port_frame, text="Refresh", command=self.get_available_com_ports)
        refresh_button.pack(side="left")
        self.dynamic_buttons.append(refresh_button)
        
        # Baud Rate
        ttk.Label(serial_frame, text="Baud Rate:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        baud_dropdown = ttk.Combobox(serial_frame, textvariable=self.baud_rate, state="readonly", width=10,
                                    values=[9600, 19200, 38400, 57600, 115200, 2000000])
        baud_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # --- Monitor Selection ---
        monitor_frame = ttk.LabelFrame(left_frame, text="Display Settings")
        monitor_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        left_row += 1
        
        ttk.Label(monitor_frame, text="Cue Monitor:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        mon_ctrl_frame = ttk.Frame(monitor_frame)
        mon_ctrl_frame.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        self.monitor_dropdown = ttk.Combobox(mon_ctrl_frame, textvariable=self.selected_monitor, state="readonly", width=10)
        self.monitor_dropdown.pack(side="left", padx=(0, 5))
        
        refresh_monitor_button = ttk.Button(mon_ctrl_frame, text="Refresh", command=self.get_available_monitors)
        refresh_monitor_button.pack(side="left")
        self.dynamic_buttons.append(refresh_monitor_button)

        # --- Randomization File Option ---
        random_frame = ttk.LabelFrame(left_frame, text="Randomization Settings")
        random_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        left_row += 1
        
        # Checkbox to use randomization file
        ttk.Checkbutton(random_frame, text="Use Randomization File", 
                      variable=self.use_randomization_file,
                      command=self.toggle_randomization_file).grid(
                          row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # File path and browse button
        self.file_path_entry = ttk.Entry(random_frame, textvariable=self.randomization_file_path, width=20, state="disabled")
        self.file_path_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
        self.browse_button = ttk.Button(random_frame, text="Browse", command=self.browse_randomization_file, state="disabled")
        self.browse_button.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.dynamic_buttons.append(self.browse_button)

        video_frame = ttk.LabelFrame(left_frame, text="Instruction Video")
        video_frame.grid(row=left_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        left_row += 1

        # Checkbox to use instruction video
        ttk.Checkbutton(video_frame, text="Show instruction video", 
                    variable=self.use_instruction_video,
                    command=self.toggle_video_settings).grid(
                        row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # File path and browse button
        self.video_path_entry = ttk.Entry(video_frame, textvariable=self.video_file_path, width=20, state="disabled")
        self.video_path_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=2)

        self.video_browse_button = ttk.Button(video_frame, text="Browse", command=self.browse_video_file, state="disabled")
        self.video_browse_button.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.dynamic_buttons.append(self.video_browse_button)

        # === RIGHT COLUMN - EXPERIMENT PARAMETERS ===
        right_row = 0

        # --- Baseline Duration ---
        param_frame = ttk.LabelFrame(right_frame, text="Experiment Parameters")
        param_frame.grid(row=right_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        right_row += 1
        
        param_row = 0
        
        ttk.Label(param_frame, text="Baseline Duration (s):").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        baseline_frame = ttk.Frame(param_frame)
        baseline_frame.grid(row=param_row, column=1, sticky="ew", padx=5, pady=2)
        param_row += 1
        
        baseline_scale = ttk.Scale(baseline_frame, from_=self.min_baseline_duration, to=10, orient="horizontal",
                                  variable=self.baseline_duration, length=150,
                                  command=lambda val: self.update_label(self.baseline_val_label, val))
        baseline_scale.pack(side="left", padx=(0, 5))
        
        self.baseline_val_label = ttk.Label(baseline_frame, text=f"{self.baseline_duration.get()} s", width=5)
        self.baseline_val_label.pack(side="left")

        # --- Activity Selection ---
        ttk.Label(param_frame, text="Activity Types:").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        act_frame = ttk.Frame(param_frame)
        act_frame.grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
        param_row += 1
        
        ttk.Checkbutton(act_frame, text="Motor Imagery", variable=self.select_motor_imagery).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(act_frame, text="Rest", variable=self.select_rest).pack(side="left")
        
        # Motor Imagery Duration
        ttk.Label(param_frame, text="Imagery Duration (s):").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        imagery_frame = ttk.Frame(param_frame)
        imagery_frame.grid(row=param_row, column=1, sticky="ew", padx=5, pady=2)
        param_row += 1
        
        imagery_scale = ttk.Scale(imagery_frame, from_=1, to=10, orient="horizontal",
                                 variable=self.imagery_duration, length=150,
                                 command=lambda val: self.update_label(self.imagery_val_label, val))
        imagery_scale.pack(side="left", padx=(0, 5))
        
        self.imagery_val_label = ttk.Label(imagery_frame, text=f"{self.imagery_duration.get()} s", width=5)
        self.imagery_val_label.pack(side="left")
        
        # Rest Duration
        ttk.Label(param_frame, text="Rest Duration (s):").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        rest_frame = ttk.Frame(param_frame)
        rest_frame.grid(row=param_row, column=1, sticky="ew", padx=5, pady=2)
        param_row += 1
        
        rest_scale = ttk.Scale(rest_frame, from_=1, to=10, orient="horizontal",
                              variable=self.rest_duration, length=150,
                              command=lambda val: self.update_label(self.rest_val_label, val))
        rest_scale.pack(side="left", padx=(0, 5))
        
        self.rest_val_label = ttk.Label(rest_frame, text=f"{self.rest_duration.get()} s", width=5)
        self.rest_val_label.pack(side="left")
        
        # Total Trials
        ttk.Label(param_frame, text="Total Trials:").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.total_trials, width=5).grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
        param_row += 1

        # --- Status and Control ---
        status_frame = ttk.LabelFrame(right_frame, text="Status and Control")
        status_frame.grid(row=right_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        right_row += 1
        
        # Trial Label
        self.trial_label = ttk.Label(status_frame, text="Trial: 0 / 0")
        self.trial_label.pack(pady=5)

        # Start/Stop Buttons 
        button_frame = ttk.Frame(status_frame)
        button_frame.pack(fill="x", pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Session", command=self.start_session)
        self.start_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.dynamic_buttons.append(self.start_button)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Session", command=self.stop_session, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        self.dynamic_buttons.append(self.stop_button)

        # --- Logs ---
        log_frame = ttk.LabelFrame(right_frame, text="Logs")
        log_frame.grid(row=right_row, column=0, sticky="nsew", pady=(0, 10), padx=5)
        right_row += 1
        right_frame.rowconfigure(right_row-1, weight=1)  # Make logs expand
        
        self.log_box = tk.Text(log_frame, height=15, state="disabled", bg="#f5f5f5")
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add a trace to handle monitor selection changes via the dropdown
        self.monitor_dropdown.bind('<<ComboboxSelected>>', self.on_monitor_selected)

    def toggle_randomization_file(self):
        """Enable/disable randomization file controls based on checkbox"""
        if self.use_randomization_file.get():
            self.file_path_entry.config(state="normal")
            self.browse_button.config(state="normal")
        else:
            self.file_path_entry.config(state="disabled")
            self.browse_button.config(state="disabled")
            
    def browse_randomization_file(self):
        """Open file dialog to select a randomization file"""
        filetypes = [
            ('Text files', '*.txt'),
            ('JSON files', '*.json'),
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Randomization File",
            filetypes=filetypes
        )
        
        if filename:
            self.randomization_file_path.set(filename)
            self.log(f"Selected randomization file: {filename}")
    
    def load_randomization_from_file(self):
        """Load trial sequence from file"""
        if not self.use_randomization_file.get() or not self.randomization_file_path.get():
            return False
            
        filepath = self.randomization_file_path.get()
        if not os.path.exists(filepath):
            self.log(f"Error: Randomization file not found: {filepath}")
            return False
            
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.json':
                # Load from JSON
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    # Check for activity_sequence key
                    if 'activity_sequence' in data:
                        sequence = data['activity_sequence']
                    elif 'sequence' in data:
                        sequence = data['sequence']
                    else:
                        # Try to interpret the whole dict as a sequence
                        sequence = [v for k, v in data.items()]
                elif isinstance(data, list):
                    # Use the list directly
                    sequence = data
                else:
                    self.log(f"Error: Invalid JSON format in {filepath}")
                    return False
                    
            elif file_ext == '.csv':
                # Load from CSV
                sequence = []
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) > 0:
                            # Take first column value
                            activity = row[0].strip().lower()
                            if activity in ['imagery', 'rest', 'motor_imagery']:
                                if activity == 'motor_imagery':
                                    activity = 'imagery'  # normalize naming
                                sequence.append(activity)
                
            else:  # .txt or other
                # Load from simple text file (one activity per line)
                sequence = []
                with open(filepath, 'r') as f:
                    for line in f:
                        activity = line.strip().lower()
                        if activity in ['imagery', 'rest', 'motor_imagery']:
                            if activity == 'motor_imagery':
                                activity = 'imagery'  # normalize naming
                            sequence.append(activity)
            
            # Validate the sequence contains only valid activity types
            valid_sequence = []
            for item in sequence:
                if item in ['imagery', 'rest']:
                    valid_sequence.append(item)
                else:
                    self.log(f"Warning: Invalid activity type '{item}' in sequence, skipping")
            
            if not valid_sequence:
                self.log("Error: No valid activities found in randomization file")
                return False
                
            # Update trial activity sequence
            self.trial_activity_sequence = valid_sequence
            self.log(f"Loaded {len(valid_sequence)} trials from randomization file")
            
            # Update total trials to match sequence length
            self.total_trials.set(len(valid_sequence))
            
            return True
            
        except Exception as e:
            self.log(f"Error loading randomization file: {e}")
            return False
    
    def toggle_video_settings(self):
        """Enable/disable video file controls based on checkbox"""
        if self.use_instruction_video.get():
            self.video_path_entry.config(state="normal")
            self.video_browse_button.config(state="normal")
        else:
            self.video_path_entry.config(state="disabled")
            self.video_browse_button.config(state="disabled")
            
    def browse_video_file(self):
        """Open file dialog to select a video file"""
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.wmv *.mkv'),
            ('MP4 files', '*.mp4'),
            ('AVI files', '*.avi'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Instruction Video",
            filetypes=filetypes
        )
        
        if filename:
            self.video_file_path.set(filename)
            self.log(f"Selected video file: {filename}")

    def on_monitor_selected(self, event=None):
        """Handle monitor selection change"""
        idx = self.monitor_dropdown.current()
        if 0 <= idx < len(self.monitor_indices):
            monitor_idx = self.monitor_indices[idx]
            self.selected_monitor.set(monitor_idx)
            self.log(f"Selected monitor {monitor_idx}")
        else:
            self.log("Invalid monitor selection")

    def to_hex(self, value):
        return str(hex(value))

    def update_label(self, label_widget, val):
        # Ensure minimum value for baseline scales is respected in label
        if label_widget == self.baseline_val_label:
            fval = max(float(val), self.min_baseline_duration)
            label_widget.config(text=f"{fval:.1f} s")
        else:
            label_widget.config(text=f"{int(float(val))} s")

    def update_cue(self, symbol, title):
        """Update the PsychoPy window to show the current cue"""
        if not self.psychopy_window:
            self.log("Warning: PsychoPy window not available for cue update")
            return
            
        try:
            # Update the text stimulus
            if self.cue_text:
                # Set the text
                self.cue_text.setText(symbol)
                
                # Update window title (may not work on all systems)
                try:
                    self.psychopy_window.winHandle.set_caption(title)
                except:
                    pass
                
                # Make sure evaluation is not showing
                self.evaluation_showing = False
                
                # Log the actual content for debugging
                self.log(f"Setting cue text to: '{symbol}' (visible: {bool(symbol)})")
                
                # Force a redraw and flip
                if symbol:  # Only draw if there's actual content
                    self.cue_text.draw()
                self.psychopy_window.flip()
                
                # Log after flip
                self.log(f"Cue updated to: '{symbol}' (Title: {title})")
            else:
                self.log("Warning: PsychoPy text stimulus not initialized")
        except Exception as e:
            self.log(f"Error updating cue: {e}")
            # Continue despite errors to keep the experiment running

    def get_available_com_ports(self):
        """Scan for available COM ports and update dropdown"""
        self.available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if not self.available_ports:
            self.log("No COM ports found. Please connect a device.")
            self.port_dropdown['values'] = ["No ports found"]
            self.port_dropdown.current(0)
            self.com_port.set("")
        else:
            self.port_dropdown['values'] = self.available_ports
            # If current selection is no longer available, select first port
            if self.com_port.get() not in self.available_ports:
                self.port_dropdown.current(0)
                self.com_port.set(self.available_ports[0])
            self.log(f"Found {len(self.available_ports)} COM ports: {', '.join(self.available_ports)}")

    def get_available_monitors(self):
        """Get available monitors using PsychoPy"""
        if not PSYCHOPY_AVAILABLE:
            self.log("PsychoPy not available, cannot detect monitors")
            self.available_monitors = [(0, "Primary Monitor")]
            self.monitor_indices = [0]
            self.monitor_dropdown['values'] = ["Primary Monitor"]
            self.monitor_dropdown.current(0)
            self.selected_monitor.set(0)
            return
            
        self.available_monitors = []
        self.monitor_indices = []
        
        try:
            # Get all monitors from PsychoPy
            all_monitors = monitors.getAllMonitors()
            
            # First add the default monitor (usually primary)
            primary_mon = monitors.Monitor('default')
            width, height = primary_mon.getSizePix() if primary_mon.getSizePix() else (1920, 1080)
            self.available_monitors.append((0, f"Primary Monitor - {width}x{height}"))
            self.monitor_indices.append(0)
            
            # Then add any additional monitors
            for i, mon_name in enumerate(all_monitors, 1):
                if mon_name != 'default':
                    mon = monitors.Monitor(mon_name)
                    width, height = mon.getSizePix() if mon.getSizePix() else (1920, 1080)
                    self.available_monitors.append((i, f"{mon_name} - {width}x{height}"))
                    self.monitor_indices.append(i)
            
            # If no monitors found in PsychoPy, use system info
            if len(self.available_monitors) <= 1:
                # Try using system-specific methods to get more monitors
                try:
                    # Try to get screen resolution using tkinter
                    self.root.update_idletasks()
                    width = self.root.winfo_screenwidth()
                    height = self.root.winfo_screenheight()
                    
                    # Add the primary monitor
                    if not self.available_monitors:
                        self.available_monitors.append((0, f"Primary Monitor - {width}x{height}"))
                        self.monitor_indices.append(0)
                    
                    # Check for secondary monitor (very simple approach)
                    # This is system-dependent and may not work reliably
                    if width > 2000:  # Heuristic: if screen width is very large, might be multiple monitors
                        self.available_monitors.append((1, f"Secondary Monitor (estimated)"))
                        self.monitor_indices.append(1)
                except Exception as e:
                    self.log(f"Error getting monitor info: {e}")
                    
            # Update the dropdown with monitor options
            monitor_values = [desc for _, desc in self.available_monitors]
            self.monitor_dropdown['values'] = monitor_values
            
            # Set selection if needed
            if self.monitor_dropdown.current() < 0 and monitor_values:
                self.monitor_dropdown.current(0)
                self.selected_monitor.set(self.monitor_indices[0])
                
            self.log(f"Found {len(self.available_monitors)} monitor(s)")
            
        except Exception as e:
            self.log(f"Error detecting monitors: {e}")
            import traceback
            self.log(traceback.format_exc())
            
            # Fallback
            self.available_monitors = [(0, "Primary Monitor")]
            self.monitor_indices = [0]
            self.monitor_dropdown['values'] = ["Primary Monitor"]
            self.monitor_dropdown.current(0)
            self.selected_monitor.set(0)

    def setup_serial(self):
        """Initialize the serial connection with selected COM port"""
        try:
            port = self.com_port.get()
            if not port:
                self.log("Error: No COM port selected")
                messagebox.showerror("Error", "Please select a COM port")
                return False
            
            baud = self.baud_rate.get()
            self.serial_connection = serial.Serial(port, baud, timeout=1)
            self.log(f"Connected to {port} at {baud} baud")
            return True
        
        except Exception as e:
            self.log(f"Error connecting to serial port: {e}")
            messagebox.showerror("Serial Connection Error", f"Could not connect to {port}:\n{e}")
            return False

    def create_psychopy_window(self):
        """Create a PsychoPy window on the selected monitor"""
        if not PSYCHOPY_AVAILABLE:
            self.log("Error: PsychoPy is not available. Please install with: pip install psychopy")
            messagebox.showerror("Error", "PsychoPy is not installed. Please install with: pip install psychopy")
            self.stop_session()
            return False
            
        # Close any existing window
        if hasattr(self, 'psychopy_window') and self.psychopy_window:
            try:
                self.psychopy_window.close()
            except:
                pass
            
        # Get the selected monitor index
        monitor_idx = self.selected_monitor.get()
        self.log(f"Attempting to create PsychoPy window on monitor {monitor_idx}")
        
        try:
            # Get screen information using tkinter
            self.root.update_idletasks()
            main_screen_width = self.root.winfo_screenwidth()
            main_screen_height = self.root.winfo_screenheight()
            
            # Different approach based on monitor index
            if monitor_idx == 0:
                # Primary monitor
                self.log(f"Creating window on primary monitor ({main_screen_width}x{main_screen_height})")
                self.psychopy_window = visual.Window(
                    size=(main_screen_width, main_screen_height),
                    fullscr=True,
                    color='gray',  # Explicitly use 'gray' instead of (-1,-1,-1)
                    units='norm',
                    allowGUI=False,
                    screen=0
                )
            else:
                # For secondary monitor, try with specific position
                # Assuming secondary monitor is to the right of primary (most common setup)
                self.log(f"Creating window on secondary monitor at position ({main_screen_width}, 0)")
                pos = (main_screen_width, 0)  # Position to the right of primary monitor
                
                # Try different creation methods for different PsychoPy versions
                try:
                    # First try with explicit position
                    self.psychopy_window = visual.Window(
                        size=(main_screen_width, main_screen_height),  # Assume same size as primary
                        fullscr=True,
                        color='gray',
                        units='norm',
                        allowGUI=False,
                        pos=pos,  # Position at start of second monitor
                        screen=1
                    )
                    self.log("Created window with explicit position")
                except Exception as pos_err:
                    self.log(f"Error creating window with position: {pos_err}")
                    # If explicit positioning fails, try just with screen index
                    try:
                        self.psychopy_window = visual.Window(
                            fullscr=True,
                            color='gray',
                            units='norm',
                            allowGUI=False,
                            screen=1  # Try using screen=1 for secondary monitor
                        )
                        self.log("Created window with screen=1")
                    except Exception as screen_err:
                        self.log(f"Error creating window with screen=1: {screen_err}")
                        # Last resort: create on primary monitor
                        self.log("Falling back to primary monitor")
                        self.psychopy_window = visual.Window(
                            fullscr=True,
                            color='gray',
                            units='norm',
                            allowGUI=False
                        )
            
            # Create text stimuli with bright colors to ensure visibility
            self.cue_text = visual.TextStim(
                win=self.psychopy_window,
                text='',
                font='Arial',
                pos=(0, 0),
                height=0.3,
                wrapWidth=None,
                color='white',  # Use string 'white' instead of (1,1,1)
                bold=True,      # Make text bold
                opacity=1.0     # Ensure full opacity
            )
            
            self.question_text = visual.TextStim(
                win=self.psychopy_window,
                text='Did you perform the activity?',
                font='Arial',
                pos=(0, 0.0),
                height=0.1,
                wrapWidth=None,
                color='white',
                bold=True,
                opacity=1.0
            )
            
            self.instruction_text = visual.TextStim(
                win=self.psychopy_window,
                text='Press Y for YES or N for NO',
                font='Arial',
                pos=(0, -0.2),
                height=0.07,
                wrapWidth=None,
                color='white',
                bold=True,
                opacity=1.0
            )

            self.instruction_stim = visual.TextStim(
                win=self.psychopy_window,
                text=self.instructions,
                font='Arial',
                pos=(0, 0),
                height=0.05,  # Smaller text for longer content
                wrapWidth=1.8,  # Allow text wrapping
                color='white',
                alignText='center',
                anchorHoriz='center',
                anchorVert='center',
                bold=False
            )

            # After creating other text stimuli, add:
            self.landing_stim = visual.TextStim(
                win=self.psychopy_window,
                text=self.landing_text,
                font='Arial',
                pos=(0, 0),
                height=0.07,
                wrapWidth=1.8,
                color='white',
                alignText='center',
                anchorHoriz='center',
                anchorVert='center',
                bold=True
            )

            # Create video stimulus if video file is specified
            if self.use_instruction_video.get() and self.video_file_path.get():
                try:
                    self.video_stimulus = visual.MovieStim(
                        win=self.psychopy_window,
                        filename=self.video_file_path.get(),
                        pos=(0, 0),
                        size=None,  # Use original video size
                        flipVert=False,
                        flipHoriz=False,
                        volume=1.0
                    )
                    self.log("Video stimulus created successfully")
                except Exception as e:
                    self.log(f"Error creating video stimulus: {e}")
                    self.video_stimulus = None
            
            # Draw initial text and flip to show it's working
            if self.show_instructions:
                self.instruction_stim.draw()
                self.psychopy_window.flip()
            
            # Start PsychoPy updates using Tkinter's event loop
            self.schedule_psychopy_update()
            
            self.log("PsychoPy window created successfully")
            return True
            
        except Exception as e:
            self.log(f"Error creating PsychoPy window: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to create PsychoPy window:\n{e}")
            return False

    def schedule_psychopy_update(self):
        """Schedule periodic updates for PsychoPy in Tkinter's event loop"""
        # Define the update function that will be called repeatedly
        def update_psychopy():
            try:
                # Only process if we have a valid window and are running
                if self.psychopy_window and self.running:
                    # Check for key presses with better output
                    keys = event.getKeys(keyList=['y', 'Y', 'n', 'N', 'escape', 'space'])
                    
                    # Process keys
                    if keys:
                        self.log(f"Key pressed: {keys}")
                        
                        # Handle Escape key
                        if 'escape' in keys and self.running:
                            self.stop_session()
                        
                        # Handle spacebar for instructions
                        if 'space' in keys and self.show_instructions:
                            self.show_instructions = False
                            
                            # Check if we should show video
                            if self.use_instruction_video.get() and self.video_stimulus:
                                self.showing_video = True
                                self.video_stimulus.seek(0)  # Reset to beginning
                                self.video_stimulus.play()
                                self.log("Instructions complete, playing instruction video")
                            else:
                                # No video, go to landing page
                                self.showing_landing = True
                                self.landing_stim.draw()
                                self.psychopy_window.flip()
                                self.log("Instructions complete, showing landing page")
                        
                        # Handle landing page responses
                        if self.showing_landing:
                            if 'y' in keys or 'Y' in keys:
                                # Replay video
                                self.showing_landing = False
                                if self.video_stimulus:
                                    self.showing_video = True
                                    self.video_stimulus.seek(0)  # Reset to beginning
                                    self.video_stimulus.play()
                                    self.log("Replaying instruction video")
                                else:
                                    # If no video available, just show landing again
                                    self.landing_stim.draw()
                                    self.psychopy_window.flip()
                                    self.log("No video to replay")
                            
                            elif 'space' in keys or 'n' in keys or 'N' in keys:
                                # Proceed to experiment
                                self.showing_landing = False
                                self.psychopy_window.flip()  # Clear screen
                                self.log("Landing page complete, starting trials")
                                self.root.after(100, self.start_trial)
                        
                        # Handle evaluation keys (only when waiting for evaluation response)
                        elif self.waiting_for_response:
                            if 'y' in keys or 'Y' in keys:
                                self.log("Y key detected during evaluation")
                                self.evaluation_response(True)
                            elif 'n' in keys or 'N' in keys:
                                self.log("N key detected during evaluation")
                                self.evaluation_response(False)

                # Handle video playback
                if self.showing_video and self.video_stimulus:
                    # Draw the video frame
                    self.video_stimulus.draw()
                    self.psychopy_window.flip()
                    
                    # Check if video is finished
                    if self.video_stimulus.status == visual.FINISHED:
                        self.showing_video = False
                        self.video_stimulus.stop()
                        self.showing_landing = True
                        
                        # Show landing page
                        self.landing_stim.draw()
                        self.psychopy_window.flip()
                        self.log("Video finished, showing landing page")
                
                # Schedule the next update if we're still running
                if self.running:
                    self.root.after(30, update_psychopy)  # ~33 FPS
                
            except Exception as e:
                # Log and reschedule even on error
                self.log(f"Error in PsychoPy update: {e}")
                if self.running:
                    self.root.after(100, update_psychopy)  # Try again after a delay
        
        # Start the update cycle
        self.root.after(100, update_psychopy)
        self.log("Scheduled PsychoPy updates in Tkinter event loop")

    def send_marker(self, marker_label):
        """Send a marker through the serial port"""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                self.log(f"Error: Cannot send marker '{marker_label}' - Serial connection not open")
                return
            
            # Send the marker as a string
            marker_str = f"{marker_label}"
            self.serial_connection.write(marker_str.encode('utf-8'))
            
            # Store the marker with timestamp
            marker_timestamp = time.time()
            self.markers.append((marker_label, marker_timestamp))
            
            # Calculate relative time if recording has started
            if self.recording_start_time is not None:
                relative_time = marker_timestamp - self.recording_start_time
                self.log(f"Marker '{marker_label}' sent at {relative_time:.4f}s")
            else:
                self.log(f"Marker '{marker_label}' sent")
                
        except Exception as e:
            self.log(f"Error sending marker '{marker_label}': {e}")

    def show_evaluation_screen(self):
        """Show the evaluation screen with question based on activity type"""
        if not self.psychopy_window:
            self.log("Warning: PsychoPy window not available for evaluation screen")
            return
            
        try:
            self.evaluation_showing = True
            self.waiting_for_response = True
            
            # Choose question based on activity type
            if self.current_activity == 'imagery':
                self.question_text.setText(self.imagery_question)
            else:  # rest
                self.question_text.setText(self.rest_question)
            
            # Draw evaluation screen components
            self.question_text.draw()
            self.instruction_text.draw()
            
            # Flip to show the new content
            self.psychopy_window.flip()
            
            self.log(f"Evaluation screen shown for {self.current_activity}, waiting for Y/N response")
        except Exception as e:
            self.log(f"Error showing evaluation screen: {e}")

    def hide_evaluation_screen(self):
        """Hide the evaluation screen"""
        if not self.psychopy_window:
            return
            
        try:
            self.evaluation_showing = False
            self.waiting_for_response = False
            
            # Just flip to clear the screen (don't draw anything)
            self.psychopy_window.flip()
            
            self.log("Evaluation screen hidden")
        except Exception as e:
            self.log(f"Error hiding evaluation screen: {e}")

    def start_session(self):
        # --- Input Validation ---
        # Check subject number
        subject = self.subject_number.get().strip()
        if not subject:
            messagebox.showerror("Error", "Please enter a subject number")
            return
            
        # Check serial port
        if not self.setup_serial():
            return
            
        n_trials = self.total_trials.get()
        if n_trials <= 0:
            messagebox.showerror("Error", "Total trials must be greater than 0.")
            return

        imagery_selected = self.select_motor_imagery.get()
        rest_selected = self.select_rest.get()
        activity_selected = imagery_selected or rest_selected

        if not activity_selected:
            messagebox.showerror("Error", "At least one activity type (Motor Imagery or Rest) must be selected.")
            return
        
        

        # --- Generate Trial Sequence ---
        self.trial_activity_sequence = []
        self.eval_results = []  # Reset evaluation results

        # Try to load sequence from file if requested
        if self.use_randomization_file.get() and self.randomization_file_path.get():
            if not self.load_randomization_from_file():
                # Failed to load file - ask if we should continue
                if not messagebox.askyesno("Warning", 
                    "Failed to load randomization file. Continue with auto-generated sequence?"):
                    return  # User chose to cancel
                    
                # Otherwise, continue with auto-generated sequence
        
        # If we don't have a sequence yet, generate one
        if not self.trial_activity_sequence:
            # Activity Sequence (Imagery vs. Rest)
            if imagery_selected and rest_selected:
                self.trial_activity_sequence = self.generate_counterbalanced_sequence(n_trials)
                self.log(f"Generated counterbalanced sequence for {n_trials} trials.")
            elif imagery_selected:
                self.trial_activity_sequence = ['imagery'] * n_trials
            else: # only rest_selected
                self.trial_activity_sequence = ['rest'] * n_trials

        # --- Start Session ---
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.markers = []
        self.current_trial = 0
        self.trial_label.config(text=f"Trial: 0 / {n_trials}")
        self.show_instructions = True

        # Create the PsychoPy window
        if not self.create_psychopy_window():
            self.stop_session()
            return

        # Start the session
        self.recording_start_time = time.time() # Wall clock start time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get pre/post status
        session_type = "POST" if self.is_post_assessment.get() else "PRE"
        
        self.log(f"Started {session_type} assessment for subject {subject} at {timestamp}")
        self.send_marker(self.to_hex(self.session_start)) # Mark session start

        # Start the first trial after a short delay
        self.root.after(500, self.start_trial)
    
    def stop_session(self):
        if not self.running: # Prevent double stop
            return
        self.running = False
        self.log("Stopping session...")
        self.send_marker(self.to_hex(self.session_end)) # Mark session end

        # Close the serial connection
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.log("Serial connection closed")

        # Close the PsychoPy window
        if hasattr(self, 'psychopy_window') and self.psychopy_window:
            try:
                self.psychopy_window.close()
                self.log("PsychoPy window closed")
            except:
                self.log("Error closing PsychoPy window")
            self.psychopy_window = None

        # Cancel any pending `after` calls to prevent errors
        for after_id in self.root.tk.call('after', 'info'):
            try:
                self.root.after_cancel(after_id)
            except:
                pass  # Ignore errors when canceling

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        self.log(f"Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Export marker log and evaluation results
        self.export_marker_log()
        self.export_evaluation_results()

    def export_marker_log(self):
        """Export markers to a text file for reference"""
        if not self.markers:
            self.log("No markers to export")
            return
            
        try:
            # Create filename with subject number and pre/post info
            subject = self.subject_number.get().strip()
            session_type = "POST" if self.is_post_assessment.get() else "PRE"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{self.save_dir}S{subject}_{session_type}_marker_log_{timestamp_str}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"Session Marker Log - Subject {subject} - {session_type} Assessment\n")
                f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n\n")
                f.write("Relative Time (s) | Marker\n")
                f.write("-"*50 + "\n")
                
                for marker, timestamp in self.markers:
                    relative_time = timestamp - self.recording_start_time
                    f.write(f"{relative_time:15.4f} | {marker}\n")
            
            self.log(f"Marker log exported to {filename}")
            messagebox.showinfo("Export Complete", f"Marker log saved to {filename}")
            
        except Exception as e:
            self.log(f"Error exporting marker log: {e}")
            messagebox.showerror("Export Error", f"Failed to export marker log: {e}")

    def export_evaluation_results(self):
        """Export evaluation results to a text file"""
        if not self.eval_results:
            self.log("No evaluation results to export")
            return
        
        try:
            # Create filename with subject number and pre/post info
            subject = self.subject_number.get().strip()
            session_type = "POST" if self.is_post_assessment.get() else "PRE"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{self.save_dir}S{subject}_{session_type}_eval_results_{timestamp_str}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"Evaluation Results - Subject {subject} - {session_type} Assessment\n")
                f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n\n")
                f.write("Trial | Activity Type | Response | Response Time (s)\n")
                f.write("-"*70 + "\n")
                
                for trial, activity, response, response_time in self.eval_results:
                    f.write(f"{trial:5d} | {activity:12s} | {response:8s} | {response_time:14.2f}\n")
                
                # Add summary statistics
                f.write("\nSummary Statistics:\n")
                f.write("-"*70 + "\n")
                
                # Calculate stats for imagery trials
                imagery_responses = [r for t, a, r, rt in self.eval_results if a == 'imagery']
                if imagery_responses:
                    imagery_yes = sum(1 for r in imagery_responses if r == 'YES')
                    imagery_pct = (imagery_yes / len(imagery_responses)) * 100
                    f.write(f"Motor Imagery Trials: {len(imagery_responses)}, YES responses: {imagery_yes} ({imagery_pct:.1f}%)\n")
                
                # Calculate stats for rest trials
                rest_responses = [r for t, a, r, rt in self.eval_results if a == 'rest']
                if rest_responses:
                    rest_yes = sum(1 for r in rest_responses if r == 'YES')
                    rest_pct = (rest_yes / len(rest_responses)) * 100
                    f.write(f"Rest Trials: {len(rest_responses)}, YES responses: {rest_yes} ({rest_pct:.1f}%)\n")
                
                # Overall stats
                total_yes = sum(1 for t, a, r, rt in self.eval_results if r == 'YES')
                total_pct = (total_yes / len(self.eval_results)) * 100
                f.write(f"Total Trials: {len(self.eval_results)}, YES responses: {total_yes} ({total_pct:.1f}%)\n")
            
            self.log(f"Evaluation results exported to {filename}")
            messagebox.showinfo("Export Complete", f"Evaluation results saved to {filename}")
            
            # Also save as CSV for data analysis
            csv_filename = f"{self.save_dir}S{subject}_{session_type}_eval_results_{timestamp_str}.csv"
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Session', 'Trial', 'Activity', 'Response', 'ResponseTime'])
                
                for trial, activity, response, response_time in self.eval_results:
                    writer.writerow([subject, session_type, trial, activity, response, f"{response_time:.4f}"])
            
            self.log(f"Evaluation results also saved as CSV: {csv_filename}")
            
        except Exception as e:
            self.log(f"Error exporting evaluation results: {e}")
            messagebox.showerror("Export Error", f"Failed to export evaluation results: {e}")

    def start_trial(self):
        """Initiates a single trial."""
        # Don't start trials if still showing instructions, video, or landing page
        if self.show_instructions or self.showing_video or self.showing_landing:
            return
            
        if not self.running or self.current_trial >= self.total_trials.get():
            if self.running: # If stopped externally, stop_session handles logging/saving
                self.stop_session()
            return

        self.current_trial += 1
        self.trial_label.config(text=f"Trial: {self.current_trial} / {self.total_trials.get()}")
        self.log(f"\n--- Starting Trial {self.current_trial} ---")

        # Start with baseline phase
        self.start_baseline_phase()

    def start_baseline_phase(self):
        """Handles the baseline period."""
        self.update_cue("+", "Baseline")
        self.send_marker(self.to_hex(self.baseline))

        # Use max duration from slider, randomize between min and max
        max_duration = self.baseline_duration.get()
        duration_randomized = random.uniform(self.min_baseline_duration, max_duration)
        self.log(f"Phase: Baseline ({duration_randomized:.2f} s)")

        # Next will be the activity phase
        self.root.after(int(duration_randomized * 1000), self.start_activity_phase)

    def start_activity_phase(self):
        """Handles the activity phase (Imagery or Rest)."""
        # Get the activity for the current trial from the pre-shuffled list
        activity = self.trial_activity_sequence[self.current_trial - 1]
        
        if activity == 'imagery':
            self.update_cue("•", "Motor Imagery")
            marker_start = self.to_hex(self.motor_imagery)
            duration = self.imagery_duration.get()
            self.log(f"Phase: Motor Imagery ({duration} s)")
        else: # activity == 'rest'
            self.update_cue("", "Rest") # Blank screen for rest
            marker_start = self.to_hex(self.rest)
            duration = self.rest_duration.get()
            self.log(f"Phase: Rest ({duration} s)")

        self.send_marker(marker_start)
        
        # Schedule evaluation phase
        self.root.after(duration * 1000, self.start_evaluation_phase, activity)

    def start_evaluation_phase(self, activity_type):
        """Start the evaluation phase after an activity."""
        # Clear any previous cue
        self.update_cue("", "Evaluation")
        
        # Send marker
        self.send_marker(self.to_hex(self.eval_start))
        
        # Store current activity type for the evaluation
        self.current_activity = activity_type
        
        # Store start time for response time measurement
        self.eval_start_time = time.time()
        
        # Add a slight delay before showing eval screen to ensure transition is visible
        self.root.after(100, self.show_evaluation_screen)
        
        self.log("Evaluation phase started - waiting for Y/N keypress")
        
        # Note: No timeout - waiting for user keystroke

    def evaluation_response(self, response):
        """Handle yes/no response during evaluation period."""
        if not self.waiting_for_response:
            return  # Prevent double responses
            
        # Calculate response time
        response_time = time.time() - self.eval_start_time
        
        # Hide evaluation screen
        self.hide_evaluation_screen()
        
        # Log and store the response
        if response:
            self.log(f"Evaluation: YES (response time: {response_time:.2f}s)")
            self.send_marker(self.to_hex(self.eval_yes))
        else:
            self.log(f"Evaluation: NO (response time: {response_time:.2f}s)")
            self.send_marker(self.to_hex(self.eval_no))
            
        # Store result
        self.eval_results.append((
            self.current_trial, 
            self.current_activity, 
            "YES" if response else "NO",
            response_time
        ))
        
        # Move to next trial
        self.root.after(500, self.start_trial)  # Small delay before next trial

    def generate_counterbalanced_sequence(self, n_trials):
        """Generate a counterbalanced sequence of trials
        
        This ensures:
        1. Equal (or as close as possible) number of each trial type
        2. Balanced transitions between trial types
        3. No more than 3 consecutive trials of the same type
        """
        # Determine number of each trial type
        n_imagery = n_trials // 2
        n_rest = n_trials - n_imagery  # Handles odd trial counts
        
        # Start with a basic alternating sequence to ensure transitions
        seq = []
        for i in range(n_trials):
            if i % 2 == 0:
                if len([t for t in seq if t == 'imagery']) < n_imagery:
                    seq.append('imagery')
                else:
                    seq.append('rest')
            else:
                if len([t for t in seq if t == 'rest']) < n_rest:
                    seq.append('rest')
                else:
                    seq.append('imagery')
        
        # Apply additional randomization while maintaining counterbalancing
        # We'll work with blocks of 2 trials to preserve transitions
        blocks = [seq[i:i+2] for i in range(0, len(seq), 2)]
        random.shuffle(blocks)
        
        # Flatten the blocks back into a sequence
        final_seq = [item for block in blocks for item in block]
        
        # Verify counterbalancing
        img_count = final_seq.count('imagery')
        rest_count = final_seq.count('rest')
        self.log(f"Counterbalanced sequence: {img_count} imagery trials, {rest_count} rest trials")
        
        # Check for runs of more than 3 consecutive trials
        has_long_runs = False
        for i in range(len(final_seq) - 3):
            if (final_seq[i] == final_seq[i+1] == final_seq[i+2] == final_seq[i+3]):
                has_long_runs = True
                break
        
        # If we have runs of 4 or more, recursively regenerate
        if has_long_runs and n_trials > 6:  # Only for longer sequences
            self.log("Detected runs of 4+ consecutive trials, regenerating sequence")
            return self.generate_counterbalanced_sequence(n_trials)
        
        return final_seq

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


if __name__ == "__main__":
    # Check for PsychoPy
    if not PSYCHOPY_AVAILABLE:
        print("WARNING: PsychoPy is not installed. Installing PsychoPy...")
        try:
            import pip
            pip.main(['install', 'psychopy'])
            print("PsychoPy installed successfully. Please restart the application.")
        except:
            print("ERROR: Could not install PsychoPy. Please install manually with:")
            print("pip install psychopy")
        input("Press Enter to exit...")
        sys.exit(1)
    
    root = tk.Tk()
    app = EEGMarkerGUI(root)
    root.mainloop()