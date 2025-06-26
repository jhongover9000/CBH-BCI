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

        # Directories
        self.video_dir = "./mi_assessment/resources/finger_tap_ex_1080.mp4"
        self.order_dir = "./mi_assessment/resources/counterbalanced_order.txt"
        self.save_directory = "./mi_assessment/saves/"
        self.debug_mode = tk.BooleanVar(value=False)
        self.auto_participant_management = tk.BooleanVar(value=True)

        # Marker Values
        self.baseline = 2
        self.motor_imagery = 3
        self.rest = 4
        self.eval_start = 13
        self.eval_yes = 14
        self.eval_no = 15

        # Subject and session information
        self.subject_number = tk.StringVar(value="")
        self.is_post_assessment = tk.BooleanVar(value=False)
        
        # Randomization options
        self.use_randomization_file = tk.BooleanVar(value=True)
        self.randomization_file_path = tk.StringVar(value=self.order_dir)

        # Adjustable durations (in seconds)
        self.baseline_duration = tk.IntVar(value=4)     # Baseline duration
        self.imagery_duration = tk.IntVar(value=3)      # Motor imagery duration
        self.rest_duration = tk.IntVar(value=3)         # Rest duration

        self.total_trials = tk.IntVar(value=40)

        self.rest_type = "text" # options: text (read neutral word), shape (complex shape/image), fixation (big fixation cue) 

        # Minimum baseline duration (in seconds) for randomization
        self.min_baseline_duration = 2.5

        # Activity selection checkboxes
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        # --- Serial Port Variables ---
        self.com_port = tk.StringVar(value="COM4")
        self.available_ports = []
        self.serial_connection = None
        self.baud_rate = tk.IntVar(value=2000000)

        # --- Monitor Selection Variables ---
        self.available_monitors = []
        self.selected_monitor = tk.IntVar(value=1)  # Default to primary monitor

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
        self.video_file_path = tk.StringVar(value=self.video_dir)
        self.use_instruction_video = tk.BooleanVar(value=True)
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
        if(self.rest_type == "fixation"):
            self.rest_question = "Were you able to maintain a resting state?"
        elif self.rest_type == "text":
            self.rest_question = "Were you able to maintain fixation?"
        self.instructions = "Welcome to the Motor Imagery Assessment.\n\n\n\nAfter a baseline period, you will be asked to either imagine a motor movement or maintain a resting state. After each task, you will evaluate your performance.\n\n+ : Fixation\n\u2022 : Motor Imagery\n Blank Screen :  Rest\n\nPlease let the experimenter know when you are ready to view the motor movement to imagine."
        self.show_instructions = True

        # Sound
        self.sound_system_available = False
        self.sound_backend = "unknown"
        self.beep_sound = None
        self.use_baseline_beep = tk.BooleanVar(value=True)
        self.beep_frequency = 400  # Hz
        self.beep_duration = 0.1   # seconds
        self.sound_fallback_method = tk.StringVar(value="system")  # "system", "visual", "none"
        
        self.setup_ui()
        self.get_available_com_ports()
        self.get_available_monitors()

    def initialize_sound_system(self):
        self.sound_system_available = False
        self.sound_backend = "none"
        
        # Try different PsychoPy sound backends
        sound_backends = ['ptb', 'pygame', 'pyo', 'sounddevice']
        
        for backend in sound_backends:
            try:
                from psychopy import prefs
                prefs.hardware['audioLib'] = [backend]
                
                from psychopy import sound
                # Force reload of sound module
                import importlib
                importlib.reload(sound)
                
                # Test creating a simple sound
                test_sound = sound.Sound(value=440, secs=0.1)
                
                self.sound_system_available = True
                self.sound_backend = backend
                self.log(f"Sound system initialized successfully with {backend} backend")
                break
                
            except Exception as e:
                self.log(f"Failed to initialize {backend} sound backend: {e}")
                continue
        
        if not self.sound_system_available:
            self.log("WARNING: No PsychoPy sound backend available. Using fallback methods.")
            return False
        
        return True

    def create_beep_sound(self):
        """Create beep sound with error handling"""
        if not self.sound_system_available:
            return False
        
        try:
            # # Update beep parameters from UI
            # try:
            #     self.beep_frequency = float(self.freq_var.get())
            #     self.beep_duration = float(self.duration_var.get())
            # except (ValueError, AttributeError):
            #     self.log("Invalid beep parameters, using defaults")
            #     self.beep_frequency = 400
            #     self.beep_duration = 0.1
            
            from psychopy import sound
            self.beep_sound = sound.Sound(
                value=self.beep_frequency,
                secs=self.beep_duration,
                stereo=True,
                volume=0.5,
                loops=0
            )
            self.log(f"Beep sound created: {self.beep_frequency}Hz for {self.beep_duration}s")
            return True
            
        except Exception as e:
            self.log(f"Error creating beep sound: {e}")
            self.beep_sound = None
            return False

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

        # Auto management checkbox
        ttk.Checkbutton(subj_frame, text="Auto-manage participant numbers", 
            variable=self.auto_participant_management,
            command=self.toggle_auto_management).grid(
                row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Debug mode checkbox
        ttk.Checkbutton(subj_frame, text="Debug mode (separate files)", 
              variable=self.debug_mode).grid(
                row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Manual subject number (disabled when auto mode is on)
        ttk.Label(subj_frame, text="Subject Number:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.subject_entry = ttk.Entry(subj_frame, textvariable=self.subject_number, width=10)
        self.subject_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # Manual assessment type (disabled when auto mode is on)
        ttk.Label(subj_frame, text="Assessment Type:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.assessment_frame = ttk.Frame(subj_frame)
        self.assessment_frame.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        self.pre_radio = ttk.Radiobutton(self.assessment_frame, text="Pre", variable=self.is_post_assessment, value=False)
        self.pre_radio.pack(side="left")
        self.post_radio = ttk.Radiobutton(self.assessment_frame, text="Post", variable=self.is_post_assessment, value=True)
        self.post_radio.pack(side="left")
        self.pre_radio = ttk.Radiobutton(self.assessment_frame, text="Pre", 
                                        variable=self.is_post_assessment, value=False,
                                        command=self.on_assessment_type_change)
        self.post_radio = ttk.Radiobutton(self.assessment_frame, text="Post", 
                                        variable=self.is_post_assessment, value=True,
                                        command=self.on_assessment_type_change)

        # Status label to show current settings
        self.participant_status_label = ttk.Label(subj_frame, text="", foreground="blue")
        self.participant_status_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)


        # Update participant info on startup
        self.root.after(100, self.update_participant_info)

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
        # ttk.Label(serial_frame, text="Baud Rate:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        # baud_dropdown = ttk.Combobox(serial_frame, textvariable=self.baud_rate, state="readonly", width=10,
        #                             values=[9600, 19200, 38400, 57600, 115200, 2000000])
        # baud_dropdown.grid(row=1, column=1, sticky="w", padx=5, pady=2)

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
        
        # # File path and browse button
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
        # param_frame = ttk.LabelFrame(right_frame, text="Experiment Parameters")
        # param_frame.grid(row=right_row, column=0, sticky="ew", pady=(0, 10), padx=5)
        # right_row += 1
        
        # param_row = 0
        
        # ttk.Label(param_frame, text="Baseline Duration (s):").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        # baseline_frame = ttk.Frame(param_frame)
        # baseline_frame.grid(row=param_row, column=1, sticky="ew", padx=5, pady=2)
        # param_row += 1
        
        # baseline_scale = ttk.Scale(baseline_frame, from_=self.min_baseline_duration, to=10, orient="horizontal",
        #                           variable=self.baseline_duration, length=150,
        #                           command=lambda val: self.update_label(self.baseline_val_label, val))
        # baseline_scale.pack(side="left", padx=(0, 5))
        
        # self.baseline_val_label = ttk.Label(baseline_frame, text=f"{self.baseline_duration.get()} s", width=5)
        # self.baseline_val_label.pack(side="left")

        # --- Activity Selection ---
        # ttk.Label(param_frame, text="Activity Types:").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        # act_frame = ttk.Frame(param_frame)
        # act_frame.grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
        # param_row += 1
        
        # ttk.Checkbutton(act_frame, text="Motor Imagery", variable=self.select_motor_imagery).pack(side="left", padx=(0, 10))
        # ttk.Checkbutton(act_frame, text="Rest", variable=self.select_rest).pack(side="left")
        
        # Motor Imagery Duration
        # ttk.Label(param_frame, text="Imagery Duration (s):").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        # imagery_frame = ttk.Frame(param_frame)
        # imagery_frame.grid(row=param_row, column=1, sticky="ew", padx=5, pady=2)
        # param_row += 1
        
        # imagery_scale = ttk.Scale(imagery_frame, from_=1, to=10, orient="horizontal",
        #                          variable=self.imagery_duration, length=150,
        #                          command=lambda val: self.update_label(self.imagery_val_label, val))
        # imagery_scale.pack(side="left", padx=(0, 5))
        
        # self.imagery_val_label = ttk.Label(imagery_frame, text=f"{self.imagery_duration.get()} s", width=5)
        # self.imagery_val_label.pack(side="left")
        
        # # Rest Duration
        # ttk.Label(param_frame, text="Rest Duration (s):").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        # rest_frame = ttk.Frame(param_frame)
        # rest_frame.grid(row=param_row, column=1, sticky="ew", padx=5, pady=2)
        # param_row += 1
        
        # rest_scale = ttk.Scale(rest_frame, from_=1, to=10, orient="horizontal",
        #                       variable=self.rest_duration, length=150,
        #                       command=lambda val: self.update_label(self.rest_val_label, val))
        # rest_scale.pack(side="left", padx=(0, 5))
        
        # self.rest_val_label = ttk.Label(rest_frame, text=f"{self.rest_duration.get()} s", width=5)
        # self.rest_val_label.pack(side="left")
        
        # # Total Trials
        # ttk.Label(param_frame, text="Total Trials:").grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
        # ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.total_trials, width=5).grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
        # param_row += 1

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

        self.startup_sound_check()

    def toggle_auto_management(self):
        """Enable/disable manual controls based on auto management setting"""
        if self.auto_participant_management.get():
            self.subject_entry.config(state="disabled")
            self.pre_radio.config(state="disabled")
            self.post_radio.config(state="disabled")
            self.update_participant_info()
        else:
            self.subject_entry.config(state="normal")
            self.pre_radio.config(state="normal")
            self.post_radio.config(state="normal")
            self.participant_status_label.config(text="Manual mode - set subject and assessment type above")

    def ensure_save_directory(self):
        """Create save directory if it doesn't exist"""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            self.log(f"Created save directory: {self.save_directory}")

    def get_existing_participants(self):
        """Scan save directory for existing participant files"""
        self.ensure_save_directory()
        participants = {}
        
        try:
            for filename in os.listdir(self.save_directory):
                if filename.startswith('S') and ('_PRE_' in filename or '_POST_' in filename):
                    # Extract participant number
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        participant_num = parts[0][1:]  # Remove 'S' prefix
                        assessment_type = parts[1]
                        
                        if participant_num not in participants:
                            participants[participant_num] = {'PRE': False, 'POST': False}
                        
                        participants[participant_num][assessment_type] = True
            
            return participants
        except Exception as e:
            self.log(f"Error scanning participants: {e}")
            return {}
        
    # Call this when assessment type changes (add to toggle methods or assessment radio button callbacks)
    def on_assessment_type_change(self):
        """Called when assessment type changes"""
        self.update_video_ui_status()
        if self.auto_participant_management.get():
            self.update_participant_info()

    def get_next_participant_info(self):
        """Determine next participant number and assessment type"""
        participants = self.get_existing_participants()
        
        if not participants:
            # No existing participants
            return "001", False  # First participant, PRE assessment
        
        # Find the highest participant number
        max_num = max([int(p) for p in participants.keys()])
        
        # Check if current max participant needs POST assessment
        max_num_str = f"{max_num:03d}"
        if max_num_str in participants and not participants[max_num_str]['POST']:
            # Current participant needs POST assessment
            return max_num_str, True
        
        # Need new participant for PRE assessment
        next_num = max_num + 1
        return f"{next_num:03d}", False

    def update_participant_info(self):
        """Update participant number and assessment type automatically"""
        if not self.auto_participant_management.get():
            return
        
        try:
            participant_num, is_post = self.get_next_participant_info()
            self.subject_number.set(participant_num)
            self.is_post_assessment.set(is_post)
            
            assessment_type = "POST" if is_post else "PRE"
            status_text = f"Auto: Participant {participant_num} - {assessment_type} assessment"
            self.participant_status_label.config(text=status_text)
            
            self.log(f"Auto-assigned: Participant {participant_num}, {assessment_type} assessment")
            
        except Exception as e:
            self.log(f"Error updating participant info: {e}")
            self.participant_status_label.config(text="Error in auto-assignment")

    def check_file_exists(self, base_filename):
        """Check if a file already exists to prevent overwrites"""
        full_path = os.path.join(self.save_directory, base_filename)
        return os.path.exists(full_path)

    def generate_safe_filename(self, base_name, extension):
        """Generate a safe filename that won't overwrite existing files"""
        subject = self.subject_number.get().strip()
        session_type = "POST" if self.is_post_assessment.get() else "PRE"
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.debug_mode.get():
            # Debug mode - separate naming
            filename = f"DEBUG_S{subject}_{session_type}_{base_name}_{timestamp_str}.{extension}"
        else:
            # Normal mode
            filename = f"S{subject}_{session_type}_{base_name}_{timestamp_str}.{extension}"
        
        # Check if file exists and add counter if needed
        counter = 1
        original_filename = filename
        while self.check_file_exists(filename):
            name_parts = original_filename.rsplit('.', 1)
            filename = f"{name_parts[0]}_v{counter}.{name_parts[1]}"
            counter += 1
        
        return filename

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
                self.monitor_dropdown.current(1)
                self.selected_monitor.set(self.monitor_indices[1])
                
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
                height=0.4,
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
                wrapWidth=2,
                color='white',
                bold=True,
                opacity=1.0
            )
            
            self.instruction_text = visual.TextStim(
                win=self.psychopy_window,
                text='Press Y for YES or N for NO',
                font='Arial',
                pos=(0, -0.2),
                height=0.1,
                wrapWidth=1.5,
                color='white',
                bold=True,
                opacity=1.0
            )

            self.instruction_stim = visual.TextStim(
                win=self.psychopy_window,
                text=self.instructions,
                font='Arial',
                pos=(0, 0),
                height=0.08,  # Smaller text for longer content
                wrapWidth=1.3,  # Allow text wrapping
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
                height=0.1,
                wrapWidth=1.8,
                color='white',
                alignText='center',
                anchorHoriz='center',
                anchorVert='center',
                bold=True
            )

            # Create beep sound stimulus
            self.initialize_sound_system()

            # Create beep sound if possible
            if self.sound_system_available:
                self.create_beep_sound()
            else:
                self.log("Sound system not available - beep cues will use fallback method")

            # Create video stimulus if video file is specified
            if self.use_instruction_video.get() and self.video_file_path.get():
                try:
                    self.video_stimulus = visual.MovieStim3(
                        win=self.psychopy_window,
                        filename=self.video_file_path.get(),
                        pos=(0, 0),
                        size=[1920,1080],  # adjust for monitor!!
                        flipVert=False,
                        flipHoriz=False
                    )
                    self.log("Video stimulus created successfully")
                except Exception as e:
                    self.log(f"Error creating video stimulus: {e}")
                    self.video_stimulus = None
            
            # Draw initial text and flip to show it's working
            if self.show_instructions:
                self.update_instruction_text()
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
        
    def play_beep_fallback(self):
        """Fallback beep methods when PsychoPy sound fails"""
        fallback_method = self.sound_fallback_method.get()
        
        if fallback_method == "system":
            try:
                import winsound
                frequency = int(self.beep_frequency)
                duration = int(self.beep_duration * 1000)  # Convert to milliseconds
                winsound.Beep(frequency, duration)
                self.log(f"System beep played: {frequency}Hz for {duration}ms")
                return True
            except ImportError:
                # Try cross-platform system beep
                try:
                    import os
                    if os.name == 'nt':  # Windows
                        os.system('echo \a')
                    else:  # Unix/Linux/Mac
                        os.system('tput bel')
                    self.log("System bell played")
                    return True
                except:
                    pass
        
        elif fallback_method == "visual":
            # Visual flash as beep substitute
            self.visual_beep_flash()
            return True
        
        # If all else fails
        self.log("No beep method available")
        return False

    def visual_beep_flash(self):
        """Visual flash to substitute for audio beep"""
        if not hasattr(self, 'psychopy_window') or not self.psychopy_window:
            return
        
        try:
            # Create brief white flash
            from psychopy import visual
            flash_stim = visual.Rect(
                win=self.psychopy_window,
                width=2, height=2,
                fillColor='white',
                lineColor='white'
            )
            
            # Flash for brief moment
            flash_stim.draw()
            self.psychopy_window.flip()
            self.root.after(50, lambda: self.psychopy_window.flip())  # Clear after 50ms
            
            self.log("Visual beep flash displayed")
            
        except Exception as e:
            self.log(f"Error creating visual beep: {e}")
    

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
                            
                            # Check if we should show video (only for PRE assessments)
                            is_pre_assessment = not self.is_post_assessment.get()
                            
                            if (self.use_instruction_video.get() and 
                                self.video_stimulus and 
                                is_pre_assessment):
                                self.showing_video = True
                                self.video_stimulus.seek(0)  # Reset to beginning
                                self.video_stimulus.play()
                                self.log("PRE assessment: Playing instruction video")
                            else:
                                # No video for POST assessment or no video selected
                                if not is_pre_assessment:
                                    self.log("POST assessment: Skipping instruction video")
                                
                                # Go to landing page or straight to trials
                                if self.use_instruction_video.get() and is_pre_assessment:
                                    # PRE assessment but no video available
                                    self.showing_landing = True
                                    self.landing_stim.draw()
                                    self.psychopy_window.flip()
                                    self.log("PRE assessment: No video available, showing landing page")
                                else:
                                    # POST assessment - skip directly to trials
                                    self.psychopy_window.flip()  # Clear screen
                                    self.log("Instructions complete, starting trials")
                                    self.root.after(100, self.start_trial)
                        
                        # Handle landing page responses
                        if self.showing_landing:
                            is_pre_assessment = not self.is_post_assessment.get()
                            
                            if ('y' in keys or 'Y' in keys) and is_pre_assessment:
                                # Replay video (only available for PRE assessments)
                                self.showing_landing = False
                                if self.video_stimulus:
                                    self.showing_video = True
                                    self.video_stimulus.seek(0)  # Reset to beginning
                                    self.video_stimulus.play()
                                    self.log("Replaying instruction video")
                                else:
                                    # If no video available, just show landing again
                                    self.create_landing_stimulus()
                                    self.landing_stim.draw()
                                    self.psychopy_window.flip()
                                    self.log("No video to replay")
                            
                            elif 'space' in keys or 'n' in keys or 'N' in keys:
                                # Proceed to experiment
                                self.showing_landing = False
                                self.psychopy_window.flip()  # Clear screen
                                self.log("Landing page complete, starting trials")
                                self.root.after(100, self.start_trial)
                            
                            elif ('y' in keys or 'Y' in keys) and not is_pre_assessment:
                                # Y pressed during POST assessment - ignore and log
                                self.log("POST assessment: Video replay not available")
                        
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

    def trigger_baseline_beep(self):
        """Trigger baseline beep with fallback handling"""
        if not self.use_baseline_beep.get():
            return
        
        success = False
        
        success = self.play_beep_fallback()
        
        # Add to markers with method used
        method = "psychopy" if (success and self.sound_system_available) else self.sound_fallback_method.get()
        self.log(f"Trial_{self.current_trial}_Baseline_Beep_{method}")

    # Call this in __init__ after UI setup:
    def startup_sound_check(self):
        """Check sound system on startup and update UI"""
        self.root.after(1000, self._delayed_sound_check)

    def _delayed_sound_check(self):
        """Delayed sound check to avoid startup conflicts"""
        self.initialize_sound_system()
        self.update_sound_status()
        
        if not self.sound_system_available:
            messagebox.showwarning(
                "Audio System Warning",
                f"PsychoPy audio system unavailable.\n"
                f"Fallback method '{self.sound_fallback_method.get()}' will be used for beep cues.\n\n"
                f"For full audio support, ensure PortAudio drivers are installed."
            )

    def send_marker(self, marker_label):
        """Send a marker through the serial port with nibble placement based on subject number
        
        Following BrainVision TriggerBox Plus R/S marker convention:
        - Odd subject numbers: use lower nibble (S marker style) - bits 0-3
        - Even subject numbers: use upper nibble (R marker style) - bits 4-7
        
        Args:
            marker_label: The marker value to send (can be string, hex string, or int)
        """
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                self.log(f"Error: Cannot send marker '{marker_label}' - Serial connection not open")
                return
            
            # Parse the marker value from the label
            marker_value = self._parse_marker_value(marker_label)
            if marker_value is None:
                return  # Error already logged in _parse_marker_value
            
            # Get subject number and determine nibble placement
            try:
                subject_num = int(self.subject_number.get().strip())
                
                # Validate marker value range (1-15 for nibble operations)
                if marker_value < 1 or marker_value > 15:
                    self.log(f"Warning: Marker {marker_value} is out of valid range (1-15). Clamping to valid range.")
                    marker_value = max(1, min(15, marker_value))
                
                # Determine nibble placement based on subject number parity
                is_odd_subject = (subject_num % 2 == 1)
                
                if is_odd_subject:
                    # Odd subject - use lower nibble (S marker style)
                    final_marker_byte = marker_value & 0x0F  # Keep only lower 4 bits
                    marker_type = "S-style (lower nibble)"
                else:
                    # Even subject - use upper nibble (R marker style)  
                    final_marker_byte = (marker_value & 0x0F) << 4  # Shift to upper 4 bits
                    marker_type = "R-style (upper nibble)"
                    
                self.log(f"Subject {subject_num} ({'odd' if is_odd_subject else 'even'}): "
                        f"Sending {marker_type} marker {marker_value} as byte {final_marker_byte} "
                        f"(0x{final_marker_byte:02X}, binary: {final_marker_byte:08b})")
                
            except (ValueError, AttributeError):
                self.log(f"Warning: Could not parse subject number '{self.subject_number.get()}', "
                        f"sending marker {marker_value} without nibble encoding")
                final_marker_byte = marker_value
            
            # Send the marker byte
            self._send_raw_byte(final_marker_byte)
            
            # Store the marker with timestamp for logging
            marker_timestamp = time.time()
            self.markers.append((final_marker_byte, marker_timestamp))
            
            # Log timing information
            if self.recording_start_time is not None:
                relative_time = marker_timestamp - self.recording_start_time
                self.log(f"Marker byte {final_marker_byte} sent at {relative_time:.4f}s")
            else:
                self.log(f"Marker byte {final_marker_byte} sent")
                
        except Exception as e:
            self.log(f"Error sending marker '{marker_label}': {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    def _parse_marker_value(self, marker_label):
        """Parse marker value from various input formats
        
        Args:
            marker_label: Can be int, hex string (0x...), or numeric string
            
        Returns:
            int: Parsed marker value, or None if parsing failed
        """
        try:
            if isinstance(marker_label, int):
                return marker_label
            elif isinstance(marker_label, str):
                if marker_label.startswith('0x') or marker_label.startswith('0X'):
                    return int(marker_label, 16)
                elif marker_label.isdigit():
                    return int(marker_label)
                else:
                    # Try to convert to int anyway
                    return int(str(marker_label))
            else:
                return int(str(marker_label))
        except (ValueError, TypeError):
            self.log(f"Error: Could not parse marker value '{marker_label}'. "
                    f"Expected integer, hex string (0x...), or numeric string.")
            return None

    def _send_raw_byte(self, byte_value):
        """Send raw byte to serial port with BrainVision TriggerBox timing
        
        Args:
            byte_value (int): Byte value to send (0-255)
        """
        try:
            # Ensure byte_value is within valid range
            byte_value = max(0, min(255, int(byte_value)))
            
            # Send the marker byte
            self.serial_connection.write(bytes([byte_value]))
            self.serial_connection.flush()
            
            # Small delay for trigger pulse duration (similar to TriggerBox timing)
            time.sleep(0.01)  # 10ms pulse duration
            
            # Send clear signal (0) to reset trigger
            self.serial_connection.write(bytes([0]))
            self.serial_connection.flush()
            
            # Brief delay after clearing
            time.sleep(0.05)  # 50ms clear delay
            
        except Exception as e:
            self.log(f"Error sending raw byte {byte_value}: {e}")
            raise

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

    # In create_psychopy_window, modify the landing page text creation:
    def update_landing_text(self):
        """Update landing page text based on assessment type"""
        is_pre_assessment = not self.is_post_assessment.get()
        
        if is_pre_assessment:
            self.landing_text = "Press Y to replay the instruction video\nPress SPACEBAR or N to begin the experiment"
        else:
            self.landing_text = "Press SPACEBAR or N to begin the experiment"

    def create_landing_stimulus(self):
        """Create or update the landing page stimulus"""
        self.update_landing_text()
        
        if hasattr(self, 'landing_stim'):
            self.landing_stim.setText(self.landing_text)
        else:
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

    def update_instruction_text(self):
        """Update instruction text based on assessment type"""
        is_pre_assessment = not self.is_post_assessment.get()
        
        if is_pre_assessment:
            self.instruction = "Pre-Training Motor Imagery Assessment.\n\n\n\nAfter a baseline period, there will be a beep cue and depending on the visual cue presented, you will need to perform a different task. After each task, you will evaluate your performance.\n\n+ : Fixation\n\u2022 : Motor Imagery\n Text : Read the text silently. \n\nPlease let the experimenter know when you are ready to view the motor movement to imagine."
        else:
            self.instruction = "Post-Training Motor Imagery Assessment.\n\n\n\nAfter a baseline period, you will be asked to either imagine a motor movement or maintain a resting state. After each task, you will evaluate your performance.\n\n+ : Fixation\n\u2022 : Motor Imagery\n Blank Screen : Rest\n\nPlease let the experimenter know when you are ready to begin."
        
        # Update the stimulus if it exists
        if hasattr(self, 'instruction_stim'):
            self.instruction_stim.setText(self.instruction)

    def start_session(self):

        # Update participant info if in auto mode
        if self.auto_participant_management.get():
            self.update_participant_info()
        
        # Check for existing files to prevent overwrites
        subject = self.subject_number.get().strip()
        session_type = "POST" if self.is_post_assessment.get() else "PRE"
        
        if not self.debug_mode.get():
            # Check if files already exist for this participant/session
            test_filename = f"S{subject}_{session_type}_eval_results_"
            existing_files = [f for f in os.listdir(self.save_directory) 
                            if f.startswith(test_filename)] if os.path.exists(self.save_directory) else []
            
            if existing_files and not messagebox.askyesno("File Exists", 
                f"Files already exist for Subject {subject} {session_type} assessment.\n"
                f"Continue anyway? (Files will be versioned to prevent overwrites)"):
                return
            
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

        self.update_instruction_text()

        # Start the first trial after a short delay
        self.root.after(500, self.start_trial)
    
    def stop_session(self):
        if not self.running: # Prevent double stop
            return
        self.running = False
        self.log("Stopping session...")
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
            self.ensure_save_directory()
            filename = self.generate_safe_filename("marker_log", "txt")
            filepath = os.path.join(self.save_directory, filename)
            
            subject = self.subject_number.get().strip()
            session_type = "POST" if self.is_post_assessment.get() else "PRE"
            
            with open(filepath, 'w') as f:
                f.write(f"Session Marker Log - Subject {subject} - {session_type} Assessment\n")
                f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n\n")
                f.write("Relative Time (s) | Marker\n")
                f.write("-"*50 + "\n")
                
                for marker, timestamp in self.markers:
                    relative_time = timestamp - self.recording_start_time
                    f.write(f"{relative_time:15.4f} | {marker}\n")
            
            self.log(f"Marker log exported to {filepath}")
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
            self.ensure_save_directory()
            txt_filename = self.generate_safe_filename("eval_results", "txt")
            csv_filename = self.generate_safe_filename("eval_results", "csv")
            txt_filepath = os.path.join(self.save_directory, txt_filename)
            csv_filepath = os.path.join(self.save_directory, csv_filename)
            
            subject = self.subject_number.get().strip()
            session_type = "POST" if self.is_post_assessment.get() else "PRE"
            
            # Export text file (same as before but with new filepath)
            with open(txt_filepath, 'w') as f:
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
            
            # Export CSV file
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Session', 'Trial', 'Activity', 'Response', 'ResponseTime'])
                
                for trial, activity, response, response_time in self.eval_results:
                    writer.writerow([subject, session_type, trial, activity, response, f"{response_time:.4f}"])
            
            self.log(f"Results exported to {txt_filename} and {csv_filename}")
            messagebox.showinfo("Export Complete", f"Results saved to data folder")
            
            # Update participant info for next session
            if self.auto_participant_management.get():
                self.root.after(1000, self.update_participant_info)
            
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
        """Display fixation cross and handle baseline beep"""

        self.cue_text.height = 0.5
        self.update_cue("+", "Baseline")
        
        self.send_marker(self.baseline)

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

        # Play baseline beep at fixation start
        self.trigger_baseline_beep()
        
        if activity == 'imagery':
            self.cue_text.height = 0.5
            self.update_cue("•", "Motor Imagery")
            
            marker_start = self.motor_imagery
            duration = self.imagery_duration.get()
            self.log(f"Phase: Motor Imagery ({duration} s)")
        else: # activity == 'rest'

            if self.rest_type == 'text':
                self.cue_text.height = 0.2
                self.update_cue("REST", "Rest")
                
            elif self.rest_type == 'fixation':
                self.cue_text.height = 0.5
                self.update_cue("+", "Rest")
            marker_start = self.rest
            duration = self.rest_duration.get()
            self.log(f"Phase: Rest ({duration} s)")

        self.send_marker(marker_start)
        
        # Schedule evaluation phase
        self.root.after(duration * 1000, self.start_evaluation_phase, activity)

    def select_rest_text(self, previous_word):
        if previous_word is None:
            # Select a new word randomly from the list
            word = random.choice(self.rest_texts)
        else:
            # Select a new word that is different from the previous one
            available_words = [w for w in self.rest_texts if w != previous_word]
            if not available_words:
                self.log("Warning: No available rest words left to select")
                return None
            word = random.choice(available_words)
        self.previous_word = word
        self.log(f"Selected rest word: {word}")

    def start_evaluation_phase(self, activity_type):
        """Start the evaluation phase after an activity."""
        # Clear any previous cue
        self.update_cue("", "Evaluation")
        
        # Store current activity type for the evaluation
        self.current_activity = activity_type
        
        # Store start time for response time measurement
        self.eval_start_time = time.time()
        
        # Add a slight delay before showing eval screen to ensure transition is visible
        self.root.after(100, self.show_evaluation_screen)

        # Send marker
        self.send_marker(self.eval_start)
        
        self.log("Evaluation phase started - waiting for Y/N keypress")
        
        # Note: No timeout - waiting for user keystroke

    def update_video_ui_status(self):
        """Update video UI elements to show availability based on assessment type"""
        is_pre_assessment = not self.is_post_assessment.get()
        
        if hasattr(self, 'video_frame'):  # If video frame exists in UI
            if is_pre_assessment:
                status_text = "Video will be shown during PRE assessment"
                color = "blue"
            else:
                status_text = "Video will be skipped during POST assessment"
                color = "orange"
            
            # Add or update status label in video frame
            if not hasattr(self, 'video_status_label'):
                self.video_status_label = ttk.Label(self.video_frame, text="", foreground=color)
                self.video_status_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
            
            self.video_status_label.config(text=status_text, foreground=color)

    def evaluation_response(self, response):
        """Handle yes/no response during evaluation period."""
        if not self.waiting_for_response:
            return  # Prevent double responses
        
        # Calculate response time
        response_time = time.time() - self.eval_start_time
        
        # Log and store the response
        if response:
            self.log(f"Evaluation: YES (response time: {response_time:.2f}s)")
            self.send_marker(self.eval_yes)
        else:
            self.log(f"Evaluation: NO (response time: {response_time:.2f}s)")
            self.send_marker(self.eval_no) 
        
        # Hide evaluation screen
        self.hide_evaluation_screen()
        
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