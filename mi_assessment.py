import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import random
import serial
import serial.tools.list_ports
import math

# For better monitor detection
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    import ctypes
    import platform

class EEGMarkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Marker GUI")
        # Increased initial height slightly for the new controls
        self.root.geometry("550x950")
        self.root.minsize(600, 600)

        # Adjustable durations (in seconds)
        self.blank1_duration = tk.IntVar(value=1)      # Pre-trial blank (Block 1 or 2)
        self.baseline1_duration = tk.IntVar(value=3)     # First baseline (Block 1 or 2)
        self.motor_duration = tk.IntVar(value=2)         # Motor execution (Block 1 if enabled)
        self.blank2_duration = tk.IntVar(value=1)        # Blank between blocks
        self.baseline2_duration = tk.IntVar(value=4)     # Second baseline (Block 2)
        self.imagery_duration = tk.IntVar(value=2)       # Motor imagery (Activity Block)
        self.rest_duration = tk.IntVar(value=2)          # Rest (Activity Block)

        self.total_trials = tk.IntVar(value=10)

        # Activity selection checkboxes
        self.select_motor_exec = tk.BooleanVar(value=False)
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        # --- NEW: Randomization Option ---
        self.randomize_block_order = tk.BooleanVar(value=False) # Option to randomize Execution vs Activity block order

        # --- NEW: Serial Port Variables ---
        self.com_port = tk.StringVar(value="")
        self.available_ports = []
        self.serial_connection = None
        self.baud_rate = tk.IntVar(value=9600)

        # --- NEW: Monitor Selection Variables ---
        self.available_monitors = []
        self.selected_monitor = tk.IntVar(value=0)  # Default to primary monitor

        self.running = False
        self.current_trial = 0
        self.markers = []
        self.recording_start_time = None # Wall clock time when recording started

        # Minimum baseline duration (in seconds) for randomization
        self.min_baseline_duration = 2.5

        # --- Sequences for balanced randomization ---
        self.trial_activity_sequence = [] # Holds 'imagery' or 'rest' for each trial if balanced
        self.trial_block_order_sequence = [] # Holds 'motor_first' or 'activity_first' if randomized

        self.setup_ui()
        self.get_available_com_ports()
        self.get_available_monitors()

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

        row_idx = 0

        # --- NEW: Serial Port Selection ---
        lbl_serial = tk.Label(scrollable_frame, text="Serial Port Settings")
        lbl_serial.grid(row=row_idx, column=0, sticky="w", pady=(0, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_serial)
        
        # COM Port dropdown
        port_frame = tk.Frame(scrollable_frame)
        port_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 5)); row_idx += 1
        
        lbl_port = tk.Label(port_frame, text="COM Port:")
        lbl_port.pack(side="left", padx=(0, 5))
        self.dynamic_labels.append(lbl_port)
        
        self.port_dropdown = ttk.Combobox(port_frame, textvariable=self.com_port, state="readonly", width=20)
        self.port_dropdown.pack(side="left", padx=(0, 10))
        
        refresh_button = ttk.Button(port_frame, text="Refresh", command=self.get_available_com_ports)
        refresh_button.pack(side="left")
        self.dynamic_buttons.append(refresh_button)
        
        # Baud Rate frame
        baud_frame = tk.Frame(scrollable_frame)
        baud_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1
        
        lbl_baud = tk.Label(baud_frame, text="Baud Rate:")
        lbl_baud.pack(side="left", padx=(0, 5))
        self.dynamic_labels.append(lbl_baud)
        
        baud_dropdown = ttk.Combobox(baud_frame, textvariable=self.baud_rate, state="readonly", width=10,
                                    values=[9600, 19200, 38400, 57600, 115200])
        baud_dropdown.pack(side="left")

        # --- NEW: Monitor Selection ---
        lbl_monitor = tk.Label(scrollable_frame, text="Display Settings")
        lbl_monitor.grid(row=row_idx, column=0, sticky="w", pady=(10, 2)); row_idx += 1
        self.dynamic_labels.append(lbl_monitor)
        
        monitor_frame = tk.Frame(scrollable_frame)
        monitor_frame.grid(row=row_idx, column=0, sticky="ew", pady=(0, 10)); row_idx += 1
        
        lbl_monitor_select = tk.Label(monitor_frame, text="Cue Display Monitor:")
        lbl_monitor_select.pack(side="left", padx=(0, 5))
        self.dynamic_labels.append(lbl_monitor_select)
        
        self.monitor_dropdown = ttk.Combobox(monitor_frame, textvariable=self.selected_monitor, state="readonly", width=20)
        self.monitor_dropdown.pack(side="left", padx=(0, 10))
        
        refresh_monitor_button = ttk.Button(monitor_frame, text="Refresh", command=self.get_available_monitors)
        refresh_monitor_button.pack(side="left")
        self.dynamic_buttons.append(refresh_monitor_button)

        # --- Separator ---
        separator = ttk.Separator(scrollable_frame, orient="horizontal")
        separator.grid(row=row_idx, column=0, sticky="ew", pady=10); row_idx += 1
        '''
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

        '''

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

        # --- Randomize Block Order Checkbox ---
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
        self.start_button = tk.Button(scrollable_frame, text="Start Session", command=self.start_session)
        self.start_button.grid(row=row_idx, column=0, sticky="ew", pady=(0, 5)); row_idx += 1
        self.dynamic_buttons.append(self.start_button)
        self.stop_button = tk.Button(scrollable_frame, text="Stop Session", command=self.stop_session, state="disabled")
        self.stop_button.grid(row=row_idx, column=0, sticky="ew"); row_idx += 1
        self.dynamic_buttons.append(self.stop_button)

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
        """Scan for available monitors and update dropdown using platform-specific methods"""
        self.available_monitors = []
        
        try:
            # First try using the screeninfo library if available
            if SCREENINFO_AVAILABLE:
                monitors = get_monitors()
                
                for idx, monitor in enumerate(monitors):
                    # Format information about each monitor
                    primary_text = " (Primary)" if idx == 0 else ""
                    monitor_desc = f"Monitor {idx}{primary_text} - {monitor.width}x{monitor.height} at ({monitor.x},{monitor.y})"
                    self.available_monitors.append((idx, monitor_desc))
                    
                    # Store the monitor's position information for later
                    setattr(self, f"monitor_{idx}_x", monitor.x)
                    setattr(self, f"monitor_{idx}_y", monitor.y)
                    setattr(self, f"monitor_{idx}_width", monitor.width)
                    setattr(self, f"monitor_{idx}_height", monitor.height)
                
                self.log(f"Found {len(monitors)} monitor(s) using screeninfo library")
            
            # Fallback to platform-specific methods if screeninfo is not available
            else:
                system = platform.system()
                
                if system == "Windows":
                    # Windows-specific code to get monitor information
                    self.log("Using Windows API for monitor detection")
                    user32 = ctypes.windll.user32
                    
                    # Get primary monitor dimensions
                    primary_width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
                    primary_height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
                    
                    # Check if multiple monitors are present
                    monitor_count = user32.GetSystemMetrics(80)  # SM_CMONITORS
                    
                    # Add primary monitor
                    self.available_monitors.append((0, f"Primary Monitor - {primary_width}x{primary_height}"))
                    
                    # If multiple monitors are detected, use virtual screen information
                    if monitor_count > 1:
                        # Get virtual screen coordinates (all monitors combined)
                        virtual_left = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
                        virtual_top = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
                        virtual_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
                        virtual_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
                        
                        # Add a second monitor indicator (simplified approach)
                        # This is a best-guess since Windows API doesn't easily expose
                        # individual monitor coordinates through ctypes
                        self.available_monitors.append(
                            (1, f"Secondary Monitor - Virtual screen: {virtual_width}x{virtual_height}")
                        )
                        
                        self.log(f"Found {monitor_count} monitor(s) using Windows API")
                    
                elif system == "Darwin":  # macOS
                    # Use Tkinter's screen dimensions as fallback on macOS
                    self.log("Using macOS/Tkinter fallback for monitor detection")
                    self.root.update_idletasks()
                    
                    # Get primary monitor dimensions
                    primary_width = self.root.winfo_screenwidth()
                    primary_height = self.root.winfo_screenheight()
                    
                    # Add primary monitor
                    self.available_monitors.append((0, f"Primary Monitor - {primary_width}x{primary_height}"))
                    
                    # Test for a second monitor using a heuristic approach
                    # Create a temporary window with position outside primary screen
                    try:
                        test_win = tk.Toplevel(self.root)
                        test_win.geometry(f"10x10+{primary_width+100}+100")  # Position outside primary
                        test_win.update()
                        
                        # If position is accepted, likely a secondary monitor exists
                        if test_win.winfo_x() > primary_width:
                            self.available_monitors.append(
                                (1, "Secondary Monitor - Detected")
                            )
                            self.log("Detected secondary monitor")
                        
                        test_win.destroy()
                    except:
                        pass
                
                else:  # Linux or other
                    # Use Tkinter's built-in values as fallback
                    self.log("Using Tkinter fallback for monitor detection")
                    self.root.update_idletasks()
                    
                    # Get primary monitor dimensions
                    primary_width = self.root.winfo_screenwidth()
                    primary_height = self.root.winfo_screenheight()
                    
                    # Add primary monitor
                    self.available_monitors.append((0, f"Primary Monitor - {primary_width}x{primary_height}"))
                    
                    # Unable to reliably detect secondary monitors without screeninfo
                    self.log("Secondary monitors might not be detected correctly without screeninfo library")
            
            # If no monitors were found (shouldn't happen), add a default one
            if not self.available_monitors:
                self.root.update_idletasks()
                width = self.root.winfo_screenwidth()
                height = self.root.winfo_screenheight()
                self.available_monitors.append((0, f"Default Monitor - {width}x{height}"))
                self.log("Using default monitor as fallback")
            
            # Update the dropdown with monitor options
            monitor_values = [desc for _, desc in self.available_monitors]
            self.monitor_dropdown['values'] = monitor_values
            
            # Find the current monitor index in our available monitors list
            current_val = self.selected_monitor.get()
            current_idx = 0
            for i, (idx, _) in enumerate(self.available_monitors):
                if idx == current_val:
                    current_idx = i
                    break
            
            # Set current selection
            self.monitor_dropdown.current(current_idx)
            self.selected_monitor.set(self.available_monitors[current_idx][0])
            
            self.log(f"Updated monitor selection dropdown with {len(self.available_monitors)} option(s)")
            
        except Exception as e:
            self.log(f"Error detecting monitors: {e}")
            # Fallback to primary monitor only
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

    def send_marker(self, marker_label):
        """Send a marker through the serial port"""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                self.log(f"Error: Cannot send marker '{marker_label}' - Serial connection not open")
                return
            
            # Send the marker as a string followed by newline
            marker_str = f"{marker_label}\n"
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

    def start_session(self):
        # --- Input Validation ---
        # Check serial port
        if not self.setup_serial():
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

        # --- Start Session ---
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.markers = []
        self.current_trial = 0
        self.trial_label.config(text=f"Trial: 0 / {n_trials}")

        # Get monitor information for cue display
        selected_monitor_idx = self.selected_monitor.get()
        
        # Setup Cue Window on selected monitor
        self.cue_win = tk.Toplevel(self.root)
        self.cue_win.title("Ready")
        
        # Position on the selected monitor
        if SCREENINFO_AVAILABLE:
            # If using screeninfo library, we have the exact positions
            try:
                # Get the position attributes we stored during detection
                x = getattr(self, f"monitor_{selected_monitor_idx}_x", 0)
                y = getattr(self, f"monitor_{selected_monitor_idx}_y", 0)
                width = getattr(self, f"monitor_{selected_monitor_idx}_width", self.root.winfo_screenwidth())
                height = getattr(self, f"monitor_{selected_monitor_idx}_height", self.root.winfo_screenheight())
                
                # Position the window on the selected monitor
                self.cue_win.geometry(f"{width}x{height}+{x}+{y}")
                self.log(f"Positioning cue display on monitor {selected_monitor_idx} at ({x},{y})")
            except AttributeError:
                # Fallback if attributes not found
                self.log(f"Could not find position for monitor {selected_monitor_idx}, using fullscreen on primary")
                self.cue_win.attributes("-fullscreen", True)
        elif selected_monitor_idx == 0:
            # Primary monitor - use fullscreen
            self.cue_win.attributes("-fullscreen", True)
        else:
            # For secondary monitors without screeninfo, use best-guess positioning
            monitor_position = (0, 0)  # Default
            
            if platform.system() == "Windows":
                # On Windows, make a reasonable guess
                if selected_monitor_idx == 1:  # First secondary monitor
                    primary_width = self.root.winfo_screenwidth()
                    monitor_position = (primary_width, 0)  # Assume it's to the right
            else:
                # On macOS/Linux, use similar heuristic
                if selected_monitor_idx == 1:
                    primary_width = self.root.winfo_screenwidth()
                    monitor_position = (primary_width, 0)  # Assume it's to the right
            
            # Position and maximize on the secondary monitor
            x, y = monitor_position
            self.cue_win.geometry(f"+{x}+{y}")
            self.log(f"Positioning cue display on monitor {selected_monitor_idx} at estimated position ({x},{y})")
            self.cue_win.attributes("-fullscreen", True)
        
        self.cue_win.configure(bg="black")
        
        # Ensure it grabs focus and handles Esc
        self.cue_win.focus_force()
        self.cue_win.bind("<Escape>", lambda e: self.stop_session())

        # Create the cue label
        self.cue_label = tk.Label(self.cue_win, text="", fg="white", bg="black", font=("Arial", 100, "bold"))
        self.cue_label.pack(expand=True)
        self.cue_win.update() # Make sure it's drawn

        # Bind Escape key to main window as well
        self.root.bind("<Escape>", lambda e: self.stop_session())

        # Start the session
        self.recording_start_time = time.time() # Wall clock start time
        self.log(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.send_marker("session_start") # Mark session start

        # Start the first trial after a short delay
        self.root.after(500, self.start_trial)

    def stop_session(self):
        if not self.running: # Prevent double stop
            return
        self.running = False
        self.log("Stopping session...")
        self.send_marker("session_end") # Mark session end

        # Close the serial connection
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.log("Serial connection closed")

        if hasattr(self, 'cue_win') and self.cue_win.winfo_exists():
            self.cue_win.destroy()

        # Cancel any pending `after` calls to prevent errors
        for after_id in self.root.tk.call('after', 'info'):
            self.root.after_cancel(after_id)

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        # Unbind Escape keys
        self.root.unbind("<Escape>")

        self.log(f"Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Export marker log if needed
        self.export_marker_log()

    def export_marker_log(self):
        """Export markers to a text file for reference"""
        if not self.markers:
            self.log("No markers to export")
            return
            
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"marker_log_{timestamp_str}.txt"
            
            with open(filename, 'w') as f:
                f.write(f"Session Marker Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
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
        self.send_marker(marker)
        self.log(f"Phase: {marker} ({duration} s)")
        self.root.after(duration * 1000, getattr(self, f'start_{next_phase}_phase')) # Calls start_baseline1_phase or start_baseline2_phase

    def start_baseline1_phase(self):
        """Handles the baseline period BEFORE the FIRST activity/execution block."""
        self.update_cue("+", "Baseline")
        self.send_marker('baseline_1')

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
        self.send_marker('baseline_2')

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
        self.send_marker('execution_start')
        duration = self.motor_duration.get()
        self.log(f"Phase: Motor Execution ({duration} s)")

        # Schedule end marker and next phase
        self.root.after(duration * 1000, self.end_motor_execution_phase, is_first_block)

    def end_motor_execution_phase(self, is_first_block):
        """Sends end marker and transitions from motor execution."""
        self.send_marker('execution_end')
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
            self.update_cue("•", "Motor Imagery")
            duration = self.imagery_duration.get()
            self.log(f"Phase: Motor Imagery ({duration} s)")
        else: # activity == 'rest'
            self.update_cue("", "Rest") # Blank screen for rest
            duration = self.rest_duration.get()
            self.log(f"Phase: Rest ({duration} s)")

        self.send_marker(marker_start)
        self.root.after(duration * 1000, self.end_activity_phase, is_first_block, activity)


    def end_activity_phase(self, is_first_block, activity_type):
        """Sends end marker and transitions from activity phase."""
        marker_end = f"{activity_type}_end"
        self.send_marker(marker_end)
        if is_first_block:
            # Finished first block, move to inter-block blank/baseline
            self.start_blank_phase(is_first_block=False, next_phase='baseline2')
        else:
            # Finished second block, end trial
            self.start_trial()

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
    root = tk.Tk()
    app = EEGMarkerGUI(root)
    root.mainloop()