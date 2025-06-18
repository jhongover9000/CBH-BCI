'''
BCI_ERD.py
--------
BCI Main Code Implementation with ERD Detection

Description: Main code for the BCI system using ERD detection
instead of machine learning classification. Supports emulation
and livestreaming modes with auto-scaling and real-time
threshold adjustment.

Joseph Hong
'''

# =============================================================
# =============================================================

# Includes
import numpy as np
from datetime import datetime
import os
import argparse
import mne
import threading
import tkinter as tk
from tkinter import ttk
import time
from collections import deque
import queue
import sys
try:
    import msvcrt # For Windows
except ImportError:
    import select # For Unix-based systems

# ERD Detection imports
from erd_detection_system import AdaptiveERDDetector
from erd_quick_fix import fix_erd_detector
from livestream_autoscaler import LivestreamAutoScaler


# =============================================================
# =============================================================
# Variables

# Command line arguments
is_virtual = False
is_verbose = False
is_lsl = False
is_broadcasting = False
show_gui = True
auto_scale = True

# Directories
data_dir = './data/'
results_dir = "./results/"

# Emulator Variables
vhdr_name_loc = ""
raw_eeg_loc = ""
latency = ""

# Livestreamer Variables (Editable)
streamer_ip = "169.254.1.147"
streamer_port = 51244

# ERD Detection Variables (Editable)
erd_channels = ['C3','CP3','P3','C5','C1','FC3'] 
erd_band = 'mu'  # Frequency band (mu: 8-12 Hz)
erd_threshold = 80.0  # ERD detection threshold (%)
baseline_duration = 2.0  # Seconds for baseline
adaptation_method = 'hybrid'  # Baseline adaptation method

# Data Info Variables (Autoinitialized)
sfreq = 0
sampling_interval_us = 0
num_channels = 0
ch_names = []
selected_channel_indices = []

# Receiver Variables (Editable)
seconds_to_run = 600  # Total run time
update_interval = 0.5  # How often to print status (seconds)

# Statistics
detection_count = 0
start_time = None
last_update_time = None

# GUI Components
gui_thread = None
gui_queue = None
root = None

# For real-time threshold and baseline adjustment
stop_thread = threading.Event()
manual_baseline_reset_requested = False

# =============================================================
# =============================================================
# Functions

def input_listener_thread():
    """A thread to listen for keyboard input for real-time adjustments."""
    global erd_threshold, manual_baseline_reset_requested
    
    # Instructions
    print("\nReal-time controls enabled:")
    print("  'u'/'d': Increase/Decrease ERD Threshold")
    print("  'b'    : Manually reset baseline calibration")
    print("Press Enter to start...")

    while not stop_thread.is_set():
        try:
            if 'msvcrt' in sys.modules:
                if msvcrt.kbhit():
                    char = msvcrt.getch().decode('utf-8').lower()
                else:
                    time.sleep(0.1)
                    continue
            else: # Unix
                dr, _, _ = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    char = sys.stdin.read(1)
                else:
                    continue
            
            if char == 'u':
                erd_threshold += 5.0
                print(f"\n*** Threshold INCREASED to {erd_threshold:.1f}% ***")
            elif char == 'd':
                erd_threshold = max(0.0, erd_threshold - 5.0)
                print(f"\n*** Threshold DECREASED to {erd_threshold:.1f}% ***")
            elif char == 'b':
                # Set a flag for the main loop to handle the reset
                manual_baseline_reset_requested = True
        
        except (IOError, UnicodeDecodeError):
            time.sleep(0.1)

# Initialize BCI System Type
def initialize_bci(args):
    """Initialize appropriate receiver based on arguments"""
    if args.virtual:
        from receivers import virtual_receiver
        receiver = virtual_receiver.Emulator()
        
        # Apply auto-scaling patch if enabled
        if auto_scale:
            try:
                from erd_quick_fix import patch_virtual_receiver
                receiver = patch_virtual_receiver(receiver)
                print("Applied auto-scaling to virtual receiver")
            except:
                print("Warning: Could not apply auto-scaling patch")
                
        return receiver
        
    elif args.supernumerary and args.lsl:
        from receivers import cair_receiver
        return cair_receiver.CAIRReceiver()
        
    else:
        from receivers import livestream_receiver
        receiver = livestream_receiver.LivestreamReceiver(
            address=streamer_ip,
            port=streamer_port,
            broadcast=args.broadcast
        )
        
        # Wrap with auto-scaler if enabled
        if auto_scale:
            try:
                from livestream_autoscaler import ScaledLivestreamReceiver
                # Create wrapper that preserves original methods
                class AutoScaledReceiver:
                    def __init__(self, original_receiver):
                        self.receiver = original_receiver
                        self.scaler = LivestreamAutoScaler(
                            target_std=50.0,
                            learning_duration=5.0,
                            fs=1000  # Will be updated after connection
                        )
                        self.scaler_initialized = False
                    
                    def initialize_connection(self):
                        result = self.receiver.initialize_connection()
                        # Update scaler with actual sampling rate
                        self.scaler.fs = result[0]
                        self.scaler_initialized = True
                        return result
                    
                    def get_data(self):
                        raw_data = self.receiver.get_data()
                        if raw_data is not None and self.scaler_initialized:
                            return self.scaler.process(raw_data)
                        return raw_data
                    
                    def __getattr__(self, name):
                        # Pass through other methods
                        return getattr(self.receiver, name)
                
                return AutoScaledReceiver(receiver)
                
            except ImportError:
                print("Warning: Auto-scaling not available for livestream")
                
        return receiver


# ERD-specific GUI
class ERDMonitorGUI:
    """Simple GUI for monitoring ERD detection"""
    
    def __init__(self, queue_in):
        self.queue = queue_in
        self.root = tk.Tk()
        self.root.title("ERD Detection Monitor")
        self.root.geometry("600x400")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Detection Status", padding="10")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(status_frame, text="NO DETECTION", 
                                     font=("Arial", 20, "bold"))
        self.status_label.pack()
        
        # ERD values
        values_frame = ttk.LabelFrame(main_frame, text="ERD Values", padding="10")
        values_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.values_text = tk.Text(values_frame, width=30, height=10, font=("Courier", 10))
        self.values_text.pack()
        
        # Statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.stats_text = tk.Text(stats_frame, width=30, height=10, font=("Courier", 10))
        self.stats_text.pack()
        
        # Configure grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Start update loop
        self.update_gui()
        
    def update_gui(self):
        """Update GUI with latest data"""
        try:
            # Get all available updates
            while not self.queue.empty():
                data = self.queue.get_nowait()
                
                # Update status
                if data['detected']:
                    self.status_label.config(text="ERD DETECTED!", foreground="green")
                else:
                    self.status_label.config(text="NO DETECTION", foreground="gray")
                
                # Update ERD values
                self.values_text.delete(1.0, tk.END)
                self.values_text.insert(tk.END, "Channel   ERD%\n")
                self.values_text.insert(tk.END, "-" * 20 + "\n")
                for ch, erd in data['erd_values'].items():
                    self.values_text.insert(tk.END, f"{ch:8s} {erd:6.1f}\n")
                
                # Update statistics
                self.stats_text.delete(1.0, tk.END)
                stats = data.get('stats', {})
                self.stats_text.insert(tk.END, f"Runtime: {stats.get('runtime', 0):.1f}s\n")
                self.stats_text.insert(tk.END, f"Detections: {stats.get('count', 0)}\n")
                self.stats_text.insert(tk.END, f"Rate: {stats.get('rate', 0):.1f}/min\n")
                self.stats_text.insert(tk.END, f"\nBaseline: {stats.get('baseline_set', False)}\n")
                self.stats_text.insert(tk.END, f"Method: {stats.get('method', 'N/A')}\n")
                # **NEW**: Show current threshold in GUI
                self.stats_text.insert(tk.END, f"Threshold: {stats.get('threshold', 0):.1f}%\n")
                
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(1, self.update_gui)
    
    def run(self):
        self.root.mainloop()


def run_gui_thread(gui_queue):
    """Run GUI in separate thread"""
    from erd_gui_new import OptimizedERDGUI
    gui = ERDMonitorGUI(gui_queue)
    gui.run()


# Print time and message
def logTime(message):
    if is_verbose:
        print("===================================")
        print(f"{message} {datetime.now()}")
        print("")


# Initialize ERD detector with channel selection
def setup_erd_detector(detector, channel_names, target_channels):
    """Setup ERD detector with proper channel selection"""
    selected_indices = []
    found_channels = []
    
    # Find target channels
    for target in target_channels:
        found = False
        # Exact match first
        if target in channel_names:
            idx = channel_names.index(target)
            selected_indices.append(idx)
            found_channels.append(target)
            found = True
        else:
            # Try case-insensitive match
            for i, ch in enumerate(channel_names):
                if target.lower() in ch.lower():
                    selected_indices.append(i)
                    found_channels.append(ch)
                    found = True
                    break
        
        if not found and is_verbose:
            print(f"Warning: Channel {target} not found")
    
    # If no channels found, use first few channels
    if not selected_indices:
        print("Warning: No target channels found, using first 3 channels")
        selected_indices = list(range(min(3, len(channel_names))))
        found_channels = [channel_names[i] for i in selected_indices]
    
    # Configure detector
    detector.set_channels(channel_names, selected_indices)
    
    print(f"ERD monitoring channels: {found_channels}")
    return selected_indices


# =============================================================
# =============================================================
# Execution
if __name__ == "__main__":

    # Grab arguments from command line
    parser = argparse.ArgumentParser(description="BCI System with ERD Detection")

    # Add arguments
    parser.add_argument('--virtual', action='store_true',
                        help="Enable virtual streaming using an emulator")
    parser.add_argument('--verbose', action='store_true', 
                        help="Enable verbose logging")
    parser.add_argument('--lsl', action='store_true', 
                        help="Stream using LSL")
    parser.add_argument('--broadcast', action='store_true', 
                        help="Broadcast to other application")
    parser.add_argument('--supernumerary', action='store_true', 
                        help="Send predictions to supernumerary thumb")
    parser.add_argument('--no-gui', action='store_true',
                        help="Disable GUI monitor")
    parser.add_argument('--no-autoscale', action='store_true',
                        help="Disable automatic scaling")
    
    # ERD-specific arguments
    parser.add_argument('--channels', nargs='+', default=['C3'],
                        help="Channels to monitor for ERD (default: C3)")
    parser.add_argument('--band', choices=['mu', 'beta', 'alpha'], default='mu',
                        help="Frequency band for ERD (default: mu)")
    parser.add_argument('--threshold', type=float, default=20.0,
                        help="ERD detection threshold in percent (default: 20)")
    parser.add_argument('--baseline', type=float, default=2.0,
                        help="Baseline duration in seconds (default: 2)")
    parser.add_argument('--adaptation', choices=['static', 'sliding', 'exponential', 'kalman', 'hybrid'],
                        default='hybrid', help="Baseline adaptation method (default: hybrid)")
    parser.add_argument('--duration', type=int, default=600,
                        help="Run duration in seconds (default: 600)")

    # Parse arguments
    args = parser.parse_args()
    is_virtual = args.virtual
    is_verbose = args.verbose
    is_lsl = args.lsl
    is_broadcasting = args.broadcast
    show_gui = not args.no_gui
    auto_scale = not args.no_autoscale
    
    # ERD parameters
    # erd_channels = args.channels
    erd_band = args.band
    # erd_threshold = args.threshold
    baseline_duration = args.baseline
    adaptation_method = args.adaptation
    seconds_to_run = args.duration

    # Print configuration
    print("===================================")
    print("BCI System with ERD Detection")
    print("===================================")
    print("Configuration:")
    print(f"  Mode:              {'Virtual' if is_virtual else 'Livestream'}")
    print(f"  Broadcasting:      {is_broadcasting}")
    print(f"  Auto-scaling:      {auto_scale}")
    print(f"  GUI:               {show_gui}")
    print(f"  Verbose:           {is_verbose}")
    print("\nERD Parameters:")
    print(f"  Channels:          {erd_channels}")
    print(f"  Frequency Band:    {erd_band}")
    print(f"  Initial Threshold: {erd_threshold}%")
    print(f"  Baseline Duration: {baseline_duration}s")
    print(f"  Adaptation:        {adaptation_method}")
    print(f"  Run Duration:      {seconds_to_run}s")
    print("===================================\n")

    # Initialize BCI receiver
    logTime("Initializing BCI receiver...")
    bci = initialize_bci(args)

    # Initialize connection
    logTime("Establishing connection...")
    try:
        sfreq, ch_names, num_channels, _ = bci.initialize_connection()
        print(f"Connected: {num_channels} channels at {sfreq} Hz")
    except Exception as e:
        print(f"Connection failed: {e}")
        exit(1)

    # Initialize ERD detector
    logTime("Setting up ERD detector...")
    detector = AdaptiveERDDetector(sampling_freq=sfreq, buffer_size=int(2*sfreq))
    
    # Apply fixes for better detection
    detector = fix_erd_detector(detector)
    
    # Configure detector parameters (threshold will be set in the loop)
    detector.update_parameters(
        band=erd_band,
        baseline_duration=baseline_duration,
        adaptation_method=adaptation_method
    )
    
    
    # Setup channels
    selected_channel_indices = setup_erd_detector(detector, ch_names, erd_channels)
    
    # Start GUI if enabled
    if show_gui:
        gui_queue = queue.Queue()
        gui_thread = threading.Thread(target=run_gui_thread, args=(gui_queue,))
        gui_thread.daemon = True
        gui_thread.start()
        print("GUI monitor started")
    else:
        gui_queue = None

    # Start the input listener thread
    input_thread = threading.Thread(target=input_listener_thread)
    input_thread.daemon = True
    input_thread.start()

# Wait for user to press Enter
    input()

    # **NEW**: Guided initial baseline collection
    print("\n===================================")
    print("Starting Initial Baseline Calibration")
    print("Please relax, keep your eyes open, and avoid movement.")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end='\r')
        time.sleep(1)
    print("                                      ", end='\r') # Clear line
    print("CALIBRATING... Please remain relaxed.")
    print("===================================\n")

    # Initialize timing and counters
    start_time = time.time()
    last_update_time = start_time
    detection_count = 0
    sample_count = 0
    baseline_set = False

    # Main Loop
    try:
        while (time.time() - start_time) < seconds_to_run:
            # **NEW**: Check for manual baseline reset request
            if manual_baseline_reset_requested:
                detector.reset_baseline_collection()
                baseline_set = False # Mark that we are recalibrating
                print("\nCALIBRATING... Please remain relaxed.")
                manual_baseline_reset_requested = False # Reset the flag

            detector.erd_threshold = erd_threshold
            data = bci.get_data()
            
            if data is not None:
                sample_count += data.shape[1]
                detected, erd_values = detector.detect_erd(data)
                
                # Check if baseline was just set (either initially or after manual reset)
                if not baseline_set and detector.is_baseline_set:
                    baseline_set = True
                    print("\n✓ Baseline established! Starting ERD monitoring...\n")
                
                if detected:
                    detection_count += 1
                    if is_verbose:
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ERD DETECTED!")
                    bci.use_classification(1)
                
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    runtime = current_time - start_time
                    
                    if not is_verbose:
                        # **NEW**: Enhanced console output with baseline power
                        status_info = detector.get_status_info()
                        unified_erd_value = erd_values.get('ERD_Avg', 0.0)
                        baseline_power = status_info.get('unified_baseline_power', 0.0)
                        
                        erd_str = f"Avg ERD: {unified_erd_value:5.1f}%"
                        # Format power in a readable way, e.g., f-string
                        power_str = f"Baseline Power: {baseline_power:.2f}"
                        
                        status = "DETECTED" if detected else "--------"
                        if not baseline_set:
                            status = "CALIBRATING..."
                        
                        print(f"[{runtime:6.1f}s] {erd_str:<18} | {power_str:<22} | Threshold: {erd_threshold:4.1f}% | {status} | Count: {detection_count}", end='\r')
                    
                    if gui_queue:
                        # The GUI part will also need to be adapted to show the single averaged value
                        # For simplicity, we can pass the same structure. The existing GUI might only show the first value.
                        # For a proper GUI update, the ERDMonitorGUI class would need a small change.
                        detection_rate = (detection_count / runtime) * 60 if runtime > 0 else 0
                        try:
                            gui_queue.put_nowait({
                                'detected': detected,
                                'erd_values': erd_values, # This now contains {'ERD_Avg': value}
                                'stats': {
                                    'runtime': runtime,
                                    'count': detection_count,
                                    'rate': detection_rate,
                                    'baseline_set': baseline_set,
                                    'method': detector.adaptation_method,
                                    'threshold': erd_threshold
                                }
                            })
                        except queue.Full:
                            pass
                    
                    last_update_time = current_time
                
            if is_virtual:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError in main loop: {e}")
        if is_verbose:
            import traceback
            traceback.print_exc()
    
    finally:
            stop_thread.set()
            if 'msvcrt' not in sys.modules:
                print("\nPress Enter to exit.")
            input_thread.join(timeout=1.0)

    # Disconnect
    logTime("Disconnecting...")
    bci.disconnect()
    
    # Print final summary
    total_runtime = time.time() - start_time
    print("\n\n===================================")
    print("Session Summary")
    print("===================================")
    print(f"Total Runtime:     {total_runtime:.1f} seconds")
    print(f"Samples Processed: {sample_count}")
    if total_runtime > 0:
        print(f"Sampling Rate:     {sample_count/total_runtime:.1f} Hz")
        print(f"Detection Rate:    {detection_count/total_runtime*60:.1f} per minute")
    print(f"ERD Detections:    {detection_count}")
    print(f"Baseline Method:   {detector.adaptation_method}")
    print(f"Final Threshold:   {erd_threshold:.1f}%")
    print("===================================")
    print("Stream Disconnected Successfully.")

# Execution End