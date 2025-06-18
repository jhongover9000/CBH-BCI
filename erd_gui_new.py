'''
Performance-optimized GUI for ERD Detection System
Fixes threading issues and improves responsiveness
'''

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('TkAgg')  # Ensure proper backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class OptimizedERDGUI:
    """
    Performance-optimized GUI with proper threading
    """
    
    def __init__(self, mode='virtual', **kwargs):
        self.mode = mode
        self.receiver_kwargs = kwargs
        self.receiver = None
        self.detector = None
        
        # Performance settings
        self.gui_update_rate = 10  # Hz (reduced from 20)
        self.plot_update_rate = 2  # Hz (much reduced)
        self.max_plot_points = 100  # Limit plot data
        
        # Threading with proper synchronization
        self.data_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.running = False
        self.data_thread = None
        self.gui_update_counter = 0
        
        # Data storage (limited size for performance)
        self.erd_history = {ch: deque(maxlen=self.max_plot_points) 
                           for ch in ['C3', 'C4', 'Cz']}
        self.time_history = deque(maxlen=self.max_plot_points)
        
        # Performance monitoring
        self.last_update_time = time.time()
        self.dropped_frames = 0
        
        # Create GUI
        self._create_gui()
        
    def _create_gui(self):
        """Create simplified, performance-optimized GUI"""
        self.root = tk.Tk()
        self.root.title(f"ERD Detection - {self.mode.upper()} Mode (Optimized)")
        self.root.geometry("1000x700")
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        self._create_control_panel(main_frame)
        
        # Display panel
        self._create_display_panel(main_frame)
        
        # Status bar
        self._create_status_bar(main_frame)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def _create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=5)
        
        # Connection controls
        self.connect_btn = ttk.Button(control_frame, text="Connect", 
                                     command=self.connect_threadsafe)
        self.connect_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.disconnect_btn = ttk.Button(control_frame, text="Disconnect", 
                                        command=self.disconnect_threadsafe,
                                        state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Basic parameters
        ttk.Label(control_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.DoubleVar(value=20.0)
        threshold_scale = ttk.Scale(control_frame, from_=5, to=50, 
                                   variable=self.threshold_var,
                                   orient=tk.HORIZONTAL, length=150)
        threshold_scale.grid(row=1, column=1, padx=5)
        self.threshold_label = ttk.Label(control_frame, text="20%")
        self.threshold_label.grid(row=1, column=2)
        threshold_scale.configure(command=lambda x: self.update_threshold())
        
        # Performance settings
        perf_frame = ttk.LabelFrame(control_frame, text="Performance", padding="5")
        perf_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.fast_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Fast Mode", 
                       variable=self.fast_mode_var).grid(row=0, column=0)
        
        self.show_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Show Plot", 
                       variable=self.show_plot_var,
                       command=self.toggle_plot).grid(row=0, column=1)
        
    def _create_display_panel(self, parent):
        """Create display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # ERD values display
        values_frame = ttk.LabelFrame(display_frame, text="ERD Values", padding="10")
        values_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.erd_display = tk.Text(values_frame, width=40, height=8, 
                                  font=("Courier", 10))
        self.erd_display.pack()
        
        # Detection indicator
        indicator_frame = ttk.LabelFrame(display_frame, text="Detection", padding="10")
        indicator_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.indicator_label = ttk.Label(indicator_frame, text="NO DETECTION",
                                        font=("Arial", 16, "bold"),
                                        foreground="gray")
        self.indicator_label.pack()
        
        self.detection_count_label = ttk.Label(indicator_frame, text="Count: 0")
        self.detection_count_label.pack()
        
        # Plot (optional)
        self.plot_frame = ttk.LabelFrame(display_frame, text="ERD Trend", padding="5")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create lightweight plot
        self.fig = Figure(figsize=(6, 3), dpi=70)  # Reduced DPI
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel('ERD (%)')
        self.ax.set_ylim(-20, 60)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pre-create plot lines for efficiency
        self.plot_lines = {}
        colors = {'C3': 'blue', 'C4': 'red', 'Cz': 'green'}
        for ch, color in colors.items():
            line, = self.ax.plot([], [], color=color, label=ch)
            self.plot_lines[ch] = line
        self.ax.legend(loc='upper right')
        
        # Threshold line
        self.threshold_line = self.ax.axhline(y=20, color='k', linestyle='--', alpha=0.5)
        
    def _create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.performance_label = ttk.Label(status_frame, text="")
        self.performance_label.pack(side=tk.RIGHT, padx=5)
        
    def connect_threadsafe(self):
        """Thread-safe connection"""
        # Disable button to prevent double-click
        self.connect_btn.config(state=tk.DISABLED)
        
        # Run connection in separate thread
        thread = threading.Thread(target=self._connect_worker)
        thread.daemon = True
        thread.start()
        
    def _connect_worker(self):
        """Worker thread for connection"""
        try:
            # Import required modules
            if self.mode == 'virtual':
                try:
                    from receivers.virtual_receiver import Emulator
                    self.receiver = Emulator(auto_scale=True, verbose=False)
                except ImportError:
                    from receivers.virtual_receiver import Emulator
                    self.receiver = Emulator()
                    # Apply scaling patch
                    from erd_quick_fix import patch_virtual_receiver
                    self.receiver = patch_virtual_receiver(self.receiver)
            else:
                from livestream_autoscaler import ScaledLivestreamReceiver
                self.receiver = ScaledLivestreamReceiver(
                    auto_scale=True,
                    **self.receiver_kwargs
                )
            
            # Initialize connection
            fs, ch_names, n_channels, _ = self.receiver.initialize_connection()
            
            # Create detector
            from erd_detection_system import AdaptiveERDDetector
            self.detector = AdaptiveERDDetector(sampling_freq=fs)
            
            # Apply fixes
            from erd_quick_fix import fix_erd_detector
            self.detector = fix_erd_detector(self.detector)
            
            # Find motor channels
            motor_channels = ['C3']
            selected_indices = []
            for ch in motor_channels:
                if ch in ch_names:
                    selected_indices.append(ch_names.index(ch))
            
            if not selected_indices and n_channels >= 3:
                selected_indices = [0, 1, 2]
            
            self.detector.set_channels(ch_names, selected_indices)
            
            # Update GUI in main thread
            self.root.after(0, self._connection_success, fs, n_channels)
            
        except Exception as e:
            self.root.after(0, self._connection_failed, str(e))
            
    def _connection_success(self, fs, n_channels):
        """Handle successful connection (main thread)"""
        self.status_label.config(text=f"Connected: {n_channels}ch @ {fs}Hz")
        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.NORMAL)
        
        # Reset data
        self.detection_count = 0
        for ch in self.erd_history:
            self.erd_history[ch].clear()
        self.time_history.clear()
        
        # Start data acquisition
        self.running = True
        self.data_thread = threading.Thread(target=self._data_acquisition_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start GUI updates
        self._schedule_gui_update()
        
    def _connection_failed(self, error_msg):
        """Handle connection failure (main thread)"""
        self.status_label.config(text=f"Connection failed: {error_msg}")
        self.connect_btn.config(state=tk.NORMAL)
        
    def _data_acquisition_loop(self):
        """Optimized data acquisition loop"""
        sample_count = 0
        last_queue_warning = 0
        
        while self.running:
            try:
                # Get data
                data = self.receiver.get_data()
                
                if data is None or data.shape[1] == 0:
                    if self.mode == 'virtual':
                        # Loop back to beginning
                        self.receiver.current_index = 0
                    continue
                
                sample_count += data.shape[1]
                
                # Process ERD
                detected, erd_values = self.detector.detect_erd(data)
                
                # Try to put in queue (non-blocking)
                try:
                    self.data_queue.put_nowait({
                        'detected': detected,
                        'erd_values': erd_values,
                        'timestamp': time.time(),
                        'samples': sample_count
                    })
                except queue.Full:
                    # Queue is full, skip this update
                    self.dropped_frames += 1
                    if time.time() - last_queue_warning > 5:
                        print(f"Warning: Dropping frames (total: {self.dropped_frames})")
                        last_queue_warning = time.time()
                
                # Handle detection
                if detected:
                    self.receiver.use_classification(1)
                
                # Sleep based on mode and performance
                if self.mode == 'virtual':
                    if self.fast_mode_var.get():
                        time.sleep(0.01)  # Fast mode
                    else:
                        time.sleep(0.02)  # Normal mode
                        
            except Exception as e:
                if self.running:
                    print(f"Data acquisition error: {e}")
                    
    def _schedule_gui_update(self):
        """Schedule next GUI update"""
        if self.running:
            self._update_gui()
            # Schedule next update
            update_interval = int(1000 / self.gui_update_rate)  # milliseconds
            self.root.after(update_interval, self._schedule_gui_update)
            
    def _update_gui(self):
        """Optimized GUI update"""
        self.gui_update_counter += 1
        
        # Process all available data
        updates_processed = 0
        latest_data = None
        
        while not self.data_queue.empty() and updates_processed < 10:
            try:
                latest_data = self.data_queue.get_nowait()
                updates_processed += 1
            except queue.Empty:
                break
        
        if latest_data is None:
            return
        
        # Update ERD display
        self._update_erd_display(latest_data['erd_values'])
        
        # Update detection indicator
        if latest_data['detected']:
            self.detection_count += 1
            self.indicator_label.config(text="ERD DETECTED!", foreground="green")
        else:
            self.indicator_label.config(text="NO DETECTION", foreground="gray")
        
        self.detection_count_label.config(text=f"Count: {self.detection_count}")
        
        # Update plot (less frequently)
        if self.show_plot_var.get() and self.gui_update_counter % 5 == 0:
            self._update_plot(latest_data)
        
        # Update performance stats
        if self.gui_update_counter % 10 == 0:
            fps = self.gui_update_rate
            queue_size = self.data_queue.qsize()
            self.performance_label.config(
                text=f"Queue: {queue_size} | Dropped: {self.dropped_frames}"
            )
            
    def _update_erd_display(self, erd_values):
        """Update ERD text display"""
        self.erd_display.delete(1.0, tk.END)
        
        # Header
        self.erd_display.insert(tk.END, "Channel   ERD%    Status\n")
        self.erd_display.insert(tk.END, "-" * 30 + "\n")
        
        # Values
        threshold = self.threshold_var.get()
        for ch, erd in erd_values.items():
            if not np.isnan(erd):
                status = "DETECT" if erd > threshold else "-----"
                marker = "*" if erd > threshold else " "
                self.erd_display.insert(tk.END, 
                    f"{marker} {ch:6s} {erd:6.1f}%   {status}\n")
                    
    def _update_plot(self, data):
        """Update plot efficiently"""
        # Store data
        self.time_history.append(data['timestamp'])
        
        for ch in ['C3', 'C4', 'Cz']:
            if ch in data['erd_values']:
                self.erd_history[ch].append(data['erd_values'][ch])
            elif self.erd_history[ch]:
                self.erd_history[ch].append(self.erd_history[ch][-1])
        
        # Update plot data
        if len(self.time_history) > 1:
            times = np.array(self.time_history) - self.time_history[0]
            
            for ch in ['C3', 'C4', 'Cz']:
                if self.erd_history[ch]:
                    self.plot_lines[ch].set_data(times, list(self.erd_history[ch]))
            
            # Update axes limits
            self.ax.set_xlim(0, max(times[-1], 1))
            
            # Update threshold line
            self.threshold_line.set_ydata([self.threshold_var.get()] * 2)
            
            # Redraw
            self.canvas.draw_idle()  # More efficient than draw()
            
    def update_threshold(self):
        """Update threshold value"""
        value = self.threshold_var.get()
        self.threshold_label.config(text=f"{value:.0f}%")
        if self.detector:
            self.detector.erd_threshold = value
            
    def toggle_plot(self):
        """Toggle plot visibility"""
        if self.show_plot_var.get():
            self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        else:
            self.plot_frame.pack_forget()
            
    def disconnect_threadsafe(self):
        """Thread-safe disconnection"""
        self.running = False
        
        # Wait for threads to finish
        if self.data_thread:
            self.data_thread.join(timeout=1)
            
        # Disconnect receiver
        if self.receiver:
            self.receiver.disconnect()
            
        # Update GUI
        self.status_label.config(text="Disconnected")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.disconnect_threadsafe()
        self.root.destroy()


# Test function
def test_optimized_gui():
    """Test the optimized GUI"""
    print("Starting Optimized ERD GUI")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['real', 'virtual'], default='virtual')
    parser.add_argument('--address', default='169.254.1.147')
    parser.add_argument('--port', type=int, default=51244)
    args = parser.parse_args()
    
    # Create and run GUI
    gui = OptimizedERDGUI(
        mode=args.mode,
        address=args.address,
        port=args.port
    )
    
    print(f"Mode: {args.mode}")
    print("Performance optimizations enabled:")
    print("- Reduced GUI update rate (10 Hz)")
    print("- Reduced plot update rate (2 Hz)")
    print("- Limited plot history (100 points)")
    print("- Non-blocking queue operations")
    print("- Efficient plot updates")
    print("\nStarting GUI...")
    
    gui.run()


if __name__ == "__main__":
    test_optimized_gui()