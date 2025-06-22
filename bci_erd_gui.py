'''
bci_erd_gui.py
-------------
GUI Monitor for BCI ERD System

Provides real-time visualization of ERD detection
'''

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from collections import deque
import time


class ERDMonitorGUI:
    """GUI Monitor for ERD Detection System"""
    
    def __init__(self, queue_in, system_ref=None, threshold=20):
        self.queue = queue_in
        self.system = system_ref
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("BCI ERD Detection Monitor")
        self.root.geometry("1200x700")
        self.threshold = threshold
        
        # Data storage for plots
        self.time_history = deque(maxlen=200)
        self.erd_history = {}
        self.baseline_power_history = {}

        # Annotation tracking
        self.annotation_markers = []  # Store annotation markers on plot
        self.erd_detection_markers = []  # Store ERD detection markers
        
        # Initialize GUI elements
        self._create_gui()
        
        # Start update loop
        self.update_gui()
        
    def _create_gui(self):
        """Create GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls and Status
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Status frame
        status_frame = ttk.LabelFrame(left_panel, text="System Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))
        
        self.status_label = ttk.Label(status_frame, text="Initializing...", 
                                     font=("Arial", 12, "bold"))
        self.status_label.pack()
        
        self.baseline_status = ttk.Label(status_frame, text="Baseline: Not Ready", 
                                        foreground="red")
        self.baseline_status.pack(pady=5)
        
        # Detection indicator
        detection_frame = ttk.LabelFrame(left_panel, text="Detection", padding="10")
        detection_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.detection_canvas = tk.Canvas(detection_frame, width=150, height=150, bg='white')
        self.detection_canvas.pack()
        self.detection_indicator = self.detection_canvas.create_oval(10, 10, 140, 140, 
                                                                    fill='gray', outline='')
        
        self.detection_label = ttk.Label(detection_frame, text="NO DETECTION", 
                                        font=("Arial", 14, "bold"))
        self.detection_label.pack(pady=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.stats_text = tk.Text(stats_frame, width=25, height=8, font=("Courier", 10))
        self.stats_text.pack()

        # Annotation correlation frame
        ann_frame = ttk.LabelFrame(left_panel, text="Annotation Correlation", padding="10")
        ann_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.ann_text = tk.Text(ann_frame, width=25, height=6, font=("Courier", 9))
        self.ann_text.pack()
        
        # Controls frame
        control_frame = ttk.LabelFrame(left_panel, text="Controls", padding="10")
        control_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.baseline_btn = ttk.Button(control_frame, text="Recalculate Baseline",
                                      command=self.recalculate_baseline)
        self.baseline_btn.pack(pady=5)

        # Right panel - Displays
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ERD values display
        values_frame = ttk.LabelFrame(right_panel, text="Real-Time ERD Values", padding="10")
        values_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 5))
        
        self.erd_display = tk.Text(values_frame, width=50, height=8, font=("Courier", 11))
        self.erd_display.pack()
        
        # Plot frame
        plot_frame = ttk.LabelFrame(right_panel, text="ERD Trend", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('ERD (%)')
        self.ax.set_ylim(-30, 100)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pre-create plot lines
        self.plot_lines = {}
        self.colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Threshold line
        self.threshold_line = self.ax.axhline(y=self.threshold, color='k', linestyle='--', 
                                             alpha=0.5, label='Threshold')
        
        # Preprocessing info frame
        preproc_frame = ttk.LabelFrame(right_panel, text="Preprocessing Info", padding="10")
        preproc_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        self.preproc_text = tk.Text(preproc_frame, width=50, height=3, font=("Courier", 9))
        self.preproc_text.pack()
        
    def update_gui(self):
        """Update GUI with latest data"""
        try:
            # Process all available updates
            while not self.queue.empty():
                data = self.queue.get_nowait()
                self._process_update(data)
                
        except:
            pass
        
        # Schedule next update
        self.root.after(50, self.update_gui)
    
    def _process_update(self, data):
        """Process a single update"""
        # Update status
        if data.get('baseline_ready', False):
            self.baseline_status.config(text="Baseline: Ready", foreground="green")
            self.status_label.config(text="Detecting ERD...")
        else:
            progress = data.get('baseline_progress', 0)
            self.status_label.config(text=f"Collecting Baseline: {progress:.0f}%")
        
        # Update detection indicator
        if data.get('detected', False):
            self.detection_canvas.itemconfig(self.detection_indicator, fill='#00FF00')
            self.detection_label.config(text="ERD DETECTED!", foreground="green")
        else:
            self.detection_canvas.itemconfig(self.detection_indicator, fill='#FF0000')
            self.detection_label.config(text="NO DETECTION", foreground="gray")
        
        # Update ERD values display
        erd_values = data.get('erd_values', {})
        self._update_erd_display(erd_values)
        
        # Update statistics
        self._update_statistics(data)

        # Update plot with annotations
        if erd_values:
            self._update_plot_with_annotations(data)
        
        # Update plot
        if erd_values:
            self._update_plot(data)
        
        # Update preprocessing info
        self._update_preprocessing_info()
    
    def _update_erd_display(self, erd_values):
        """Update ERD values text display"""
        self.erd_display.delete(1.0, tk.END)
        
        self.erd_display.insert(tk.END, "Channel    ERD%    Power   Status\n")
        self.erd_display.insert(tk.END, "-" * 40 + "\n")
        
        for ch, erd in erd_values.items():
            status = "DETECT" if erd > self.threshold else "------"
            marker = "***" if erd > self.threshold else "   "
            
            # Get baseline power if available
            if hasattr(self.system, 'erd_detector') and self.system.erd_detector.baseline_power is not None:
                ch_idx = next((i for i, name in enumerate(self.system.ch_names) if ch in name), -1)
                if ch_idx in self.system.erd_detector.erd_channel_indices:
                    idx = self.system.erd_detector.erd_channel_indices.index(ch_idx)
                    power = self.system.erd_detector.baseline_power[idx]
                    self.erd_display.insert(tk.END, 
                        f"{marker} {ch:8s} {erd:6.1f}  {power:7.2e}  {status}\n")
                else:
                    self.erd_display.insert(tk.END, 
                        f"{marker} {ch:8s} {erd:6.1f}     ---     {status}\n")
            else:
                self.erd_display.insert(tk.END, 
                    f"{marker} {ch:8s} {erd:6.1f}     ---     {status}\n")
    
    def _update_statistics(self, data):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)
        
        runtime = data.get('runtime', 0)
        count = data.get('count', 0)
        rate = data.get('rate', 0)
        
        stats_lines = [
            f"Runtime:     {runtime:.1f} s",
            f"Detections:  {count}",
            f"Rate:        {rate:.1f} /min",
            "",
            f"Samples:     {self.system.sample_count if self.system else 'N/A'}",
            f"Buffer:      {len(self.system.main_buffer) if self.system else 'N/A'}"
        ]
        
        self.stats_text.insert(tk.END, "\n".join(stats_lines))
        
        # Update annotation correlation
        self.ann_text.delete(1.0, tk.END)
        
        annotations = data.get('annotations', [])
        erd_times = data.get('erd_detection_times', [])
        
        if annotations:
            self.ann_text.insert(tk.END, f"Annotations: {len(annotations)}\n")
            
            # Show last few annotations with correlation
            for ann in annotations[-3:]:  # Last 3 annotations
                ann_time = ann['time']
                desc_short = ann['description'].split('/')[-1]
                
                # Find closest ERD detection
                if erd_times:
                    closest_det = min(erd_times, key=lambda x: abs(x - ann_time))
                    time_diff = closest_det - ann_time
                    
                    if abs(time_diff) < 2.0:
                        self.ann_text.insert(tk.END, 
                            f"{desc_short:8s} → ERD Δ{time_diff:+.1f}s\n")
                    else:
                        self.ann_text.insert(tk.END, 
                            f"{desc_short:8s} → No ERD\n")
                else:
                    self.ann_text.insert(tk.END, 
                        f"{desc_short:8s} → No ERD\n")
                    
    def _update_plot_with_annotations(self, data):
        """Update ERD trend plot with annotation markers"""
        # Store time
        self.time_history.append(data.get('runtime', 0))
        
        # Store ERD values
        for ch, erd in data['erd_values'].items():
            if ch not in self.erd_history:
                self.erd_history[ch] = deque(maxlen=200)
            self.erd_history[ch].append(erd)
        
        # Update plot
        if len(self.time_history) > 1:
            times = list(self.time_history)
            
            # Update or create lines for each channel
            for i, (ch, values) in enumerate(self.erd_history.items()):
                if ch not in self.plot_lines:
                    color = self.colors[i % len(self.colors)]
                    line, = self.ax.plot([], [], color=color, linewidth=2, label=ch)
                    self.plot_lines[ch] = line
                
                # Update line data
                self.plot_lines[ch].set_data(times[-len(values):], list(values))
            
            # Clear old annotation markers
            for marker in self.annotation_markers:
                marker.remove()
            self.annotation_markers.clear()
            
            # Clear old ERD detection markers
            for marker in self.erd_detection_markers:
                marker.remove()
            self.erd_detection_markers.clear()
            
            # Add annotation markers
            annotations = data.get('annotations', [])
            for ann in annotations:
                if times[0] <= ann['time'] <= times[-1]:  # Only show visible annotations
                    # Draw vertical line
                    marker = self.ax.axvline(x=ann['time'], color='purple', 
                                           linestyle='--', alpha=0.7, linewidth=2)
                    self.annotation_markers.append(marker)
                    
                    # Add text label
                    desc_short = ann['description'].split('/')[-1]  # Shorten description
                    text = self.ax.text(ann['time'], self.ax.get_ylim()[1]*0.9, 
                                      desc_short, rotation=90, verticalalignment='bottom',
                                      fontsize=8, color='purple')
                    self.annotation_markers.append(text)
            
            # Add ERD detection markers
            erd_times = data.get('erd_detection_times', [])
            for det_time in erd_times:
                if times[0] <= det_time <= times[-1]:  # Only show visible detections
                    marker = self.ax.axvline(x=det_time, color='green', 
                                           linestyle=':', alpha=0.5, linewidth=1)
                    self.erd_detection_markers.append(marker)
            
            # Update axes
            self.ax.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            # Update legend
            if len(self.plot_lines) > 0:
                self.ax.legend(loc='upper right')
            
            # Redraw
            self.canvas.draw_idle()
    
    def _update_plot(self, data):
        """Update ERD trend plot"""
        # Store time
        self.time_history.append(data.get('runtime', 0))
        
        # Store ERD values
        for ch, erd in data['erd_values'].items():
            if ch not in self.erd_history:
                self.erd_history[ch] = deque(maxlen=200)
            self.erd_history[ch].append(erd)
        
        # Update plot
        if len(self.time_history) > 1:
            times = list(self.time_history)
            
            # Update or create lines for each channel
            for i, (ch, values) in enumerate(self.erd_history.items()):
                if ch not in self.plot_lines:
                    color = self.colors[i % len(self.colors)]
                    line, = self.ax.plot([], [], color=color, linewidth=2, label=ch)
                    self.plot_lines[ch] = line
                
                # Update line data
                self.plot_lines[ch].set_data(times[-len(values):], list(values))
            
            # Update axes
            self.ax.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            # Update legend
            if len(self.plot_lines) > 0:
                self.ax.legend(loc='upper right')
            
            # Redraw
            self.canvas.draw_idle()
    
    def _update_preprocessing_info(self):
        """Update preprocessing information"""
        if self.system:
            from bci_erd_main import BCIConfig
            
            self.preproc_text.delete(1.0, tk.END)
            info = [
                f"CAR: {'Enabled' if BCIConfig.USE_CAR else 'Disabled'}",
                f"Artifact Rejection: {BCIConfig.ARTIFACT_METHOD}",
                f"Mu Band: {BCIConfig.ERD_BAND[0]}-{BCIConfig.ERD_BAND[1]} Hz"
            ]
            self.preproc_text.insert(tk.END, " | ".join(info))
    
    def recalculate_baseline(self):
        """Manually trigger baseline recalculation"""
        if self.system:
            success = self.system.manual_baseline_calculation()
            if success:
                self.baseline_status.config(text="Baseline: Recalculated", foreground="green")
            else:
                self.baseline_status.config(text="Baseline: Failed", foreground="red")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


# Standalone test
if __name__ == "__main__":
    import queue
    
    # Create test queue
    test_queue = queue.Queue()
    
    # Add test data
    test_queue.put({
        'detected': False,
        'erd_values': {'C3': 15.2, 'C4': 18.5, 'Cz': 12.3},
        'runtime': 10.5,
        'count': 5,
        'rate': 28.6,
        'baseline_ready': True
    })
    
    # Create and run GUI
    gui = ERDMonitorGUI(test_queue)
    gui.run()