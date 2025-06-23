'''
bci_gui_complete.py
------------------
Complete GUI for BCI ERD System with full and minimal modes
'''

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import numpy as np
from collections import deque
from datetime import datetime

# Only import matplotlib for full GUI mode
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plotting disabled")


class BCIGUI:
    """Complete GUI for BCI System with minimal mode option"""
    
    def __init__(self, bci_system, minimal=False):
        self.bci_system = bci_system
        self.minimal = minimal
        
        # Create root window
        self.root = tk.Tk()
        self.root.title("BCI ERD Detection System")
        
        # Set window size based on mode
        if minimal:
            self.root.geometry("600x400")
        else:
            self.root.geometry("1200x800")
        
        # State variables
        self.is_running = False
        self.baseline_ready = False
        
        # Data storage for plotting (full mode only)
        if not minimal and MATPLOTLIB_AVAILABLE:
            self.time_history = deque(maxlen=500)
            self.erd_history = {}
            self.annotation_markers = []
            self.erd_detection_markers = []
        
        # Create GUI elements
        self._create_widgets()
        
        # Start GUI update loop
        self._update_gui()
        
        # Configure close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_widgets(self):
        """Create GUI widgets based on mode"""
        if self.minimal:
            self._create_minimal_widgets()
        else:
            self._create_full_widgets()
    
    def _create_minimal_widgets(self):
        """Create minimal GUI without plotting"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="BCI ERD Detection - Minimal Mode", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=(0, 10))
        
        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.start_btn = ttk.Button(button_frame, text="Start Detection", 
                                   command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Detection", 
                                  command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Manual Baseline", 
                  command=self.manual_baseline).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Reset Baseline", 
                  command=self.reset_baseline).pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Metrics grid
        metrics_frame = ttk.Frame(status_frame)
        metrics_frame.pack(fill=tk.X)
        
        self.metrics_labels = {}
        metrics = [
            ("Baseline", "Not Ready"),
            ("Runtime", "--"),
            ("ERD Avg", "--"),
            ("Detections", "0"),
            ("Rate", "--"),
            ("Updates/s", "--")
        ]
        
        for i, (label, default) in enumerate(metrics):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(metrics_frame, text=f"{label}:").grid(row=row, column=col, 
                                                           sticky=tk.W, padx=5, pady=2)
            self.metrics_labels[label] = ttk.Label(metrics_frame, text=default)
            self.metrics_labels[label].grid(row=row, column=col+1, 
                                           sticky=tk.W, padx=5, pady=2)
        
        # Simple text display for ERD values
        self.erd_frame = ttk.LabelFrame(status_frame, text="ERD Values", padding="5")
        self.erd_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.erd_text = tk.Text(self.erd_frame, height=6, width=50, font=("Courier", 10))
        self.erd_text.pack(fill=tk.BOTH, expand=True)
        
        # Threshold adjustment
        threshold_frame = ttk.LabelFrame(main_frame, text="Threshold Adjustment", padding="10")
        threshold_frame.pack(fill=tk.X, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=self.bci_system.erd_detector.baseline_power[0] 
                                          if self.bci_system.erd_detector.baseline_power 
                                          else 60.0)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=10, to=100, 
                                         variable=self.threshold_var, 
                                         orient=tk.HORIZONTAL,
                                         command=self._update_threshold)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.threshold_label = ttk.Label(threshold_frame, text="60%")
        self.threshold_label.pack(side=tk.LEFT)
        
        # Moving average controls
        ma_frame = ttk.LabelFrame(main_frame, text="Moving Average Settings", padding="10")
        ma_frame.pack(fill=tk.X, pady=5)
        
        from bci_erd_main import BCIConfig
        self.ma_enabled_var = tk.BooleanVar(value=BCIConfig.USE_MOVING_AVERAGE)
        ttk.Checkbutton(ma_frame, text="Enable Moving Average", 
                       variable=self.ma_enabled_var,
                       command=self._toggle_ma).pack(anchor=tk.W)
        
        self.ma_status_label = ttk.Label(ma_frame, 
                                        text=f"ERD MA: {BCIConfig.ERD_MA_WINDOWS} windows")
        self.ma_status_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_full_widgets(self):
        """Create full GUI with plotting"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Control Panel
        control_tab = ttk.Frame(notebook)
        notebook.add(control_tab, text="Control Panel")
        self._create_control_tab(control_tab)
        
        # Tab 2: Visualization (if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            viz_tab = ttk.Frame(notebook)
            notebook.add(viz_tab, text="Real-time Visualization")
            self._create_visualization_tab(viz_tab)
        
        # Tab 3: CSP+SVM Training
        if hasattr(self.bci_system.erd_detector, 'csp_svm_detector'):
            train_tab = ttk.Frame(notebook)
            notebook.add(train_tab, text="CSP+SVM Training")
            self._create_training_tab(train_tab)
    
    def _create_control_tab(self, parent):
        """Create control panel tab"""
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="System Control", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.start_btn = ttk.Button(button_frame, text="Start Detection", 
                                   command=self.start_detection)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Detection", 
                                  command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Manual Baseline", 
                  command=self.manual_baseline).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Reset Baseline", 
                  command=self.reset_baseline).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(button_frame, text="System Status", 
                  command=self.show_system_status).grid(row=0, column=4, padx=5, pady=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Threshold
        ttk.Label(settings_frame, text="ERD Threshold:").grid(row=0, column=0, sticky=tk.W)
        
        from bci_erd_main import BCIConfig
        self.threshold_var = tk.DoubleVar(value=BCIConfig.ERD_THRESHOLD)
        self.threshold_slider = ttk.Scale(settings_frame, from_=10, to=100, 
                                         variable=self.threshold_var, 
                                         orient=tk.HORIZONTAL,
                                         command=self._update_threshold)
        self.threshold_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=10)
        
        self.threshold_label = ttk.Label(settings_frame, text=f"{BCIConfig.ERD_THRESHOLD:.0f}%")
        self.threshold_label.grid(row=0, column=2)
        
        # Moving Average
        ttk.Label(settings_frame, text="Moving Average:").grid(row=1, column=0, sticky=tk.W, pady=(10,0))
        
        self.ma_enabled_var = tk.BooleanVar(value=BCIConfig.USE_MOVING_AVERAGE)
        ttk.Checkbutton(settings_frame, text="Enable", 
                       variable=self.ma_enabled_var,
                       command=self._toggle_ma).grid(row=1, column=1, sticky=tk.W, pady=(10,0))
        
        # MA Windows
        ma_frame = ttk.Frame(settings_frame)
        ma_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(ma_frame, text="ERD MA Windows:").pack(side=tk.LEFT, padx=(20,5))
        self.erd_ma_var = tk.IntVar(value=BCIConfig.ERD_MA_WINDOWS)
        self.erd_ma_spin = ttk.Spinbox(ma_frame, from_=1, to=10, textvariable=self.erd_ma_var,
                                       width=5, command=self._update_ma_windows)
        self.erd_ma_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ma_frame, text="Baseline MA Windows:").pack(side=tk.LEFT, padx=(20,5))
        self.baseline_ma_var = tk.IntVar(value=BCIConfig.BASELINE_MA_WINDOWS)
        self.baseline_ma_spin = ttk.Spinbox(ma_frame, from_=1, to=10, textvariable=self.baseline_ma_var,
                                           width=5, command=self._update_ma_windows)
        self.baseline_ma_spin.pack(side=tk.LEFT, padx=5)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, width=60)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", 
                                 command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Metrics
        metrics_frame = ttk.LabelFrame(main_frame, text="Real-time Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metrics_labels = {}
        metrics = ["Runtime", "Detections", "Rate", "ERD Avg", "Updates/s", "Baseline"]
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(metrics_frame, text=f"{metric}:").grid(row=row, column=col, 
                                                            sticky=tk.W, padx=5, pady=2)
            self.metrics_labels[metric] = ttk.Label(metrics_frame, text="--")
            self.metrics_labels[metric].grid(row=row, column=col+1, 
                                            sticky=tk.W, padx=(0, 20), pady=2)
    
    def _create_visualization_tab(self, parent):
        """Create visualization tab with plot"""
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Detection indicator
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left: Detection status
        detect_frame = ttk.Frame(top_frame)
        detect_frame.pack(side=tk.LEFT, padx=10)
        
        self.detection_canvas = tk.Canvas(detect_frame, width=80, height=80, bg='white')
        self.detection_canvas.pack()
        self.detection_indicator = self.detection_canvas.create_oval(10, 10, 70, 70, 
                                                                    fill='gray', outline='')
        
        self.detection_label = ttk.Label(detect_frame, text="NO DETECTION", 
                                        font=("Arial", 11, "bold"))
        self.detection_label.pack()
        
        # Right: ERD values
        values_frame = ttk.LabelFrame(top_frame, text="Current ERD Values", padding="5")
        values_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.erd_display = tk.Text(values_frame, width=50, height=4, font=("Courier", 10))
        self.erd_display.pack()
        
        # Plot
        plot_frame = ttk.LabelFrame(main_frame, text="ERD Trend", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 5), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('ERD (%)')
        self.ax.set_ylim(-20, 120)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pre-create plot elements
        self.plot_lines = {}
        self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Threshold line
        from bci_erd_main import BCIConfig
        self.threshold_line = self.ax.axhline(y=BCIConfig.ERD_THRESHOLD, color='k', 
                                             linestyle='--', alpha=0.5, label='Threshold')
    
    def _create_training_tab(self, parent):
        """Create CSP+SVM training tab"""
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Training controls
        control_frame = ttk.LabelFrame(main_frame, text="Training Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.train_status_label = ttk.Label(control_frame, text="Ready to collect data")
        self.train_status_label.pack(pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5)
        
        self.rest_btn = ttk.Button(button_frame, text="Collect REST", 
                                  command=lambda: self.start_collecting('rest'))
        self.rest_btn.pack(side=tk.LEFT, padx=5)
        
        self.mi_btn = ttk.Button(button_frame, text="Collect MI", 
                                command=lambda: self.start_collecting('mi'))
        self.mi_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_collect_btn = ttk.Button(button_frame, text="Stop Collecting", 
                                          command=self.stop_collecting, state=tk.DISABLED)
        self.stop_collect_btn.pack(side=tk.LEFT, padx=5)
        
        # Data count
        self.data_count_label = ttk.Label(control_frame, text="Rest: 0 | MI: 0")
        self.data_count_label.pack(pady=5)
        
        # Model management
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(pady=5)
        
        ttk.Button(model_frame, text="Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Save Model", 
                  command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)
    
    def _update_threshold(self, value):
        """Update threshold value"""
        from bci_erd_main import BCIConfig
        BCIConfig.ERD_THRESHOLD = float(value)
        self.threshold_label.config(text=f"{float(value):.0f}%")
        
        # Update plot threshold line if available
        if hasattr(self, 'threshold_line'):
            self.threshold_line.set_ydata([float(value), float(value)])
            self.canvas.draw_idle()
    
    def _toggle_ma(self):
        """Toggle moving average"""
        from bci_erd_main import BCIConfig
        BCIConfig.USE_MOVING_AVERAGE = self.ma_enabled_var.get()
        if hasattr(self, 'ma_status_label'):
            status = f"ERD MA: {BCIConfig.ERD_MA_WINDOWS} windows" if BCIConfig.USE_MOVING_AVERAGE else "Disabled"
            self.ma_status_label.config(text=status)
    
    def _update_ma_windows(self):
        """Update moving average window sizes"""
        from bci_erd_main import BCIConfig
        if hasattr(self, 'erd_ma_var'):
            BCIConfig.ERD_MA_WINDOWS = self.erd_ma_var.get()
        if hasattr(self, 'baseline_ma_var'):
            BCIConfig.BASELINE_MA_WINDOWS = self.baseline_ma_var.get()
    
    def start_detection(self):
        """Start ERD detection"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Clear plot data
            if hasattr(self, 'time_history'):
                self.time_history.clear()
                self.erd_history.clear()
            
            # Start BCI system in separate thread
            self.bci_thread = threading.Thread(target=self.bci_system.run)
            self.bci_thread.daemon = True
            self.bci_thread.start()
            
            self.log_status("Detection started")
    
    def stop_detection(self):
        """Stop ERD detection"""
        self.is_running = False
        self.bci_system.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_status("Detection stopped")
    
    def manual_baseline(self):
        """Trigger manual baseline calculation"""
        self.bci_system.manual_baseline_calculation()
    
    def reset_baseline(self):
        """Reset baseline"""
        if self.bci_system.erd_detector:
            self.bci_system.erd_detector.reset_baseline()
            self.baseline_ready = False
            self.log_status("Baseline reset")
            if hasattr(self, 'metrics_labels'):
                self.metrics_labels['Baseline'].config(text="Not Ready", foreground="red")
    
    def show_system_status(self):
        """Show system status in dialog"""
        self.bci_system.print_system_status()
    
    def log_status(self, message):
        """Log status message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}\n"
        
        if hasattr(self, 'status_text'):
            self.status_text.insert(tk.END, msg)
            self.status_text.see(tk.END)
        else:
            print(msg.strip())
    
    def _update_gui(self):
        """Update GUI from queue"""
        try:
            while True:
                msg = self.bci_system.gui_queue.get_nowait()
                
                if 'error' in msg:
                    self.log_status(f"Error: {msg['error']}")
                
                elif 'baseline_ready' in msg:
                    self.baseline_ready = True
                    self.log_status("Baseline established!")
                    if 'Baseline' in self.metrics_labels:
                        self.metrics_labels['Baseline'].config(text="Ready", foreground="green")
                
                elif 'detected' in msg:
                    self._update_metrics(msg)
                    self._update_erd_display(msg)
                    
                    if not self.minimal and MATPLOTLIB_AVAILABLE:
                        self._update_plot(msg)
                    
                    if msg['detected']:
                        self.log_status(f"ERD DETECTED! Confidence: {msg.get('confidence', 0):.1f}%")
                
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(50, self._update_gui)
    
    def _update_metrics(self, data):
        """Update metric displays"""
        runtime = data.get('runtime', 0)
        count = data.get('count', 0)
        rate = (count / runtime * 60) if runtime > 0 else 0
        update_rate = data.get('update_rate', 0)
        avg_erd = data.get('erd_values', {}).get('avg', 0)
        
        if 'Runtime' in self.metrics_labels:
            self.metrics_labels['Runtime'].config(text=f"{runtime:.1f}s")
        if 'Detections' in self.metrics_labels:
            self.metrics_labels['Detections'].config(text=str(count))
        if 'Rate' in self.metrics_labels:
            self.metrics_labels['Rate'].config(text=f"{rate:.1f}/min")
        if 'ERD Avg' in self.metrics_labels:
            self.metrics_labels['ERD Avg'].config(text=f"{avg_erd:.1f}%")
        if 'Updates/s' in self.metrics_labels:
            self.metrics_labels['Updates/s'].config(text=f"{update_rate:.1f}")
        
        # Update detection indicator
        if hasattr(self, 'detection_indicator'):
            if data['detected']:
                self.detection_canvas.itemconfig(self.detection_indicator, fill='#00FF00')
                self.detection_label.config(text="ERD DETECTED!", foreground="green")
            else:
                self.detection_canvas.itemconfig(self.detection_indicator, fill='#FF0000')
                self.detection_label.config(text="NO DETECTION", foreground="gray")
    
    def _update_erd_display(self, data):
        """Update ERD values display"""
        erd_values = data.get('erd_values', {})
        
        if self.minimal:
            # Simple text display for minimal mode
            self.erd_text.delete(1.0, tk.END)
            for ch, erd in erd_values.items():
                if ch != 'csp_conf':
                    status = "DETECT" if erd > self.threshold_var.get() else ""
                    self.erd_text.insert(tk.END, f"{ch:8s}: {erd:6.1f}% {status}\n")
        else:
            # Formatted display for full mode
            if hasattr(self, 'erd_display'):
                self.erd_display.delete(1.0, tk.END)
                for ch, erd in erd_values.items():
                    if ch != 'csp_conf':
                        status = "***" if erd > self.threshold_var.get() else "   "
                        self.erd_display.insert(tk.END, f"{status} {ch:8s}: {erd:6.1f}%\n")
    
    def _update_plot(self, data):
        """Update real-time plot"""
        if not hasattr(self, 'time_history'):
            return
        
        # Add time point
        self.time_history.append(data.get('runtime', 0))
        
        # Store ERD values
        for ch, erd in data['erd_values'].items():
            if ch != 'csp_conf' and ch != 'avg':  # Skip non-channel values
                if ch not in self.erd_history:
                    self.erd_history[ch] = deque(maxlen=500)
                self.erd_history[ch].append(erd)
        
        # Update plot
        if len(self.time_history) > 1:
            times = list(self.time_history)
            
            # Update or create lines
            for i, (ch, values) in enumerate(self.erd_history.items()):
                if ch not in self.plot_lines:
                    color = self.colors[i % len(self.colors)]
                    line, = self.ax.plot([], [], color=color, linewidth=2, label=ch)
                    self.plot_lines[ch] = line
                
                self.plot_lines[ch].set_data(times[-len(values):], list(values))
            
            # Update axis limits
            self.ax.set_xlim(max(0, times[-1] - 30), times[-1] + 1)
            
            # Update legend
            if len(self.plot_lines) > 0 and not self.ax.get_legend():
                self.ax.legend(loc='upper right')
            
            # Redraw
            self.canvas.draw_idle()
    
    def start_collecting(self, mode):
        """Start collecting training data"""
        # This would need to interface with the BCI system's keyboard handler
        # For now, just update the GUI
        self.train_status_label.config(text=f"Collecting {mode.upper()} data...")
        self.rest_btn.config(state=tk.DISABLED if mode == 'rest' else tk.NORMAL)
        self.mi_btn.config(state=tk.DISABLED if mode == 'mi' else tk.NORMAL)
        self.stop_collect_btn.config(state=tk.NORMAL)
    
    def stop_collecting(self):
        """Stop collecting training data"""
        self.train_status_label.config(text="Collection stopped")
        self.rest_btn.config(state=tk.NORMAL)
        self.mi_btn.config(state=tk.NORMAL)
        self.stop_collect_btn.config(state=tk.DISABLED)
    
    def train_model(self):
        """Train CSP+SVM model"""
        if hasattr(self.bci_system.erd_detector, 'csp_svm_detector'):
            detector = self.bci_system.erd_detector.csp_svm_detector
            if detector:
                rest_count = len(detector.training_data['rest'])
                mi_count = len(detector.training_data['mi'])
                
                if rest_count >= 10 and mi_count >= 10:
                    self.log_status("Training CSP+SVM model...")
                    detector.train()
                    if detector.is_trained:
                        self.train_status_label.config(text="Model trained successfully!")
                else:
                    messagebox.showwarning("Insufficient Data", 
                                         f"Need at least 10 samples each.\n"
                                         f"Current: Rest={rest_count}, MI={mi_count}")
    
    def save_model(self):
        """Save CSP+SVM model"""
        if hasattr(self.bci_system.erd_detector, 'csp_svm_detector'):
            detector = self.bci_system.erd_detector.csp_svm_detector
            if detector and detector.is_trained:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
                )
                if filename:
                    detector.save_model(filename)
                    self.log_status(f"Model saved to {filename}")
            else:
                messagebox.showwarning("No Model", "No trained model to save")
    
    def load_model(self):
        """Load CSP+SVM model"""
        if hasattr(self.bci_system.erd_detector, 'csp_svm_detector'):
            filename = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                self.bci_system.erd_detector.csp_svm_detector.load_model(filename)
                self.log_status(f"Model loaded from {filename}")
                self.train_status_label.config(text="Model loaded successfully!")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_detection()
        self.bci_system.running = False
        self.root.destroy()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


# Test function
if __name__ == "__main__":
    print("This GUI requires the BCI system to be initialized.")
    print("Please run: python bci_erd_main.py")