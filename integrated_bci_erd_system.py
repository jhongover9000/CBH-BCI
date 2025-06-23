'''
integrated_bci_erd_system.py
---------------------------
Complete integrated BCI ERD system with CSP+SVM, GUI controls, model persistence,
and real-time plotting with annotation markers
'''

import numpy as np
import time
from datetime import datetime
import threading
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import os

# Matplotlib imports for plotting
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import existing modules
from bci_erd_main import BCIConfig, Preprocessor, ERDDetectionSystem
from csp_svm_integration import CSPSVMDetector, RobustBaseline
from receivers import virtual_receiver, livestream_receiver
from broadcasting import TCP_Server

# Fix for unpacking error - ensure consistent return values
def fix_detect_erd_method(erd_detector):
    """Fix the detect_erd method to always return 3 values"""
    original_detect = erd_detector.detect_erd
    
    def fixed_detect_erd(data):
        result = original_detect(data)
        if len(result) == 2:
            detected, erd_values = result
            # Calculate confidence from ERD values
            avg_erd = np.mean(list(erd_values.values())) if erd_values else 0
            confidence = min(100, max(0, avg_erd))  # Clamp to 0-100
            return detected, erd_values, confidence
        else:
            return result
    
    erd_detector.detect_erd = fixed_detect_erd
    return erd_detector


class IntegratedBCIERDGUI:
    """Complete GUI for BCI ERD System with all controls and real-time plotting"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BCI ERD Detection System - Integrated Control Panel")
        self.root.geometry("1400x1000")
        
        # System components
        self.bci_system = None
        self.csp_svm_detector = None
        self.is_running = False
        self.collection_mode = None  # None, 'rest', or 'mi'
        
        # Queue for thread communication
        self.gui_queue = queue.Queue()
        
        # Data storage for plots
        self.time_history = deque(maxlen=500)
        self.erd_history = {}
        self.annotation_markers = []  # Store annotation markers on plot
        self.erd_detection_markers = []  # Store ERD detection markers
        
        # Create GUI elements
        self._create_widgets()
        
        # Start GUI update loop
        self._update_gui()
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container with notebook for tabbed interface
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Control Panel
        control_tab = ttk.Frame(notebook)
        notebook.add(control_tab, text="Control Panel")
        self._create_control_panel(control_tab)
        
        # Tab 2: Real-time Visualization
        # viz_tab = ttk.Frame(notebook)
        # notebook.add(control_tab, text="Real-time Visualization")
        # self._create_visualization_panel(control_tab)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def _create_control_panel(self, parent):
        """Create the control panel tab"""
        # Left and right columns
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === Connection Panel ===
        conn_frame = ttk.LabelFrame(left_frame, text="Connection Settings", padding="10")
        conn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Mode selection
        ttk.Label(conn_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="virtual")
        ttk.Radiobutton(conn_frame, text="Virtual", variable=self.mode_var, 
                       value="virtual").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(conn_frame, text="Livestream", variable=self.mode_var, 
                       value="livestream").grid(row=0, column=2, sticky=tk.W)
        
        # IP and Port for livestream
        ttk.Label(conn_frame, text="IP:").grid(row=1, column=0, sticky=tk.W)
        self.ip_var = tk.StringVar(value=BCIConfig.LIVESTREAM_IP)
        ttk.Entry(conn_frame, textvariable=self.ip_var, width=15).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(conn_frame, text="Port:").grid(row=1, column=2, sticky=tk.W, padx=(10,0))
        self.port_var = tk.StringVar(value=str(BCIConfig.LIVESTREAM_PORT))
        ttk.Entry(conn_frame, textvariable=self.port_var, width=8).grid(row=1, column=3, sticky=tk.W)
        
        # Initialize button
        self.init_btn = ttk.Button(conn_frame, text="Initialize System", 
                                  command=self.initialize_system)
        self.init_btn.grid(row=2, column=0, columnspan=4, pady=10)
        
        # === ERD Settings Panel ===
        erd_frame = ttk.LabelFrame(left_frame, text="ERD Detection Settings", padding="10")
        erd_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Threshold slider
        ttk.Label(erd_frame, text="ERD Threshold (%):").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=BCIConfig.ERD_THRESHOLD)
        self.threshold_slider = ttk.Scale(erd_frame, from_=10, to=100, 
                                         variable=self.threshold_var, orient=tk.HORIZONTAL)
        self.threshold_slider.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.threshold_label = ttk.Label(erd_frame, text=f"{BCIConfig.ERD_THRESHOLD:.0f}%")
        self.threshold_label.grid(row=0, column=2)
        self.threshold_slider.configure(command=self._update_threshold)
        
        # Detection mode
        ttk.Label(erd_frame, text="Detection Mode:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.detection_mode_var = tk.StringVar(value="simple")
        ttk.Radiobutton(erd_frame, text="Simple ERD", variable=self.detection_mode_var,
                       value="simple").grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(erd_frame, text="Advanced (CSP+SVM)", variable=self.detection_mode_var,
                       value="advanced").grid(row=2, column=1, sticky=tk.W)
        
        # Preprocessing options
        self.use_car_var = tk.BooleanVar(value=BCIConfig.USE_CAR)
        ttk.Checkbutton(erd_frame, text="Use CAR", 
                       variable=self.use_car_var).grid(row=3, column=0, sticky=tk.W, pady=5)
        
        # Show annotations
        self.show_annotations_var = tk.BooleanVar(value=BCIConfig.SHOW_ANNOTATIONS)
        ttk.Checkbutton(erd_frame, text="Show Annotations", 
                       variable=self.show_annotations_var).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # === CSP+SVM Training Panel ===
        train_frame = ttk.LabelFrame(right_frame, text="CSP+SVM Training", padding="10")
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Training status
        self.train_status_label = ttk.Label(train_frame, text="Status: Not initialized")
        self.train_status_label.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Collection buttons
        self.rest_btn = ttk.Button(train_frame, text="Collect REST", 
                                  command=lambda: self.start_collecting('rest'),
                                  state=tk.DISABLED)
        self.rest_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.mi_btn = ttk.Button(train_frame, text="Collect MI", 
                                command=lambda: self.start_collecting('mi'),
                                state=tk.DISABLED)
        self.mi_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.stop_collect_btn = ttk.Button(train_frame, text="Stop Collecting", 
                                          command=self.stop_collecting,
                                          state=tk.DISABLED)
        self.stop_collect_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Training data count
        self.data_count_label = ttk.Label(train_frame, text="Rest: 0 | MI: 0")
        self.data_count_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Model management
        ttk.Button(train_frame, text="Save Model", 
                  command=self.save_model).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Load Model", 
                  command=self.load_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Train Now", 
                  command=self.train_model).grid(row=3, column=2, pady=5)
        
        # === Control Panel ===
        control_frame = ttk.LabelFrame(left_frame, text="System Control", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Main control buttons
        self.start_btn = ttk.Button(control_frame, text="Start Detection", 
                                   command=self.start_detection, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Detection", 
                                  command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Baseline controls
        ttk.Button(control_frame, text="Manual Baseline", 
                  command=self.manual_baseline).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Reset Baseline", 
                  command=self.reset_baseline).grid(row=0, column=3, padx=5)
        
        # Broadcasting control
        self.broadcast_var = tk.BooleanVar(value=BCIConfig.BROADCAST_ENABLED)
        ttk.Checkbutton(control_frame, text="Enable Broadcasting", 
                       variable=self.broadcast_var).grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # # === Status Display ===
        status_frame = ttk.LabelFrame(right_frame, text="Detection Status", padding="10")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # # Create text widget for status
        self.status_text = tk.Text(status_frame, height=15, width=50)
        self.status_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ERD values display
        values_frame = ttk.LabelFrame(right_frame, text="Real-Time ERD Values", padding="10")
        values_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.erd_display = tk.Text(values_frame, width=40, height=6, font=("Courier", 10))
        self.erd_display.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        
        # # Scrollbar
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # === Real-time Metrics ===
        metrics_frame = ttk.LabelFrame(left_frame, text="Real-time Metrics", padding="10")
        metrics_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.metrics_labels = {}
        metrics = ["Runtime", "Detection Count", "Detection Rate", "Avg ERD", "CSP Confidence"]
        for i, metric in enumerate(metrics):
            ttk.Label(metrics_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, padx=5)
            self.metrics_labels[metric] = ttk.Label(metrics_frame, text="--")
            self.metrics_labels[metric].grid(row=i, column=1, sticky=tk.W, padx=5)
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(left_frame, text="ERD Trend with Annotations", padding="5")
        plot_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 3), dpi=20)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('ERD (%)')
        self.ax.set_ylim(-30, 120)
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pre-create plot lines
        self.plot_lines = {}
        self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Threshold line
        self.threshold_line = self.ax.axhline(y=BCIConfig.ERD_THRESHOLD, color='k', 
                                             linestyle='--', alpha=0.5, label='Threshold')
        
        # Annotation correlation display
        # ann_frame = ttk.LabelFrame(parent, text="Annotation Correlation", padding="5")
        # ann_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # self.ann_text = tk.Text(ann_frame, width=80, height=4, font=("Courier", 9))
        # self.ann_text.pack()
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        
    def _update_threshold(self, value):
        """Update threshold value"""
        BCIConfig.ERD_THRESHOLD = float(value)
        self.threshold_label.config(text=f"{float(value):.0f}%")
        
        # Update threshold line on plot
        if hasattr(self, 'threshold_line'):
            self.threshold_line.set_ydata([float(value), float(value)])
            self.canvas.draw_idle()
        
    def initialize_system(self):
        """Initialize the BCI system"""
        try:
            self.log_status("Initializing BCI system...")
            
            # Update config
            BCIConfig.LIVESTREAM_IP = self.ip_var.get()
            BCIConfig.LIVESTREAM_PORT = int(self.port_var.get())
            BCIConfig.VIRTUAL = (self.mode_var.get() == "virtual")
            BCIConfig.USE_CAR = self.use_car_var.get()
            BCIConfig.BROADCAST_ENABLED = self.broadcast_var.get()
            BCIConfig.SHOW_ANNOTATIONS = self.show_annotations_var.get()
            BCIConfig.GUI_ENABLED = False  # We're already in GUI
            
            # Create BCI system
            from bci_erd_main import BCIERDSystem
            self.bci_system = BCIERDSystem()
            
            # Create custom args object
            class Args:
                virtual = BCIConfig.VIRTUAL
                
            args = Args()
            
            # Initialize system
            self.bci_system.initialize(args)
            
            # Fix the detect_erd method
            fix_detect_erd_method(self.bci_system.erd_detector)
            
            # Initialize CSP+SVM if advanced mode
            if self.detection_mode_var.get() == "advanced":
                self.setup_csp_svm()
            
            # Enable controls
            self.start_btn.config(state=tk.NORMAL)
            self.rest_btn.config(state=tk.NORMAL)
            self.mi_btn.config(state=tk.NORMAL)
            
            self.log_status("System initialized successfully!")
            self.train_status_label.config(text="Status: Initialized, ready for training")
            
        except Exception as e:
            self.log_status(f"Initialization failed: {str(e)}")
            messagebox.showerror("Initialization Error", str(e))
            
    def setup_csp_svm(self):
        """Setup CSP+SVM detector"""
        if self.bci_system and self.bci_system.erd_detector:
            self.csp_svm_detector = CSPSVMDetector(
                fs=self.bci_system.fs,
                n_channels=len(self.bci_system.erd_detector.erd_channel_indices),
                use_multiband=True
            )
            
            # Integrate with ERD system
            self.bci_system.erd_detector.csp_svm_detector = self.csp_svm_detector
            
            # Override detect method
            self._integrate_csp_svm_detection()
            
            self.log_status("CSP+SVM detector initialized")
            
    def _integrate_csp_svm_detection(self):
        """Integrate CSP+SVM with ERD detection"""
        original_detect = self.bci_system.erd_detector.detect_erd
        
        def enhanced_detect(data):
            # Get original ERD results
            erd_detected, erd_values, erd_confidence = original_detect(data)
            
            # If CSP+SVM is trained, use it
            if self.csp_svm_detector and self.csp_svm_detector.is_trained:
                # Prepare data for CSP+SVM
                if data.ndim == 2:
                    window_data = data[self.bci_system.erd_detector.erd_channel_indices, :]
                    csp_pred, csp_conf = self.csp_svm_detector.predict(window_data)
                    
                    if csp_pred is not None:
                        # Combine decisions
                        combined_conf = 0.6 * csp_conf + 0.4 * (erd_confidence / 100.0)
                        detected = combined_conf > 0.5
                        
                        # Add CSP info to values
                        erd_values['csp_conf'] = csp_conf * 100
                        
                        return detected, erd_values, combined_conf * 100
            
            return erd_detected, erd_values, erd_confidence
        
        self.bci_system.erd_detector.detect_erd = enhanced_detect
        
    def start_detection(self):
        """Start ERD detection"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Clear plot history
            self.time_history.clear()
            self.erd_history.clear()
            
            # Start detection in separate thread
            self.detection_thread = threading.Thread(target=self._run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.log_status("Detection started")
            
    def stop_detection(self):
        """Stop ERD detection"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_status("Detection stopped")
        
    def _run_detection(self):
        """Run detection loop (in separate thread)"""
        try:
            self.bci_system.running = True
            self.bci_system.session_start_time = time.time()
            last_update_time = time.time()
            
            # Ensure we have the operating window size
            operating_window_size = int(BCIConfig.OPERATING_WINDOW_DURATION * self.bci_system.fs)
            
            while self.is_running and self.bci_system.running:
                # Get data
                data = self.bci_system.receiver.get_data()
                if data is None:
                    continue
                
                # Add to buffer
                for i in range(data.shape[1]):
                    self.bci_system.main_buffer.append(data[:, i])
                
                # Add to collection if active
                if self.collection_mode and len(self.bci_system.main_buffer) >= operating_window_size:
                    window_data = np.array(list(self.bci_system.main_buffer))[-operating_window_size:].T
                    
                    if self.csp_svm_detector:
                        # Get ERD channels only
                        erd_window = window_data[self.bci_system.erd_detector.erd_channel_indices, :]
                        label = 0 if self.collection_mode == 'rest' else 1
                        self.csp_svm_detector.collect_training_data(erd_window, label)
                        
                        # Update counts
                        self._update_data_counts()
                
                # Process baseline
                if not self.bci_system.baseline_ready:
                    self.bci_system.erd_detector.add_to_baseline(data)
                    if self.bci_system.erd_detector.baseline_calculated:
                        self.bci_system.baseline_ready = True
                        self.gui_queue.put({'baseline_ready': True})
                
                # Check for annotations
                self.bci_system._check_for_annotations()
                
                # Detect ERD
                if self.bci_system.baseline_ready and len(self.bci_system.main_buffer) >= operating_window_size:
                    window_data = np.array(list(self.bci_system.main_buffer))[-operating_window_size:].T
                    
                    detected, erd_values, confidence = self.bci_system.erd_detector.detect_erd(window_data)
                    
                    if detected:
                        self.bci_system.detection_count += 1
                        current_runtime = time.time() - self.bci_system.session_start_time
                        self.bci_system.erd_detection_times.append(current_runtime)
                        
                        if self.bci_system.config.BROADCAST_ENABLED:
                            self.bci_system.receiver.use_classification(1)
                    
                    # Update display
                    current_time = time.time()
                    if current_time - last_update_time >= 0.1:  # Update more frequently for smoother plot
                        runtime = current_time - self.bci_system.session_start_time
                        self.gui_queue.put({
                            'detected': detected,
                            'erd_values': erd_values,
                            'runtime': runtime,
                            'count': self.bci_system.detection_count,
                            'confidence': confidence,
                            'annotations': self.bci_system.annotations,
                            'erd_detection_times': list(self.bci_system.erd_detection_times)
                        })
                        last_update_time = current_time
                        
        except Exception as e:
            self.gui_queue.put({'error': str(e)})
            
    def start_collecting(self, mode):
        """Start collecting training data"""
        self.collection_mode = mode
        self.rest_btn.config(state=tk.DISABLED if mode == 'rest' else tk.NORMAL)
        self.mi_btn.config(state=tk.DISABLED if mode == 'mi' else tk.NORMAL)
        self.stop_collect_btn.config(state=tk.NORMAL)
        
        label_name = "REST" if mode == 'rest' else "MOTOR IMAGERY"
        self.log_status(f"Collecting {label_name} data...")
        
    def stop_collecting(self):
        """Stop collecting training data"""
        self.collection_mode = None
        self.rest_btn.config(state=tk.NORMAL)
        self.mi_btn.config(state=tk.NORMAL)
        self.stop_collect_btn.config(state=tk.DISABLED)
        self.log_status("Stopped collecting")
        
    def _update_data_counts(self):
        """Update training data counts"""
        if self.csp_svm_detector:
            rest_count = len(self.csp_svm_detector.training_data['rest'])
            mi_count = len(self.csp_svm_detector.training_data['mi'])
            self.data_count_label.config(text=f"Rest: {rest_count} | MI: {mi_count}")
            
            # Auto-train if enough data
            if (rest_count >= 20 and mi_count >= 20 and 
                not self.csp_svm_detector.is_trained):
                self.train_model()
                
    def train_model(self):
        """Manually trigger model training"""
        if self.csp_svm_detector:
            rest_count = len(self.csp_svm_detector.training_data['rest'])
            mi_count = len(self.csp_svm_detector.training_data['mi'])
            
            if rest_count >= 10 and mi_count >= 10:
                self.log_status("Training CSP+SVM model...")
                self.csp_svm_detector.train()
                
                if self.csp_svm_detector.is_trained:
                    self.train_status_label.config(text="Status: Model trained!")
                    self.log_status("Model training completed")
                else:
                    self.log_status("Model training failed")
            else:
                self.log_status(f"Need more data: Rest={rest_count}/10, MI={mi_count}/10")
                
    def save_model(self):
        """Save trained model to file"""
        if self.csp_svm_detector and self.csp_svm_detector.is_trained:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                self.csp_svm_detector.save_model(filename)
                self.log_status(f"Model saved to {filename}")
        else:
            messagebox.showwarning("No Model", "No trained model to save")
            
    def load_model(self):
        """Load model from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename and self.csp_svm_detector:
            self.csp_svm_detector.load_model(filename)
            if self.csp_svm_detector.is_trained:
                self.train_status_label.config(text="Status: Model loaded!")
                self.log_status(f"Model loaded from {filename}")
                # Re-integrate with detection
                self._integrate_csp_svm_detection()
                
    def manual_baseline(self):
        """Manually calculate baseline"""
        if self.bci_system and self.bci_system.erd_detector:
            success = self.bci_system.manual_baseline_calculation()
            if success:
                self.log_status("Manual baseline calculated")
            else:
                self.log_status("Manual baseline calculation failed")
                
    def reset_baseline(self):
        """Reset baseline"""
        if self.bci_system and self.bci_system.erd_detector:
            self.bci_system.reset_baseline()
            self.log_status("Baseline reset")
            
    def log_status(self, message):
        """Log message to status display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        
    def _update_gui(self):
        """Update GUI from queue"""
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                
                if 'error' in msg:
                    self.log_status(f"Error: {msg['error']}")
                    
                elif 'baseline_ready' in msg:
                    self.log_status("Baseline established!")
                    # self.baseline_status.config(text="Ready", foreground="green")
                    
                elif 'detected' in msg:
                    # Update metrics
                    self.metrics_labels["Runtime"].config(text=f"{msg['runtime']:.1f}s")
                    self.metrics_labels["Detection Count"].config(text=str(msg['count']))
                    
                    rate = (msg['count'] / msg['runtime']) * 60 if msg['runtime'] > 0 else 0
                    self.metrics_labels["Detection Rate"].config(text=f"{rate:.1f}/min")
                    
                    if 'avg' in msg['erd_values']:
                        self.metrics_labels["Avg ERD"].config(text=f"{msg['erd_values']['avg']:.1f}%")
                    
                    if 'csp_conf' in msg['erd_values']:
                        self.metrics_labels["CSP Confidence"].config(text=f"{msg['erd_values']['csp_conf']:.1f}%")
                    
                    # Update detection indicator
                    # if msg['detected']:
                    #     self.detection_canvas.itemconfig(self.detection_indicator, fill='#00FF00')
                    #     self.detection_label.config(text="ERD DETECTED!", foreground="green")
                    # else:
                    #     self.detection_canvas.itemconfig(self.detection_indicator, fill='#FF0000')
                    #     self.detection_label.config(text="NO DETECTION", foreground="gray")
                    
                    # Update ERD display
                    self._update_erd_display(msg['erd_values'])
                    
                    # Update plot with annotations
                    # self._update_plot_with_annotations(msg)
                    
                    # Update annotation correlation
                    # self._update_annotation_correlation(msg)
                    
                    # Log detection
                    if msg['detected']:
                        self.log_status(f"ERD DETECTED! Confidence: {msg.get('confidence', 0):.1f}%")
                        
        except queue.Empty:
            pass
            
        # Schedule next update
        self.root.after(10, self._update_gui)
        
    def _update_erd_display(self, erd_values):
        """Update ERD values text display"""
        self.erd_display.delete(1.0, tk.END)
        
        for ch, erd in erd_values.items():
            if ch != 'csp_conf':  # Skip CSP confidence in this display
                status = "DETECT" if erd > self.threshold_var.get() else "------"
                marker = "***" if erd > self.threshold_var.get() else "   "
                self.erd_display.insert(tk.END, f"{marker} {ch:8s} {erd:6.1f}%  {status}\n")
    
    def _update_plot_with_annotations(self, data):
        """Update ERD trend plot with annotation markers"""
        # Store time
        self.time_history.append(data.get('runtime', 0))
        
        # Store ERD values
        for ch, erd in data['erd_values'].items():
            if ch != 'csp_conf':  # Don't plot CSP confidence
                if ch not in self.erd_history:
                    self.erd_history[ch] = deque(maxlen=500)
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
            
            # Add annotation markers if enabled
            if self.show_annotations_var.get():
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
            if len(self.plot_lines) > 0 and not self.ax.get_legend():
                self.ax.legend(loc='upper right')
            
            # Redraw
            self.canvas.draw_idle()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_detection()
        if self.bci_system:
            self.bci_system.cleanup()
        self.root.destroy()


def main():
    """Main entry point for integrated system"""
    # Create and run GUI
    gui = IntegratedBCIERDGUI()
    gui.root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    gui.run()


if __name__ == "__main__":
    main()