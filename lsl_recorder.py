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

class EEGMarkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Marker GUI")
        self.root.geometry("550x850")  # Initial window size; scroll will allow access to all controls.
        self.root.minsize(600, 600)

        # Adjustable durations (in seconds)
        self.blank1_duration = tk.IntVar(value=2)      # Pre-trial blank
        self.baseline1_duration = tk.IntVar(value=5)     # First baseline
        self.motor_duration = tk.IntVar(value=5)         # Motor execution (Block 1)
        self.blank2_duration = tk.IntVar(value=2)        # Blank after motor execution
        self.baseline2_duration = tk.IntVar(value=5)     # Second baseline
        self.imagery_duration = tk.IntVar(value=5)       # Motor imagery (Block 2)
        self.rest_duration = tk.IntVar(value=5)          # Rest (Block 2)

        self.total_trials = tk.IntVar(value=10)
        self.file_format = tk.StringVar(value="fif")

        # Activity selection checkboxes
        self.select_motor_exec = tk.BooleanVar(value=True)
        self.select_motor_imagery = tk.BooleanVar(value=True)
        self.select_rest = tk.BooleanVar(value=True)

        self.running = False
        self.current_trial = 0
        self.eeg_data = []
        self.timestamps = []
        self.markers = []
        self.start_timestamp_lsl = None

        self.eeg_inlet = None
        self.marker_outlet = None
        self.eeg_channels = []
        self.eeg_indices = []
        self.sfreq = 0

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

        def responsive_font(base_size):
            width = self.root.winfo_width()
            scale = max(min(width / 600, 1.0), 0.75)
            return int(base_size * scale)

        def update_fonts():
            available_fonts = ["Noto Sans", "DejaVu Sans", "Liberation Sans", "Arial", "TkDefaultFont"]
            font_name = available_fonts[0]
            for fn in available_fonts:
                try:
                    tk.Label(self.root, font=(fn, 1))  # Test font
                    font_name = fn
                    break
                except Exception:
                    continue

            font_small = (font_name, responsive_font(9))
            font_normal = (font_name, responsive_font(10))
            font_bold = (font_name, responsive_font(12), 'bold')

            self.trial_label.config(font=font_bold)
            self.log_box.config(font=("DejaVu Sans Mono", responsive_font(11)))
            for widget in self.dynamic_labels:
                widget.config(font=font_normal)
            for widget in self.dynamic_buttons:
                widget.config(font=font_normal)

            if hasattr(self, 'cue_label'):
                screen_width = self.cue_win.winfo_screenwidth()
                screen_height = self.cue_win.winfo_screenheight()
                cue_font_size = 20  # Larger cue font
                self.cue_label.config(font=(font_name, cue_font_size, "bold"))

        # --- Pre-trial Blank Duration ---
        lbl_blank1 = tk.Label(scrollable_frame, text="Pre-trial Blank Duration (s)")
        lbl_blank1.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_blank1)
        blank1_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                 variable=self.blank1_duration,
                                 command=lambda val: self.update_label(self.blank1_val_label, val))
        blank1_scale.grid(row=1, column=0, sticky="ew")
        self.blank1_val_label = tk.Label(scrollable_frame, text="2 s")
        self.blank1_val_label.grid(row=2, column=0, sticky="e", pady=(0, 10))

        # --- Baseline 1 Duration ---
        lbl_baseline1 = tk.Label(scrollable_frame, text="Baseline 1 Duration (s)")
        lbl_baseline1.grid(row=3, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_baseline1)
        baseline1_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                    variable=self.baseline1_duration,
                                    command=lambda val: self.update_label(self.baseline1_val_label, val))
        baseline1_scale.grid(row=4, column=0, sticky="ew")
        self.baseline1_val_label = tk.Label(scrollable_frame, text="5 s")
        self.baseline1_val_label.grid(row=5, column=0, sticky="e", pady=(0, 10))

        # --- Motor Execution Duration & Checkbox (Block 1) ---
        lbl_motor = tk.Label(scrollable_frame, text="Motor Execution Duration (s)")
        lbl_motor.grid(row=6, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_motor)
        motor_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                variable=self.motor_duration,
                                command=lambda val: self.update_label(self.motor_val_label, val))
        motor_scale.grid(row=7, column=0, sticky="ew")
        self.motor_val_label = tk.Label(scrollable_frame, text="5 s")
        self.motor_val_label.grid(row=8, column=0, sticky="e", pady=(0, 10))
        cb_motor = tk.Checkbutton(scrollable_frame, text="Enable Motor Execution", variable=self.select_motor_exec)
        cb_motor.grid(row=9, column=0, sticky="w")
        self.dynamic_buttons.append(cb_motor)

        # --- Blank 2 Duration ---
        lbl_blank2 = tk.Label(scrollable_frame, text="Post-Execution Blank Duration (s)")
        lbl_blank2.grid(row=10, column=0, sticky="w", pady=(10, 2))
        self.dynamic_labels.append(lbl_blank2)
        blank2_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                 variable=self.blank2_duration,
                                 command=lambda val: self.update_label(self.blank2_val_label, val))
        blank2_scale.grid(row=11, column=0, sticky="ew")
        self.blank2_val_label = tk.Label(scrollable_frame, text="2 s")
        self.blank2_val_label.grid(row=12, column=0, sticky="e", pady=(0, 10))

        # --- Baseline 2 Duration ---
        lbl_baseline2 = tk.Label(scrollable_frame, text="Baseline 2 Duration (s)")
        lbl_baseline2.grid(row=13, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_baseline2)
        baseline2_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                    variable=self.baseline2_duration,
                                    command=lambda val: self.update_label(self.baseline2_val_label, val))
        baseline2_scale.grid(row=14, column=0, sticky="ew")
        self.baseline2_val_label = tk.Label(scrollable_frame, text="5 s")
        self.baseline2_val_label.grid(row=15, column=0, sticky="e", pady=(0, 10))

        # --- Activity Phase (Block 2) Selection & Durations ---
        lbl_activity = tk.Label(scrollable_frame, text="Block 2 Activity (choose one or both):")
        lbl_activity.grid(row=16, column=0, sticky="w", pady=(10, 2))
        self.dynamic_labels.append(lbl_activity)
        act_frame = tk.Frame(scrollable_frame)
        act_frame.grid(row=17, column=0, sticky="w", pady=(0, 10))
        cb_imagery = tk.Checkbutton(act_frame, text="Motor Imagery", variable=self.select_motor_imagery)
        cb_imagery.pack(side="left", padx=(0, 10))
        self.dynamic_buttons.append(cb_imagery)
        cb_rest = tk.Checkbutton(act_frame, text="Rest", variable=self.select_rest)
        cb_rest.pack(side="left")
        self.dynamic_buttons.append(cb_rest)
        # Motor Imagery Duration slider
        lbl_imagery = tk.Label(scrollable_frame, text="Motor Imagery Duration (s)")
        lbl_imagery.grid(row=18, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_imagery)
        imagery_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                                  variable=self.imagery_duration,
                                  command=lambda val: self.update_label(self.imagery_val_label, val))
        imagery_scale.grid(row=19, column=0, sticky="ew")
        self.imagery_val_label = tk.Label(scrollable_frame, text="5 s")
        self.imagery_val_label.grid(row=20, column=0, sticky="e", pady=(0, 10))
        # Rest Duration slider
        lbl_rest = tk.Label(scrollable_frame, text="Rest Duration (s)")
        lbl_rest.grid(row=21, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_rest)
        rest_scale = ttk.Scale(scrollable_frame, from_=1, to=10, orient="horizontal",
                               variable=self.rest_duration,
                               command=lambda val: self.update_label(self.rest_val_label, val))
        rest_scale.grid(row=22, column=0, sticky="ew")
        self.rest_val_label = tk.Label(scrollable_frame, text="5 s")
        self.rest_val_label.grid(row=23, column=0, sticky="e", pady=(0, 10))

        # --- Total Trials ---
        lbl_trials = tk.Label(scrollable_frame, text="Total Trials")
        lbl_trials.grid(row=24, column=0, sticky="w", pady=(10, 2))
        self.dynamic_labels.append(lbl_trials)
        t_spinbox = ttk.Spinbox(scrollable_frame, from_=1, to=100, textvariable=self.total_trials, width=5)
        t_spinbox.grid(row=25, column=0, sticky="w", pady=(0, 10))

        # --- File Format ---
        lbl_format = tk.Label(scrollable_frame, text="Save Format")
        lbl_format.grid(row=26, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl_format)
        format_menu = ttk.OptionMenu(scrollable_frame, self.file_format, "fif", "fif", "mat")
        format_menu.grid(row=27, column=0, sticky="ew", pady=(0, 10))

        # --- Trial Label ---
        self.trial_label = tk.Label(scrollable_frame, text="Trial: 0 / 0")
        self.trial_label.grid(row=28, column=0, pady=(10, 5))

        # --- Logs ---
        lbl_logs = tk.Label(scrollable_frame, text="Logs")
        lbl_logs.grid(row=29, column=0, sticky="w")
        self.dynamic_labels.append(lbl_logs)
        self.log_box = tk.Text(scrollable_frame, height=6, state="disabled", bg="#f5f5f5")
        self.log_box.grid(row=30, column=0, sticky="nsew", pady=(0, 10))
        scrollable_frame.grid_rowconfigure(30, weight=2)

        # --- Start and Stop Buttons ---
        self.start_button = tk.Button(scrollable_frame, text="Start Session", command=self.start_stream)
        self.start_button.grid(row=31, column=0, sticky="ew", pady=(0, 5))
        self.dynamic_buttons.append(self.start_button)
        self.stop_button = tk.Button(scrollable_frame, text="Stop and Save", command=self.stop_stream, state="disabled")
        self.stop_button.grid(row=32, column=0, sticky="ew")
        self.dynamic_buttons.append(self.stop_button)

        # Font update binding
        update_fonts()
        self.root.bind('<Configure>', lambda e: update_fonts())
        self.update_fonts = update_fonts



    def update_label(self, label_widget, val):
        label_widget.config(text=f"{int(float(val))} s")

    def update_cue(self, symbol, title):
        self.cue_win.title(title)
        self.cue_label.config(text=symbol)
        self.cue_win.update()
        self.update_fonts()

    def setup_lsl(self):
        self.log("Looking for EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=5)
        if not streams:
            messagebox.showerror("Error", "No EEG stream found.")
            self.root.destroy()
            return

        self.eeg_inlet = StreamInlet(streams[0])
        self.log(f"Connected to EEG stream: {streams[0].name()}")

        marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string')
        self.marker_outlet = StreamOutlet(marker_info)

        info = self.eeg_inlet.info()
        self.sfreq = info.nominal_srate()
        ch_list = info.desc().child("channels").first_child()
        channel_names = []
        while ch_list.name() == "channel":
            channel_names.append(ch_list.child_value("label"))
            ch_list = ch_list.next_sibling()

        self.eeg_channels = [ch for ch in channel_names if not ch.startswith("acc")]
        self.eeg_indices = [i for i, ch in enumerate(channel_names) if ch in self.eeg_channels]
        self.log(f"EEG Channels: {self.eeg_channels}")
        self.log(f"Excluded Channels: {[ch for ch in channel_names if ch.startswith('acc')]}")

    def start_stream(self):
        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.eeg_data = []
        self.timestamps = []
        self.markers = []
        self.current_trial = 0
        self.trial_label.config(text=f"Trial: 0 / {self.total_trials.get()}")

        self.cue_win = tk.Toplevel(self.root)
        self.cue_win.attributes("-fullscreen", True)
        self.cue_win.configure(bg="black")
        self.cue_win.update_idletasks()

        self.cue_label = tk.Label(self.cue_win, text="", fg="white", bg="black")
        self.cue_label.pack(expand=True)

        self.root.bind("<Escape>", lambda e: self.stop_stream())
        self.cue_win.bind("<Escape>", lambda e: self.stop_stream())

        first_sample, self.start_timestamp_lsl = self.eeg_inlet.pull_sample()
        self.timestamps.append(self.start_timestamp_lsl)
        self.eeg_data.append([first_sample[i] for i in self.eeg_indices])

        threading.Thread(target=self.record_loop, daemon=True).start()
        self.trial_loop()

    def stop_stream(self):
        self.running = False
        if hasattr(self, 'cue_win'):
            self.cue_win.destroy()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.log("Recording stopped. Saving data...")
        self.save_data()

    def trial_loop(self):
        if not self.running or self.current_trial >= self.total_trials.get():
            self.stop_stream()
            return
        self.current_trial += 1
        self.trial_label.config(text=f"Trial: {self.current_trial} / {self.total_trials.get()}")
        self.start_blank1_phase()

    def start_blank1_phase(self):
        self.update_cue("", "Blank")
        now = local_clock()
        self.marker_outlet.push_sample(['blank1'])
        rel_time = (now - self.start_timestamp_lsl) * 1000
        self.markers.append(('blank1', now, rel_time))
        self.log(f"Pre-trial blank phase ({self.blank1_duration.get()} s)")
        self.root.after(self.blank1_duration.get() * 1000, self.start_baseline1_phase)

    def start_baseline1_phase(self):
        self.update_cue("+", "Baseline")
        now = local_clock()
        self.marker_outlet.push_sample(['baseline'])
        rel_time = (now - self.start_timestamp_lsl) * 1000
        self.markers.append(('baseline', now, rel_time))
        self.log(f"Baseline 1 phase ({self.baseline1_duration.get()} s)")
        self.root.after(self.baseline1_duration.get() * 1000, self.start_motor_execution_phase)

    def start_motor_execution_phase(self):
        if self.select_motor_exec.get():
            self.update_cue("M", "Motor Execution")
            now = local_clock()
            self.marker_outlet.push_sample(['execution'])
            rel_time = (now - self.start_timestamp_lsl) * 1000
            self.markers.append(('execution', now, rel_time))
            self.log(f"Motor Execution phase ({self.motor_duration.get()} s)")
            self.root.after(self.motor_duration.get() * 1000, self.start_blank2_phase)
        else:
            self.log("Motor Execution phase skipped")
            self.start_blank2_phase()

    def start_blank2_phase(self):
        self.update_cue("", "Blank")
        now = local_clock()
        self.marker_outlet.push_sample(['blank2'])
        rel_time = (now - self.start_timestamp_lsl) * 1000
        self.markers.append(('blank2', now, rel_time))
        self.log(f"Post-execution blank phase ({self.blank2_duration.get()} s)")
        self.root.after(self.blank2_duration.get() * 1000, self.start_baseline2_phase)

    def start_baseline2_phase(self):
        self.update_cue("+", "Baseline")
        now = local_clock()
        self.marker_outlet.push_sample(['baseline'])
        rel_time = (now - self.start_timestamp_lsl) * 1000
        self.markers.append(('baseline', now, rel_time))
        self.log(f"Baseline 2 phase ({self.baseline2_duration.get()} s)")
        self.root.after(self.baseline2_duration.get() * 1000, self.start_activity_phase)

    def start_activity_phase(self):
        imagery = self.select_motor_imagery.get()
        rest = self.select_rest.get()
        if not (imagery or rest):
            messagebox.showerror("Error", "Select at least one activity for Block 2 (Motor Imagery or Rest).")
            self.stop_stream()
            return
        if imagery and rest:
            activity = random.choice(['imagery', 'rest'])
        elif imagery:
            activity = 'imagery'
        else:
            activity = 'rest'
        now = local_clock()
        if activity == 'imagery':
            self.update_cue("I", "Motor Imagery")
            self.marker_outlet.push_sample(['imagery'])
            rel_time = (now - self.start_timestamp_lsl) * 1000
            self.markers.append(('imagery', now, rel_time))
            self.log(f"Motor Imagery phase ({self.imagery_duration.get()} s)")
            self.root.after(self.imagery_duration.get() * 1000, self.trial_loop)
        else:
            self.update_cue("", "Rest")
            self.marker_outlet.push_sample(['rest'])
            rel_time = (now - self.start_timestamp_lsl) * 1000
            self.markers.append(('rest', now, rel_time))
            self.log(f"Rest phase ({self.rest_duration.get()} s)")
            self.root.after(self.rest_duration.get() * 1000, self.trial_loop)

    def record_loop(self):
        while self.running:
            chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=1.0)
            if timestamps:
                for sample, ts in zip(chunk, timestamps):
                    self.eeg_data.append([sample[i] for i in self.eeg_indices])
                    self.timestamps.append(ts)

    def log(self, message):
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.config(state="disabled")
        self.log_box.see("end")

    def save_data(self):
        eeg_array = np.array(self.eeg_data).T
        times_array = np.array(self.timestamps)
        marker_labels = [str(m[0]) for m in self.markers]
        marker_abs_times = [m[1] for m in self.markers]
        marker_rel_times = [m[2] for m in self.markers]

        info = mne.create_info(ch_names=[self.eeg_channels[i] for i in self.eeg_indices],
                               sfreq=self.sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_array, info)

        onsets_sec = np.array(marker_rel_times) / 1000
        annotations = mne.Annotations(onset=onsets_sec,
                              duration=[0.001] * len(onsets_sec),
                              description=marker_labels)
        raw.set_annotations(annotations)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eeg_recording_{timestamp_str}.{self.file_format.get()}"
        if self.file_format.get() == "fif":
            raw.save(filename, overwrite=True)
        else:
            savemat(filename, {
                'data': eeg_array,
                'timestamps': times_array,
                'markers': marker_labels,
                'marker_abs_times': marker_abs_times,
                'marker_rel_times': marker_rel_times,
                'sfreq': self.sfreq,
                'channels': [self.eeg_channels[i] for i in self.eeg_indices]
            })
        self.log(f"Data saved to {filename}")
        print("Saved annotations:", set(marker_labels))
        if filename.endswith(".fif"):
            try:
                raw_check = mne.io.read_raw_fif(filename, preload=False)
                saved_labels = set(raw_check.annotations.description)
                print("Verified annotations in saved file:", saved_labels)
                expected_labels = set(marker_labels)
                if expected_labels == saved_labels:
                    print("✅ All marker labels saved correctly.")
                else:
                    print("⚠️ Mismatch in saved labels! Expected:", expected_labels, "Got:", saved_labels)
            except Exception as e:
                print("Error verifying saved annotations:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGMarkerGUI(root)
    root.mainloop()
