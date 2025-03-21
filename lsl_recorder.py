import numpy as np
import mne
import threading
import time
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from scipy.io import savemat

class EEGMarkerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Marker GUI")
        self.root.geometry("500x550")
        self.root.minsize(400, 400)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.baseline_duration = tk.IntVar(value=5)
        self.imagery_duration = tk.IntVar(value=5)
        self.total_trials = tk.IntVar(value=10)
        self.file_format = tk.StringVar(value="fif")

        self.running = False
        self.current_trial = 0
        self.eeg_data = []
        self.timestamps = []
        self.markers = []
        self.recording_start_time = None
        self.start_timestamp_lsl = None

        self.eeg_inlet = None
        self.marker_outlet = None
        self.eeg_channels = []
        self.eeg_indices = []
        self.sfreq = 0

        self.setup_ui()
        self.setup_lsl()

    def setup_ui(self):
        def responsive_font(base_size):
            width = self.root.winfo_width()
            scale = max(min(width / 500, 2.0), 1.0)
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

            font_small = (font_name, responsive_font(10))
            font_normal = (font_name, responsive_font(12))
            font_bold = (font_name, responsive_font(14), 'bold')

            self.b_val_label.config(font=font_small)
            self.i_val_label.config(font=font_small)
            self.trial_label.config(font=font_bold)
            self.log_box.config(font=("DejaVu Sans Mono", responsive_font(11)))
            for widget in self.dynamic_labels:
                widget.config(font=font_normal)
            for widget in self.dynamic_buttons:
                widget.config(font=font_normal)
            if hasattr(self, 'cue_label'):
                screen_width = self.cue_win.winfo_screenwidth()
                screen_height = self.cue_win.winfo_screenheight()
                cue_font_size = min(screen_width, screen_height) // 2  # Larger cue font
                self.cue_label.config(font=(font_name, cue_font_size, "bold"))

        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.grid(sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        for i in range(12):
            frame.grid_rowconfigure(i, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.dynamic_labels = []
        self.dynamic_buttons = []

        # Baseline Duration
        lbl1 = tk.Label(frame, text="Baseline Duration (s)")
        lbl1.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl1)

        b_scale = ttk.Scale(frame, from_=1, to=10, orient="horizontal",
                            variable=self.baseline_duration,
                            command=lambda val: self.update_label(self.b_val_label, val))
        b_scale.grid(row=1, column=0, sticky="ew")

        self.b_val_label = tk.Label(frame, text="5 s")
        self.b_val_label.grid(row=2, column=0, sticky="e", pady=(0, 10))

        # Imagery Duration
        lbl2 = tk.Label(frame, text="Imagery Duration (s)")
        lbl2.grid(row=3, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl2)

        i_scale = ttk.Scale(frame, from_=1, to=10, orient="horizontal",
                            variable=self.imagery_duration,
                            command=lambda val: self.update_label(self.i_val_label, val))
        i_scale.grid(row=4, column=0, sticky="ew")

        self.i_val_label = tk.Label(frame, text="5 s")
        self.i_val_label.grid(row=5, column=0, sticky="e", pady=(0, 10))

        # Total Trials
        lbl3 = tk.Label(frame, text="Total Trials")
        lbl3.grid(row=6, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl3)

        t_spinbox = ttk.Spinbox(frame, from_=1, to=100, textvariable=self.total_trials, width=5)
        t_spinbox.grid(row=7, column=0, sticky="w", pady=(0, 10))

        # File Format
        lbl4 = tk.Label(frame, text="Save Format")
        lbl4.grid(row=8, column=0, sticky="w", pady=(0, 2))
        self.dynamic_labels.append(lbl4)

        format_menu = ttk.OptionMenu(frame, self.file_format, "fif", "fif", "mat")
        format_menu.grid(row=9, column=0, sticky="ew", pady=(0, 10))

        # Trial Label
        self.trial_label = tk.Label(frame, text="Trial: 0 / 0")
        self.trial_label.grid(row=10, column=0, pady=(10, 5))

        # Logs
        lbl5 = tk.Label(frame, text="Logs")
        lbl5.grid(row=11, column=0, sticky="w")
        self.dynamic_labels.append(lbl5)

        self.log_box = tk.Text(frame, height=5, state="disabled", bg="#f5f5f5")
        self.log_box.grid(row=12, column=0, sticky="nsew", pady=(0, 10))
        frame.grid_rowconfigure(12, weight=2)

        # Start and Stop Buttons
        self.start_button = tk.Button(frame, text="Start Session", command=self.start_stream)
        self.start_button.grid(row=13, column=0, sticky="ew", pady=(0, 5))
        self.dynamic_buttons.append(self.start_button)

        self.stop_button = tk.Button(frame, text="Stop and Save", command=self.stop_stream, state="disabled")
        self.stop_button.grid(row=14, column=0, sticky="ew")
        self.dynamic_buttons.append(self.stop_button)

        # Initial font scaling and bind resize
        update_fonts()
        self.root.bind('<Configure>', lambda e: update_fonts())
        self.update_fonts = update_fonts  # store for later use

    def update_cue(self, symbol, title):
        self.cue_win.title(title)
        self.cue_label.config(text=symbol)
        self.cue_win.update()
        self.update_fonts()

    def update_label(self, label_widget, val):
        label_widget.config(text=f"{int(float(val))} s")

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

        self.recording_start_time = time.time()
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

        self.update_cue("+", "Baseline")
        self.marker_outlet.push_sample(['Baseline'])
        now_lsl = self.eeg_inlet.pull_sample()[1]
        rel_time_ms = (now_lsl - self.start_timestamp_lsl) * 1000  # Convert to milliseconds
        self.markers.append(('Baseline', now_lsl, rel_time_ms))
        self.log(f"Baseline cue ({self.baseline_duration.get()} s)")
        self.root.after(self.baseline_duration.get() * 1000, self.start_imagery)

    def start_imagery(self):
        self.update_cue("•", "Imagery")
        self.marker_outlet.push_sample(['Imagery'])
        now_lsl = self.eeg_inlet.pull_sample()[1]
        rel_time_ms = (now_lsl - self.start_timestamp_lsl) * 1000
        self.markers.append(('Imagery', now_lsl, rel_time_ms))
        self.log(f"Imagery cue ({self.imagery_duration.get()} s)")
        self.root.after(self.imagery_duration.get() * 1000, self.trial_loop)

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
        marker_labels = [m[0] for m in self.markers]
        marker_abs_times = [m[1] for m in self.markers]
        marker_rel_times = [m[2] for m in self.markers]

        info = mne.create_info(ch_names=[self.eeg_channels[i] for i in self.eeg_indices],
                               sfreq=self.sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_array, info)
        raw._times = times_array - times_array[0]

        annotations = mne.Annotations(onset=marker_rel_times,
                                      duration=[0] * len(marker_rel_times),
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

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGMarkerGUI(root)
    root.mainloop()