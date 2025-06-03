import socket
import threading
import time
import numpy as np
import argparse
import random
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime
import os

class TCPServer:
    def __init__(self, host='127.0.0.1', tcp_port=65507, buffer_size=1024):
        self.host = host
        self.tcp_port = tcp_port
        self.buffer_size = buffer_size
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_conn = None
        self.client_address = None
        self.running = True

    def initialize_connection(self):
        """Start the TCP server and wait for a single client to connect."""
        print(f"Starting TCP server on {self.host}:{self.tcp_port}")

        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.tcp_server_socket.bind((self.host, self.tcp_port))
            self.tcp_server_socket.listen(1)
            print(f"[LISTENING] TCP Server is listening on {self.host}:{self.tcp_port}")
            
            self.client_conn, self.client_address = self.tcp_server_socket.accept()
            print(f"[CONNECTED] Client connected from {self.client_address}")

            # Start listening for TCP messages in a thread
            # threading.Thread(target=self.listen_to_client_tcp, daemon=True).start()

        except socket.error as e:
            print(f"[ERROR] Server initialization failed: {e}")
            self.disconnect()

    def listen_to_client_tcp(self):
        """Continuously receive messages from the client over TCP."""
        try:
            while self.running and self.client_conn:
                data = self.client_conn.recv(self.buffer_size)
                if not data:
                    print(f"[DISCONNECTED] Client {self.client_address} disconnected (TCP).")
                    break

                message = data.decode('utf-8').strip()
                print(f"[RECEIVED TCP] From {self.client_address}: {message}")

                # Handle client connection confirmation
                if message == "CLIENT_CONNECTED":
                    print(f"[INFO] Client confirmed connection")
                    self.send_message_tcp("SERVER_READY")
                    continue

                # Process other messages from client if needed
                print(f"[INFO] Received message from client: {message}")

        except Exception as e:
            print(f"[ERROR] Error in TCP communication: {e}")
        finally:
            self.disconnect()

    def send_message_tcp(self, message):
        """Send a message to the connected client over TCP."""
        if self.client_conn:
            try:
                print(f"[SENDING TCP] To client: {message}")
                self.client_conn.sendall(message.encode('utf-8'))
                return True
            except Exception as e:
                print(f"[ERROR] Failed to send TCP message: {e}")
                return False
        else:
            print(f"[WARNING] No client connected. Cannot send message: {message}")
            return False

    def is_client_connected(self):
        """Check if a client is currently connected."""
        return self.client_conn is not None and self.running

    def use_classification(self, prediction):
        """Hook for external classifier integration."""
        if prediction == 1:
            self.send_message_tcp("TAP")

    def disconnect(self):
        """Cleanly shut down the server and close connections."""
        self.running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
            self.client_conn = None
        try:
            self.tcp_server_socket.close()
        except:
            pass
        print("[SERVER SHUTDOWN]")

    @staticmethod
    def get_local_ip():
        """Utility to get the local IP address of the machine."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

class ServerGUI:
    def __init__(self, server: TCPServer):
        self.server = server
        self.root = tk.Tk()
        self.root.title("Unity Experiment Controller")
        self.root.geometry("500x500")
        self.root.configure(bg="#f0f0f0")

        # --- Style ---
        self.style = ttk.Style()
        
        # Configure button style with better contrast
        self.style.configure("TButton",
                             padding=10,
                             font=('Arial', 12, 'bold'),
                             foreground="white")
        
        # Try to set background color (may not work on all systems)
        try:
            self.style.configure("TButton", background="#4CAF50")
        except:
            pass
            
        self.style.map("TButton",
                         foreground=[('disabled', 'gray'),
                                     ('active', 'white'),
                                     ('pressed', 'white')],
                         background=[('disabled', '#cccccc'),
                                     ('active', '#45a049'),
                                     ('pressed', '#45a049')])

        self.style.configure("TLabel", font=('Arial', 12), background="#f0f0f0", foreground="black")
        self.style.configure("TScale", background="#f0f0f0")
        self.style.configure("TFrame", background="#f0f0f0")

        # Connection status
        self.status_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(self.status_frame, 
                                    text="Status: Waiting for connection...", 
                                    font=('Arial', 12, 'bold'),
                                    bg="#f0f0f0",
                                    fg="#333333")
        self.status_label.pack()
        
        # Update status periodically
        self.update_status()

        # --- Button Controls ---
        self.button_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)

        controls = [
            ("Swap Rig", "rigSwap"),
            ("Freeze Rig", "rigFreeze"),
            ("Unfreeze Rig", "rigUnfreeze"),
            ("Finger Tap", "fingerTap"),
            ("Spawn Coins", "spawnCoins"),
            ("Stop Spawning", "stopSpawning"),
            ("Raise Table", "raiseTable"),
            ("Lower Table", "lowerTable"),
        ]

        row = 0
        col = 0
        for label, command in controls:
            btn = tk.Button(self.button_frame, 
                           text=label, 
                           command=lambda c=command: self.send_command(c),
                           font=('Arial', 10, 'bold'),
                           bg="#4CAF50",
                           fg="white",
                           activebackground="#45a049",
                           activeforeground="white",
                           relief="raised",
                           bd=2,
                           padx=10,
                           pady=5)
            btn.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            
            # Make columns expand evenly
            self.button_frame.grid_columnconfigure(col, weight=1)
            
            col += 1
            if col > 2:
                col = 0
                row += 1

        # --- Variable Sliders ---
        self.slider_frame = ttk.Frame(self.root, padding=10, style="TFrame")
        self.slider_frame.pack(fill=tk.X)

        self.variables = {
            "beltVelocity": tk.DoubleVar(value=0.1),
            "speedIncrement": tk.DoubleVar(value=0.05),
            "spawnInterval": tk.DoubleVar(value=3.0),
        }

        self.slider_values = {}

        slider_row = 0
        for name, var in self.variables.items():
            ttk.Label(self.slider_frame, text=name, style="TLabel").grid(row=slider_row, column=0, sticky="w", padx=10)
            slider = ttk.Scale(self.slider_frame,
                               variable=var,
                               from_=0,
                               to=5,
                               orient="horizontal",
                               length=200,
                               command=lambda e, n=name: self.update_slider_value(n, e),
                               style="TScale")
            slider.grid(row=slider_row, column=1, padx=5, pady=5, sticky="ew")
            self.slider_values[name] = tk.StringVar()
            self.slider_values[name].set(f"{var.get():.2f}")
            value_label = ttk.Label(self.slider_frame, textvariable=self.slider_values[name], style="TLabel")
            value_label.grid(row=slider_row, column=2, padx=5, pady=5, sticky="w")
            
            send_btn = tk.Button(self.slider_frame,
                                text="Set",
                                command=lambda n=name, v=var: self.send_variable(n, v.get()),
                                font=('Arial', 9, 'bold'),
                                bg="#9C27B0",
                                fg="white",
                                activebackground="#7B1FA2",
                                activeforeground="white",
                                relief="raised",
                                bd=2,
                                padx=8,
                                pady=3)
            send_btn.grid(row=slider_row, column=3, padx=5, pady=5, sticky="w")
            slider_row += 1

        # --- Test Buttons ---
        self.test_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.test_frame.pack(fill=tk.X, padx=10, pady=5)
        
        test_tap_btn = tk.Button(self.test_frame, 
                                text="Test TAP", 
                                command=lambda: self.send_command("TAP"),
                                font=('Arial', 10, 'bold'),
                                bg="#FF9800",
                                fg="white",
                                activebackground="#F57C00",
                                activeforeground="white",
                                relief="raised",
                                bd=2,
                                padx=15,
                                pady=5)
        test_tap_btn.pack(side=tk.LEFT, padx=5)
        
        test_conn_btn = tk.Button(self.test_frame, 
                                 text="Test Connection", 
                                 command=self.test_connection,
                                 font=('Arial', 10, 'bold'),
                                 bg="#2196F3",
                                 fg="white",
                                 activebackground="#1976D2",
                                 activeforeground="white",
                                 relief="raised",
                                 bd=2,
                                 padx=15,
                                 pady=5)
        test_conn_btn.pack(side=tk.LEFT, padx=5)

        # --- Quit ---
        self.quit_button = tk.Button(self.root, 
                                    text="Quit", 
                                    command=self.disconnect,
                                    font=('Arial', 12, 'bold'),
                                    bg="#f44336",
                                    fg="white",
                                    activebackground="#d32f2f",
                                    activeforeground="white",
                                    relief="raised",
                                    bd=2,
                                    padx=20,
                                    pady=8)
        self.quit_button.pack(pady=10)

    def update_status(self):
        """Update connection status display."""
        if self.server.is_client_connected():
            self.status_label.config(text="Status: Connected to Unity client", 
                                   fg="#4CAF50")  # Green for connected
        else:
            self.status_label.config(text="Status: Waiting for connection...", 
                                   fg="#FF5722")  # Red/orange for disconnected
        
        # Schedule next update
        self.root.after(1000, self.update_status)

    def test_connection(self):
        """Send a test message to verify connection."""
        if self.server.send_message_tcp("TEST_MESSAGE"):
            messagebox.showinfo("Test", "Test message sent successfully!")
        else:
            messagebox.showerror("Test", "Failed to send test message. Check connection.")

    def send_command(self, command):
        """Send a command to the Unity client."""
        if not self.server.is_client_connected():
            messagebox.showwarning("Warning", "No client connected!")
            return
            
        success = self.server.send_message_tcp(command)
        if not success:
            messagebox.showerror("Error", f"Failed to send command: {command}")

    def send_variable(self, name, value):
        """Send a variable setting to the Unity client."""
        if not self.server.is_client_connected():
            messagebox.showwarning("Warning", "No client connected!")
            return
            
        message = f"{name}:{value:.2f}"
        success = self.server.send_message_tcp(message)
        if not success:
            messagebox.showerror("Error", f"Failed to send variable: {message}")

    def update_slider_value(self, name, event):
        """Update the displayed slider value."""
        value = self.variables[name].get()
        self.slider_values[name].set(f"{value:.2f}")

    def disconnect(self):
        """Disconnect and close the application."""
        self.server.disconnect()
        self.root.destroy()

class BCIGUI:
    def __init__(self, server, model, X_all, y, subject_ids):
        self.server = server
        self.model = model
        self.X_all = X_all
        self.y = y
        self.subject_ids = subject_ids

        self.root = tk.Tk()
        self.root.title("CBH BCI Demo Control Panel")
        self.root.geometry("600x800")
        self.root.configure(bg="#f0f0f0")

        # --- Style ---
        self.style = ttk.Style()
        self.style.configure("TButton",
                             padding=10,
                             font=('Arial', 12),
                             background="#4CAF50",
                             foreground="white")
        self.style.map("TButton",
                         foreground=[('disabled', 'gray'),
                                     ('active', 'white')],
                         background=[('disabled', 'gray'),
                                     ('active', '#388E3C')],
                         relief=[('pressed', 'sunken'),
                                 ('!pressed', 'raised')])
        self.style.configure("TLabel", font=('Arial', 12), background="#f0f0f0")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabelframe", background="#f0f0f0")

        # --- Control Buttons ---
        self.control_frame = ttk.Frame(self.root, padding=10, style="TFrame")
        self.control_frame.pack(pady=10, fill=tk.X)

        buttons = [
            ("Embodiment Start", "embodimentStart"),
            ("Tutorial Start", "tutorialStart"),
            ("Training Start", "trainingStart"),
            ("Next Block", "nextBlock"),
            ("Swap Rig", "rigSwap"),
            ("Freeze Rig", "rigFreeze"),
            ("Unfreeze Rig", "rigUnfreeze"),
            ("Align Hand", "alignHand"),
            ("Finger Tap", "fingerTap"),
            ("Spawn Coins", "spawnCoins"),
            ("Stop Spawning", "stopSpawning"),
            ("Raise Table", "raiseTable"),
            ("Lower Table", "lowerTable")
        ]
        button_row = 0
        button_col = 0
        for label, cmd in buttons:
            ttk.Button(self.control_frame,
                      text=label,
                      command=lambda c=cmd: self.server.send_message_tcp(c),  # Changed to TCP
                      width=20,
                      style="TButton").grid(row=button_row, column=button_col, padx=5, pady=5, sticky="ew")
            button_col += 1
            if button_col > 2:
                button_col = 0
                button_row += 1

        # --- Sliders for Variables ---
        self.slider_frame = ttk.Frame(self.root, padding=10, style="TFrame")
        self.slider_frame.pack(pady=10, fill=tk.X)

        self.vars = {
            "beltVelocity": tk.DoubleVar(value=0.1),
            "speedIncrement": tk.DoubleVar(value=0.05),
            "spawnInterval": tk.DoubleVar(value=3.0),
        }
        self.slider_values = {}

        slider_row = 0
        for name, var in self.vars.items():
            ttk.Label(self.slider_frame, text=name, style="TLabel").grid(row=slider_row, column=0, sticky="w", padx=10)
            slider = ttk.Scale(self.slider_frame,
                               variable=var,
                               from_=0,
                               to=5,
                               orient="horizontal",
                               length=200,
                               command=lambda e, n=name: self.update_slider_value(n, e),
                               style="TScale")
            slider.grid(row=slider_row, column=1, padx=5, pady=5, sticky="ew")
            self.slider_values[name] = tk.StringVar()
            self.slider_values[name].set(f"{var.get():.2f}")
            value_label = ttk.Label(self.slider_frame, textvariable=self.slider_values[name], style="TLabel")
            value_label.grid(row=slider_row, column=2, padx=5, pady=5, sticky="w")
            ttk.Button(self.slider_frame,
                       text=f"Set {name}",
                       command=lambda n=name, v=var: self.send_var(n, v.get()),
                       style="TButton").grid(row=slider_row, column=3, padx=5, pady=5, sticky="w")
            slider_row += 1

        # --- Classification Buttons ---
        self.classify_frame = ttk.LabelFrame(self.root, text="Classification", padding=10, style="TLabelframe")
        self.classify_frame.pack(pady=10, fill=tk.X)

        self.motor_imagery_button = tk.Button(self.classify_frame,
                                             text="Motor Imagery",
                                             command=lambda: self.select_label(1),
                                             font=('Arial', 12),
                                             bg="#00B050",
                                             fg="white")
        self.motor_imagery_button.pack(pady=5, fill=tk.X)
        self.rest_button = tk.Button(self.classify_frame,
                                           text="Rest",
                                           command=lambda: self.select_label(0),
                                           font=('Arial', 12),
                                           bg="#3366FF",
                                           fg="white")
        self.rest_button.pack(pady=5, fill=tk.X)

        # --- Result Labels ---
        self.result_frame = ttk.Frame(self.root, padding=10, style="TFrame")
        self.result_frame.pack(pady=10, fill=tk.X)

        self.subject_label = ttk.Label(self.result_frame, text="Subject: ", style="TLabel")
        self.trial_label = ttk.Label(self.result_frame, text="Trial: ", style="TLabel")
        self.true_label = ttk.Label(self.result_frame, text="True Label: ", style="TLabel")
        self.predicted_label = ttk.Label(self.result_frame, text="Predicted Label: ", style="TLabel")

        self.subject_label.pack(fill=tk.X)
        self.trial_label.pack(fill=tk.X)
        self.true_label.pack(fill=tk.X)
        self.predicted_label.pack(fill=tk.X)

    def send_var(self, name, value):
        msg = f"{name}:{value:.2f}"
        print(f"[GUI] Sending TCP: {msg}")
        self.server.send_message_tcp(msg)  # Changed to TCP

    def update_slider_value(self, name, event):
        """Update the displayed slider value."""
        value = self.vars[name].get()
        self.slider_values[name].set(f"{value:.2f}")

    def select_label(self, label):
        trials = np.where(self.y == label)[0]
        if len(trials) == 0:
            messagebox.showerror("Error", "No trials found for the selected label.")
            return

        selected_trial = random.choice(trials)
        selected_subject = self.subject_ids[selected_trial]
        eeg_data = self.X_all[selected_trial]

        # Classify
        prob = self.model.predict(eeg_data[np.newaxis, :, :])
        pred = prob.argmax(axis=-1)[0]
        print(f"Classification: {pred}")

        self.server.use_classification(pred)

        self.subject_label.config(text=f"Subject: {selected_subject}")
        self.trial_label.config(text=f"Trial: {selected_trial}")
        self.true_label.config(text=f"True Label: {label}")
        self.predicted_label.config(text=f"Predicted Label: {pred}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    local_ip = TCPServer.get_local_ip()
    print(f"Your local IP is: {local_ip}")
    print(f"Starting TCP server on localhost:6550")

    server = TCPServer()
    # Start server in a background thread
    threading.Thread(target=server.initialize_connection, daemon=True).start()

    # Run GUI in main thread
    gui = ServerGUI(server)
    gui.root.mainloop()

    # Clean shutdown
    server.disconnect()