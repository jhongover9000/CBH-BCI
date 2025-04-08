import socket
import threading
import time
import numpy as np
import argparse
import random
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import os

class TCPServer:
    def __init__(self, host='0.0.0.0', tcp_port=9999, udp_port=12345, buffer_size=1024):
        self.host = host
        self.tcp_port = tcp_port
        self.udp_port = udp_port  # Port to send UDP messages
        self.buffer_size = buffer_size
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.udp_server_socket = None  # UDP Socket
        self.client_conn = None
        self.client_address = None
        self.quest_ip = None  # To store the Quest 3 IP
        self.running = True

    def initialize_connection(self):
        """Start the TCP server and wait for a single client to connect."""
        local_ip = TCPServer.get_local_ip()
        print(f"Your local IP is likely: {local_ip}")
        print("Ensure both devices are on the same Wi-Fi and Python/ports 9999 (TCP) and 12345 (UDP) are allowed through firewall.")

        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.tcp_server_socket.bind((self.host, self.tcp_port))
            self.tcp_server_socket.listen(1)
            print(f"[LISTENING] TCP Server is listening on {self.host}:{self.tcp_port}")
            self.client_conn, self.client_address = self.tcp_server_socket.accept()
            print(f"[CONNECTED] Client connected from {self.client_address}")

            # Start listening for TCP messages in a thread
            threading.Thread(target=self.listen_to_client_tcp, daemon=True).start()

        except socket.error as e:
            print(f"[ERROR] Server initialization failed: {e}")
            self.disconnect()

    def create_quest_send_socket(self):
        """Create a UDP socket to send messages to the Quest 3."""
        try:
            self.udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[INFO] UDP send socket created")
        except socket.error as e:
            print(f"[ERROR] Failed to create UDP send socket: {e}")

    def listen_to_client_tcp(self):
        """Continuously receive messages from the client over TCP."""
        try:
            while self.running:
                data = self.client_conn.recv(self.buffer_size)
                if not data:
                    print(f"[DISCONNECTED] Client {self.client_address} disconnected (TCP).")
                    break

                message = data.decode('utf-8').strip()
                print(f"[RECEIVED TCP] From {self.client_address}: {message}")

                # Handle the initial IP address message from the Quest 3
                if message.startswith("IP:"):
                    self.quest_ip = message.split(":")[1]
                    print(f"[INFO] Quest 3 IP address: {self.quest_ip}")
                    self.create_quest_send_socket() #create the udp socket
                    continue  # Important:  Don't try to process as a regular command

                #  Process other TCP messages if needed,  e.g.,  a confirmation from the Quest.

        except Exception as e:
            print(f"[ERROR] Error in TCP communication: {e}")
        finally:
            self.disconnect()

    def send_message_tcp(self, message):
        """Send a message to the connected client over TCP."""
        if self.client_conn:
            try:
                print(f"[ACTION] Sending TCP message: {message}.")
                self.client_conn.sendall(message.encode('utf-8'))
            except Exception as e:
                print(f"[ERROR] Failed to send TCP message: {e}")

    def send_message_udp(self, message):
        """Send a message to the Quest 3 over UDP."""
        if self.quest_ip and self.udp_server_socket:
            try:
                print(f"[ACTION] Sending UDP message to {self.quest_ip}:{self.udp_port}: {message}")
                self.udp_server_socket.sendto(message.encode('utf-8'), (self.quest_ip, self.udp_port))
            except Exception as e:
                print(f"[ERROR] Failed to send UDP message: {e}")
        elif not self.quest_ip:
            print(f"[WARNING] Cannot send UDP message. Quest 3 IP address is unknown.")
        elif not self.udp_server_socket:
            print(f"[WARNING] Cannot send UDP message. UDP socket not initialized.")

    def use_classification(self, prediction):
        """Hook for external classifier integration."""
        if prediction == 1:
            self.send_message_udp("TAP")

    def disconnect(self):
        """Cleanly shut down the server and close connections."""
        self.running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
        try:
            self.tcp_server_socket.close()
        except:
            pass
        if self.udp_server_socket:
            try:
                self.udp_server_socket.close()
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

import tkinter as tk
from tkinter import ttk
import threading

class ServerGUI:
    def __init__(self, server: TCPServer):
        self.server = server
        self.root = tk.Tk()
        self.root.title("Unity Experiment Controller")

        # --- Button Controls ---
        controls = [
            ("Swap Rig", "rigSwap"),
            ("Freeze Rig", "rigFreeze"),
            ("Unfreeze Rig", "rigUnfreeze"),
            ("Finger Tap", "fingerTap"),
            ("Spawn Coins", "spawnCoins"),
            ("Stop Spawning", "stopSpawning"),
            ("Raise Table", "raiseTable"),  # Changed to match Unity
            ("Lower Table", "lowerTable"),  # Changed to match Unity
        ]

        row = 0
        for label, command in controls:
            btn = ttk.Button(self.root, text=label, width=20, command=lambda c=command: self.send_command(c))
            btn.grid(row=row, column=0, padx=10, pady=2, sticky="w")
            row += 1

        # --- Variable Sliders ---
        self.variables = {
            "beltVelocity": tk.DoubleVar(value=0.1),
            "speedIncrement": tk.DoubleVar(value=0.05),
            "spawnInterval": tk.DoubleVar(value=3.0),
        }

        for name, var in self.variables.items():
            ttk.Label(self.root, text=name).grid(row=row, column=0, sticky="w", padx=10)
            slider = ttk.Scale(self.root, variable=var, from_=0, to=5, orient="horizontal", length=200)
            slider.grid(row=row, column=1, padx=5, pady=2)
            send_btn = ttk.Button(self.root, text="Set", command=lambda n=name, v=var: self.send_variable(n, v.get()))
            send_btn.grid(row=row, column=2)
            row += 1

        # --- Quit ---
        quit_btn = ttk.Button(self.root, text="Quit", command=lambda: self.disconnect())
        quit_btn.grid(row=row, column=0, columnspan=3, pady=10)

        # Run GUI in a non-blocking thread.
        # threading.Thread(target=self.root.mainloop, daemon=True).start()

    def send_command(self, command):
        message = f"{command}"
        print(f"[GUI] Sending UDP: {message}") #changed to UDP
        self.server.send_message_udp(message) #changed to UDP

    def send_variable(self, name, value):
        message = f"{name}:{value:.2f}"
        print(f"[GUI] Sending UDP: {message}") #changed to UDP
        self.server.send_message_udp(message) #changed to UDP

    def disconnect(self):
        self.server.disconnect()

class BCIGUI:
    def __init__(self, server, model, X_all, y, subject_ids):
        self.server = server
        self.model = model
        self.X_all = X_all
        self.y = y
        self.subject_ids = subject_ids

        self.root = tk.Tk()
        self.root.title("CBH BCI Demo Control Panel")
        self.root.geometry("500x500")

        # --- Control Buttons ---
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(pady=5)

        buttons = [
            ("Swap Rig", "rigSwap"),
            ("Freeze Rig", "rigFreeze"),
            ("Unfreeze Rig", "rigUnfreeze"),
            ("Finger Tap", "fingerTap"),
            ("Spawn Coins", "spawnCoins"),
            ("Stop Spawning", "stopSpawning"),
        ]
        for label, cmd in buttons:
            tk.Button(control_frame, text=label, command=lambda c=cmd: self.server.send_message(c), width=20).pack(pady=2)

        # --- Sliders for Variables ---
        slider_frame = tk.Frame(self.root, padx=10, pady=10)
        slider_frame.pack()

        self.vars = {
            "beltVelocity": tk.DoubleVar(value=0.1),
            "speedIncrement": tk.DoubleVar(value=0.05),
            "spawnInterval": tk.DoubleVar(value=3.0),
        }

        for name, var in self.vars.items():
            tk.Label(slider_frame, text=name).pack()
            tk.Scale(slider_frame, variable=var, from_=0, to=5, resolution=0.01,
                     orient="horizontal", length=200).pack()
            tk.Button(slider_frame, text=f"Set {name}",
                      command=lambda n=name, v=var: self.send_var(n, v.get())).pack(pady=2)

        # --- Classification Buttons ---
        classify_frame = tk.LabelFrame(self.root, text="Classification", padx=10, pady=10)
        classify_frame.pack(pady=10)

        tk.Button(classify_frame, text="Motor Imagery", width=20,
                  command=lambda: self.select_label(1)).pack(pady=2)
        tk.Button(classify_frame, text="Rest", width=20,
                  command=lambda: self.select_label(0)).pack(pady=2)

        # --- Result Labels ---
        self.subject_label = tk.Label(self.root, text="Subject: ")
        self.trial_label = tk.Label(self.root, text="Trial: ")
        self.true_label = tk.Label(self.root, text="True Label: ")
        self.predicted_label = tk.Label(self.root, text="Predicted Label: ")

        self.subject_label.pack()
        self.trial_label.pack()
        self.true_label.pack()
        self.predicted_label.pack()

    def send_var(self, name, value):
        msg = f"{name}:{value:.2f}"
        print(f"[GUI] Sending: {msg}")
        self.server.send_message(msg)

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
    print(f"Your local IP is likely: {local_ip}")
    print("Ensure both devices are on the same Wi-Fi and Python/ports 9999 (TCP) and 12345 (UDP) are allowed through firewall.")

    server = TCPServer()
    # Start server in a background thread
    threading.Thread(target=server.initialize_connection, daemon=True).start()

    # Run GUI in main thread
    gui = ServerGUI(server)
    gui.root.mainloop()

    # # Keep the main thread alive
    # while True:
    #     time.sleep(1)

    # Optional: keep main thread alive
    try:
        while server.running:
            pass
    except KeyboardInterrupt:
        server.disconnect()
