import serial
import time

class COMPortSignalSender:
    def __init__(self, port='COM3', baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
    
    def initialize_connection(self):
        """Establish a connection to the COM port."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Connected to {self.port} at {self.baudrate} baud")
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
    
    def send_signal(self, command):
        """Send a signal (command) to the COM port."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(command.encode())
            print(f"Sent command: {command}")
        else:
            print("COM port is not open.")

    def use_classification(self, prediction):
        if prediction == 0:
            pass
        elif prediction == 1:
            self.send_signal("v100")
            time.sleep(1.5)
            self.send_signal("v0")
        else:
            print("Unknown prediction value.")
    
    def disconnect(self):
        """Close the COM port connection."""
        if self.serial_conn:
            self.serial_conn.close()
            print("COM port closed.")
