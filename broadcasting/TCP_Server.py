import socket
import threading

class TCPServer:
    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_conn = None
        self.running = True  # Control flag for stopping threads

    def initialize_connection(self):
        """Initialize and start the TCP server."""
        print(f"TCP Server Starting...")
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"TCP Server Listening on {self.host}:{self.port}")

        self.client_conn, addr = self.server_socket.accept()
        print(f"Connected by {addr}")

    def send_tap_signal(self):
        """Send 'TAP' to Unity client."""
        if self.client_conn:
            try:
                print("Sending TAP signal!")
                self.client_conn.sendall("TAP".encode())
            except Exception as e:
                print("Error sending TAP:", e)

    def use_classification(self, prediction):
        if prediction == 1:
            self.send_tap_signal()

    def disconnect(self):
        """Safely closes the connection and shuts down the server."""
        self.running = False
        if self.client_conn:
            self.client_conn.close()
        self.server_socket.close()
        print("Server shutdown.")