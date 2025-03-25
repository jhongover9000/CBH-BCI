import socket
import threading

class UDPServer:
    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = port
        # Create a UDP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_addr = None
        self.running = True  # Control flag for stopping threads

    def initialize_connection(self):
        """Initialize and start the UDP server."""
        print("UDP Server Starting...")
        self.server_socket.bind((self.host, self.port))
        print(f"UDP Server Listening on {self.host}:{self.port}")
        
        # Wait for the first incoming packet to capture the client's address.
        # data, addr = self.server_socket.recvfrom(1024)
        # print(f"Initial packet received from {addr}")
        # self.client_addr = addr

    def send_tap_signal(self):
        """Broadcast 'TAP' to all clients on the local network (UDP broadcast)."""
        try:
            print("Broadcasting TAP signal!")
            # Enable broadcasting mode on the socket
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # Broadcast to all clients on port 5005
            self.server_socket.sendto("TAP".encode(), ('255.255.255.255', self.port))
        except Exception as e:
            print("Error sending TAP:", e)

    def use_classification(self, prediction):
        if prediction == 1:
            self.send_tap_signal()

    def disconnect(self):
        """Safely shuts down the server."""
        self.running = False
        self.server_socket.close()
        print("Server shutdown.")

# Example usage:
if __name__ == "__main__":
    server = UDPServer()
    
    # Running initialize_connection in a separate thread if you need to keep the main thread free.
    init_thread = threading.Thread(target=server.initialize_connection)
    init_thread.start()
    
    # Here you might have other code that uses the server. For demonstration:
    try:
        while server.running:
            # Placeholder for classification that triggers the TAP signal.
            # For example, replace with your prediction logic:
            prediction = 1  # Assume a condition that requires a TAP
            server.use_classification(prediction)
            # Sleep or wait as needed...
            threading.Event().wait(5)
    except KeyboardInterrupt:
        server.disconnect()
