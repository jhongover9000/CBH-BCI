import socket
import threading

# --- Configuration ---
# Use '0.0.0.0' to listen on all available network interfaces
# Or replace with your computer's specific local IP if needed
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 9999 # Choose a port number (avoid common ones below 1024)
BUFFER_SIZE = 1024 # Size of data buffer for receiving messages

# --- Function to handle each client connection ---
def handle_client(client_socket, client_address):
    print(f"[NEW CONNECTION] {client_address} connected.")
    connected = True
    while connected:
        try:
            # Receive data from the client
            data = client_socket.recv(BUFFER_SIZE)
            if not data:
                # No data means the client disconnected
                print(f"[DISCONNECTED] {client_address} disconnected.")
                break

            message = data.decode('utf-8') # Decode bytes to string
            print(f"[{client_address}] Received: {message}")

            # --- Example: Process received data ---
            # You can add logic here to react to messages from the Quest 3
            if message.lower() == 'hello server':
                response = "Hello Quest 3!"
            elif message.lower() == 'quit':
                response = "Goodbye!"
                connected = False # Signal to close connection after sending response
            else:
                response = f"Server received: {message}"

            # Send a response back to the client
            print(f"[{client_address}] Sending: {response}")
            client_socket.sendall(response.encode('utf-8')) # Encode string to bytes

        except ConnectionResetError:
            print(f"[CONNECTION RESET] {client_address} forcibly closed the connection.")
            connected = False
        except Exception as e:
            print(f"[ERROR] An error occurred with {client_address}: {e}")
            connected = False

    # Close the client socket when the loop ends
    client_socket.close()
    print(f"[CONNECTION CLOSED] {client_address}")

# --- Main Server Logic ---
def start_server():
    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Allow reusing the address (useful for quick restarts)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to the host and port
    try:
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        print(f"[LISTENING] Server is listening on {SERVER_HOST}:{SERVER_PORT}")
    except socket.error as e:
        print(f"[ERROR] Failed to bind server socket: {e}")
        return # Exit if binding fails

    # Start listening for incoming connections (allow up to 5 pending connections)
    server_socket.listen(5)

    print("[WAITING] Waiting for connections...")

    # Main loop to accept new connections
    while True:
        try:
            # Accept a new connection
            client_socket, client_address = server_socket.accept()

            # Create a new thread to handle the client communication
            # This allows the server to handle multiple clients concurrently
            client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
            client_thread.daemon = True # Allows program to exit even if threads are running
            client_thread.start()

        except KeyboardInterrupt:
            print("\n[SHUTTING DOWN] Server is shutting down.")
            break
        except Exception as e:
            print(f"[ERROR] An error occurred while accepting connections: {e}")
            break

    # Close the main server socket
    server_socket.close()
    print("[SERVER CLOSED]")

# --- Find Your Local IP Address ---
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1' # Fallback to localhost
    finally:
        s.close()
    return IP

# --- Start the server ---
if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Your computer's local IP address is likely: {local_ip}")
    print(f"Make sure your Quest 3 connects to this IP on port {SERVER_PORT}")
    print("Ensure both devices are on the same Wi-Fi network.")
    print("You might need to allow Python or port {} through your firewall.".format(SERVER_PORT))
    start_server()