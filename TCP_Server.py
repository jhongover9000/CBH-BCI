import socket
import threading
from pynput import keyboard  # No admin access needed!

HOST = "127.0.0.1"
PORT = 5005
REQUIRED_COUNT = 5  # Number of "1" messages needed before sending TAP

def handle_client(conn):
    """Handles messages from Unity and counts occurrences."""
    last_message = None
    count = 0

    while True:
        try:

            key = input("Press ENTER to send TAP: ")
            if key == "":
                send_tap_signal(conn)
                print("TAP signal sent!")

        except Exception as e:
            print("Error:", e)
            break

    conn.close()
    print("Connection closed.")

def send_tap_signal(conn):
    """Send 'TAP' to Unity."""
    try:
        print("Sending TAP signal!")
        conn.sendall("TAP".encode())
    except Exception as e:
        print("Error sending TAP:", e)

def listen_for_spacebar(conn):
    """Runs a non-blocking spacebar listener in the background."""
    def on_press(key):
        if key == keyboard.Key.space:
            print("Spacebar pressed! Sending TAP signal.")
            send_tap_signal(conn)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # Starts in a separate thread (non-blocking)

# Start TCP Server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("TCP Server Listening...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # Start listening for spacebar (NON-BLOCKING)
    # threading.Thread(target=listen_for_spacebar, args=(conn,), daemon=True).start()

    # Handle client messages
    handle_client(conn)
