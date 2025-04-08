import socket

HOST = '0.0.0.0'
PORT = 5005

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Client disconnected.")
                    break
                received = data.decode()
                print("Received:", received)
                response = f"Got your message: {received}"
                conn.sendall(response.encode())

if __name__ == '__main__':
    start_server()
