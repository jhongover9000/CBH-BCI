import socket
import threading
import cv2
import queue
import time
import sys
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    SCREENINFO_AVAILABLE = False
    print("screeninfo not available - using manual monitor positioning")

video_path = "./mi_assessment/resources/finger_tap_ex_1080p_no_delay.mp4"

class TCPVideoController:
    def __init__(self, video_path, server_host='127.0.0.1', server_port=65507, monitor_index=1, reconnect_delay=5):
        self.video_path = video_path
        self.server_host = server_host
        self.server_port = server_port
        self.monitor_index = monitor_index
        self.reconnect_delay = reconnect_delay
        self.playing = False
        self.command_queue = queue.Queue()
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.frame_delay = 1.0 / self.fps
        self.secondary_monitor = None
        self.client_socket = None
        self.connected = False
        self.running = True
        self.instruction_text = "Try to imagine your finger tapping to make the finger on the screen tap."

        # Get monitor information
        self.setup_monitors()
        
        # Initialize video capture
        self.setup_video()
        
        # Start TCP client thread
        self.client_thread = threading.Thread(target=self.tcp_client, daemon=True)
        self.client_thread.start()
        
        print(f"TCP Video Controller started")
        print(f"Video: {video_path}")
        print(f"Connecting to TCP server at {server_host}:{server_port}")
        if self.secondary_monitor:
            print(f"Display target: Monitor {self.monitor_index} at {self.secondary_monitor['x']}x{self.secondary_monitor['y']} ({self.secondary_monitor['width']}x{self.secondary_monitor['height']})")
        print("Waiting for TCP messages to trigger video playback")
        print("Press 'q' to quit")
    
    def setup_monitors(self):
        """Setup monitor information for secondary display"""
        if SCREENINFO_AVAILABLE:
            try:
                monitors = get_monitors()
                print(f"Found {len(monitors)} monitor(s):")
                for i, monitor in enumerate(monitors):
                    print(f"  Monitor {i}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
                
                if len(monitors) > self.monitor_index:
                    monitor = monitors[self.monitor_index]
                    self.secondary_monitor = {
                        'x': monitor.x,
                        'y': monitor.y,
                        'width': monitor.width,
                        'height': monitor.height
                    }
                else:
                    print(f"Monitor {self.monitor_index} not found, using primary monitor")
                    self.secondary_monitor = None
            except Exception as e:
                print(f"Error getting monitor info: {e}")
                self.secondary_monitor = None
        else:
            # Manual configuration - common setups
            # You can modify these values based on your setup
            monitor_configs = {
                1: {'x': 1920, 'y': 0, 'width': 1920, 'height': 1080},  # Common secondary monitor
                2: {'x': 3840, 'y': 0, 'width': 1920, 'height': 1080},  # Third monitor
            }
            
            if self.monitor_index in monitor_configs:
                self.secondary_monitor = monitor_configs[self.monitor_index]
                print(f"Using manual configuration for monitor {self.monitor_index}")
            else:
                print(f"No configuration for monitor {self.monitor_index}, using primary monitor")
                self.secondary_monitor = None
    
    def setup_video(self):
        """Initialize video capture and get video properties"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default fallback
        self.frame_delay = 1.0 / self.fps
        
        print(f"Video loaded: {self.total_frames} frames at {self.fps} FPS")
    
    def tcp_client(self):
        """TCP client that connects to server and handles reconnections"""
        while self.running:
            try:
                if not self.connected:
                    self.connect_to_server()
                
                if self.connected:
                    self.listen_for_messages()
                    
            except Exception as e:
                print(f"TCP client error: {e}")
                self.disconnect()
                
            if self.running and not self.connected:
                print(f"Reconnecting in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)
    
    def connect_to_server(self):
        """Establish connection to TCP server"""
        try:
            print(f"Attempting to connect to {self.server_host}:{self.server_port}")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(10)  # 10 second timeout for connection
            self.client_socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"Connected to TCP server at {self.server_host}:{self.server_port}")
            
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            self.disconnect()
    
    def listen_for_messages(self):
        """Listen for messages from the TCP server"""
        try:
            while self.connected and self.running:
                # Set socket to non-blocking for periodic checks
                self.client_socket.settimeout(1.0)
                
                try:
                    data = self.client_socket.recv(1024)
                    if not data:
                        print("Server closed connection")
                        break
                    
                    message = data.decode('utf-8', errors='ignore').strip()
                    if message:
                        print(f"Received TCP message: {message}")
                        
                        # If not currently playing, trigger playback
                        if not self.playing:
                            self.command_queue.put("PLAY")
                            print("Video playback triggered")
                        else:
                            print("Video already playing - message ignored")
                            
                except socket.timeout:
                    # Timeout is expected for non-blocking socket
                    continue
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
                    
        except Exception as e:
            print(f"Listen error: {e}")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Disconnect from server"""
        self.connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
            print("Disconnected from server")
    
    def setup_fullscreen_window(self):
        """Setup fullscreen window on secondary monitor"""
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        
        if self.secondary_monitor:
            # Position window on secondary monitor
            cv2.moveWindow('Video', self.secondary_monitor['x'], self.secondary_monitor['y'])
            # Resize to monitor size before going fullscreen
            cv2.resizeWindow('Video', self.secondary_monitor['width'], self.secondary_monitor['height'])
        
        # Make fullscreen
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    def add_overlay_to_frame(self, frame, instruction_text="", show_countdown=False, countdown_value=0):
        """Add text overlay to video frame"""
        height, width = frame.shape[:2]
        
        # Create a copy to avoid modifying original
        overlay_frame = frame.copy()
        
        # Semi-transparent background for text
        overlay = overlay_frame.copy()
        
        if instruction_text:
            # Main instruction text
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 2.5
            thickness = 3
            color = (255, 255, 255)  # White text
            
            # Get text size for centering
            text_size = cv2.getTextSize(instruction_text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 8
            
            # Add background rectangle
            padding = 20
            cv2.rectangle(overlay, 
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (150,150,150), -1)  # Black background
            cv2.rectangle(overlay, 
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (150,150,150), -1)  # Black background
            
            # Add text
            cv2.putText(overlay, instruction_text, (text_x, text_y), 
                    font, font_scale, color, thickness)
        
        if show_countdown:
            # Countdown timer
            countdown_text = str(countdown_value)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4.0
            thickness = 6
            color = (0, 255, 255)  # Yellow
            
            text_size = cv2.getTextSize(countdown_text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2
            
            # Add countdown
            cv2.putText(overlay_frame, countdown_text, (text_x, text_y), 
                    font, font_scale, color, thickness)
        
        # Blend overlay with original frame
        alpha = 0.8  # Transparency
        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
        
        return overlay_frame
    
    def display_first_frame(self):
        """Display the first frame of the video with overlay"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            # Add instruction overlay
            frame_with_overlay = self.add_overlay_to_frame(
                frame, 
                instruction_text=self.instruction_text,
                show_countdown=False
            )
            
            self.setup_fullscreen_window()
            cv2.imshow('Video', frame_with_overlay)

    # Modified play_video method with dynamic overlays:
    def play_video(self):
        """Play video from current position to end with overlays"""
        print("Starting video playback...")
        self.playing = True
        
        frame_count = 0
        start_time = time.time()
        
        while self.playing:
            ret, frame = self.cap.read()
            if not ret:
                print("Video playback completed")
                # time.sleep(2)
                break
            
            # Add overlays based on timing
            current_time = time.time() - start_time
            
            if current_time < 3:  # First 3 seconds
                frame = self.add_overlay_to_frame(
                    frame, 
                    instruction_text=self.instruction_text,
                )
            elif current_time < 10:  # Next 7 seconds
                frame = self.add_overlay_to_frame(
                    frame, 
                    instruction_text=self.instruction_text
                )
            
            cv2.imshow('Video', frame)
            
            # Handle timing for proper playback speed
            frame_count += 1
            expected_time = start_time + (frame_count * self.frame_delay)
            current_time = time.time()
            
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            
            # Check for 'q' key press to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        
        # Reset to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.playing = False
        print("Video reset to first frame")
        return True
    
    def run(self):
        """Main loop"""
        try:
            # Create fullscreen window on secondary monitor
            self.setup_fullscreen_window()
            
            # Display first frame initially
            self.display_first_frame()
            
            while True:
                # Check for commands from TCP thread
                try:
                    command = self.command_queue.get_nowait()
                    if command == "PLAY":
                        if not self.play_video():
                            break  # User pressed 'q'
                        self.display_first_frame()  # Show first frame again
                except queue.Empty:
                    pass
                
                # Check for 'q' key press to quit
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Shutting down...")
        self.running = False
        self.disconnect()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

def main():
    if len(sys.argv) < 2:
        print("Usage: python bci_erd_tester.py <video_file_path> [server_host] [server_port] [monitor_index]")
        print("Example: python bci_erd_tester.py video.mp4 127.0.0.1 65507 1")
        print("  server_host: TCP server hostname/IP (default: 127.0.0.1)")
        print("  server_port: TCP server port (default: 65507)")
        print("  monitor_index: 0=primary, 1=secondary (default), 2=third, etc.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    server_host = '127.0.0.1'
    server_port = 65507
    monitor_index = 1  # Default to secondary monitor
    
    if len(sys.argv) > 2:
        server_host = sys.argv[2]
    
    if len(sys.argv) > 3:
        try:
            server_port = int(sys.argv[3])
        except ValueError:
            print("Error: server_port must be an integer")
            sys.exit(1)
    
    if len(sys.argv) > 4:
        try:
            monitor_index = int(sys.argv[4])
        except ValueError:
            print("Error: monitor_index must be an integer")
            sys.exit(1)
    
    try:
        controller = TCPVideoController(
            video_path, 
            server_host=server_host, 
            server_port=server_port, 
            monitor_index=monitor_index
        )
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()