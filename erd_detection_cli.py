'''
Command-line ERD Detection System for BrainVision ActiChamp
No GUI required - suitable for headless operation

Joseph Hong
'''

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from collections import deque
import time
import argparse
from datetime import datetime
from livestream_receiver import LivestreamReceiver


class SimpleERDDetector:
    """Simplified ERD detector for command-line usage"""
    
    def __init__(self, fs=1000, channels=['C3', 'C4', 'Cz'], 
                 band='mu', threshold=20.0, adaptive=True):
        self.fs = fs
        self.target_channels = channels
        self.band = band
        self.threshold = threshold
        self.adaptive = adaptive
        
        # Frequency bands
        self.bands = {
            'mu': (8, 12),
            'beta': (13, 30),
            'alpha': (8, 13)
        }
        
        # Buffers
        self.buffers = {}
        self.baseline = {}
        self.is_baseline_set = False
        
        # Adaptive baseline
        self.alpha = 0.01  # Update rate
        self.rest_buffer = deque(maxlen=100)
        
        # Statistics
        self.detection_count = 0
        self.start_time = time.time()
        
        # Initialize filter
        self._init_filter()
        
    def _init_filter(self):
        """Initialize bandpass filter"""
        nyq = self.fs / 2
        low, high = self.bands[self.band]
        self.b, self.a = butter(4, [low/nyq, high/nyq], btype='band')
        
    def initialize(self, channel_names):
        """Initialize with channel information"""
        self.channel_names = channel_names
        self.channel_indices = []
        
        # Find target channels
        for target in self.target_channels:
            if target in channel_names:
                idx = channel_names.index(target)
                self.channel_indices.append(idx)
                self.buffers[idx] = deque(maxlen=int(2 * self.fs))
                print(f"  Found channel {target} at index {idx}")
            else:
                print(f"  Warning: Channel {target} not found!")
        
        if not self.channel_indices:
            raise ValueError("No target channels found in data!")
            
    def process(self, data):
        """Process new data and detect ERD"""
        # Update buffers
        for idx in self.channel_indices:
            self.buffers[idx].extend(data[idx, :])
        
        # Check if we have enough data
        if not all(len(self.buffers[idx]) >= self.fs for idx in self.channel_indices):
            return None
        
        # Set baseline if needed
        if not self.is_baseline_set:
            if len(self.buffers[self.channel_indices[0]]) >= 2 * self.fs:
                self._set_baseline()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Baseline set successfully")
                return None
        
        # Calculate ERD
        results = {}
        detected = False
        
        for idx in self.channel_indices:
            # Get recent data (0.5s window)
            window = list(self.buffers[idx])[-int(0.5 * self.fs):]
            
            # Calculate power
            filtered = filtfilt(self.b, self.a, window)
            current_power = np.mean(filtered ** 2)
            
            # Calculate ERD
            if self.baseline[idx] > 0:
                erd = ((self.baseline[idx] - current_power) / self.baseline[idx]) * 100
                ch_name = self.channel_names[idx]
                results[ch_name] = erd
                
                if erd > self.threshold:
                    detected = True
                
                # Adaptive baseline update
                if self.adaptive and abs(erd) < 5:  # Rest period
                    self.baseline[idx] = (1 - self.alpha) * self.baseline[idx] + self.alpha * current_power
        
        # Update statistics
        if detected:
            self.detection_count += 1
        
        return {
            'detected': detected,
            'erd_values': results,
            'timestamp': time.time()
        }
    
    def _set_baseline(self):
        """Set initial baseline"""
        for idx in self.channel_indices:
            data = list(self.buffers[idx])[-int(2 * self.fs):]
            filtered = filtfilt(self.b, self.a, data)
            self.baseline[idx] = np.mean(filtered ** 2)
        self.is_baseline_set = True
        
    def print_status(self, result):
        """Print current status"""
        if result is None:
            return
            
        # Clear line and print
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end='')
        
        # ERD values
        for ch, erd in result['erd_values'].items():
            marker = "*" if erd > self.threshold else " "
            print(f"{marker}{ch}:{erd:5.1f}% ", end='')
        
        # Detection status
        if result['detected']:
            print(" [DETECTED!]", end='')
        else:
            print(" [--------]", end='')
        
        # Statistics
        duration = time.time() - self.start_time
        rate = self.detection_count / duration * 60
        print(f" | Total: {self.detection_count} ({rate:.1f}/min)", end='', flush=True)


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description='ERD Detection System')
    parser.add_argument('--address', default='169.254.1.147', help='BrainVision IP address')
    parser.add_argument('--port', type=int, default=51244, help='Port number')
    parser.add_argument('--channels', nargs='+', default=['C3', 'C4', 'Cz'], 
                        help='Channels to monitor')
    parser.add_argument('--band', choices=['mu', 'beta', 'alpha'], default='mu',
                        help='Frequency band')
    parser.add_argument('--threshold', type=float, default=20.0, 
                        help='ERD threshold percentage')
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help='Use adaptive baseline')
    parser.add_argument('--broadcast', action='store_true', default=True,
                        help='Enable TCP broadcasting')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ERD Detection System - Command Line Interface")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Address: {args.address}:{args.port}")
    print(f"  Channels: {', '.join(args.channels)}")
    print(f"  Band: {args.band} band")
    print(f"  Threshold: {args.threshold}%")
    print(f"  Adaptive: {'Yes' if args.adaptive else 'No'}")
    print(f"  Broadcasting: {'Yes' if args.broadcast else 'No'}")
    print("=" * 60)
    
    # Create receiver
    receiver = LivestreamReceiver(
        address=args.address,
        port=args.port,
        broadcast=args.broadcast
    )
    
    # Connect
    print("\nConnecting to BrainVision ActiChamp...")
    try:
        fs, ch_names, n_channels, _ = receiver.initialize_connection()
        print(f"Connected! Sampling rate: {fs} Hz, Channels: {n_channels}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    # Create detector
    detector = SimpleERDDetector(
        fs=fs,
        channels=args.channels,
        band=args.band,
        threshold=args.threshold,
        adaptive=args.adaptive
    )
    
    # Initialize detector
    print("\nInitializing detector...")
    detector.initialize(ch_names)
    
    print("\nCollecting baseline data (2 seconds)...")
    
    # Main loop
    print("\nStarting ERD detection. Press Ctrl+C to stop.")
    print("-" * 60)
    
    try:
        while True:
            # Get data
            data = receiver.get_data()
            
            if data is not None:
                # Process
                result = detector.process(data)
                
                # Display
                if result:
                    detector.print_status(result)
                    
                    # Send command if detected
                    if result['detected']:
                        receiver.use_classification(1)
                        
                        if args.verbose:
                            print(f"\n  -> Sent TAP command via TCP")
                            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        receiver.disconnect()
        
        # Final statistics
        duration = time.time() - detector.start_time
        print("\n" + "=" * 60)
        print("Session Summary:")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Detections: {detector.detection_count}")
        print(f"  Detection rate: {detector.detection_count / duration * 60:.1f}/min")
        print("=" * 60)


if __name__ == "__main__":
    main()