'''
Simple ERD Detection Demo
Minimal example showing how to use the ERD detection system

Usage:
    python simple_erd_demo.py          # Uses virtual mode by default
    python simple_erd_demo.py --real   # Uses real hardware
'''

import numpy as np
import time
import argparse
from datetime import datetime


def run_erd_demo(mode='virtual', duration=30):
    """
    Simple ERD detection demo
    
    Args:
        mode: 'real' or 'virtual'
        duration: How long to run (seconds)
    """
    
    # Import required modules
    if mode == 'virtual':
        from receivers.virtual_receiver import Emulator as Receiver
        receiver_kwargs = {'fileName': 'MIT33'}
    else:
        from receivers.livestream_receiver import LivestreamReceiver as Receiver
        receiver_kwargs = {
            'address': '169.254.1.147',
            'port': 51244,
            'broadcast': True
        }
    
    from erd_detection_gui import AdaptiveERDDetector
    
    print(f"Starting ERD Detection Demo ({mode} mode)")
    print("=" * 50)
    
    # Step 1: Create receiver
    print("1. Creating receiver...")
    receiver = Receiver(**receiver_kwargs)
    
    # Step 2: Connect
    print("2. Connecting to EEG stream...")
    try:
        fs, ch_names, n_channels, _ = receiver.initialize_connection()
        print(f"   ✓ Connected: {n_channels} channels at {fs} Hz")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return
    
    # Step 3: Create ERD detector
    print("3. Setting up ERD detector...")
    detector = AdaptiveERDDetector(sampling_freq=fs)
    
    # Find motor cortex channels
    motor_channels = ['C3']
    selected_indices = []
    
    for ch in motor_channels:
        if ch in ch_names:
            idx = ch_names.index(ch)
            selected_indices.append(idx)
            print(f"   ✓ Using channel {ch} (index {idx})")
    
    if not selected_indices:
        print("   ✗ No motor channels found!")
        receiver.disconnect()
        return
    
    # Configure detector
    detector.set_channels(ch_names, selected_indices)
    detector.update_parameters(
        band='mu',              # 8-12 Hz for motor imagery
        threshold=50.0,         # 20% ERD threshold
        adaptation_method='hybrid'  # Adaptive baseline
    )
    
    # Step 4: Collect baseline
    print("\n4. Collecting baseline...")
    print("   Please remain relaxed with eyes open")
    
    # Step 5: Start detection
    print(f"\n5. Starting ERD detection for {duration} seconds...")
    print("   Perform motor imagery when ready")
    print("\n" + "-" * 50)
    
    start_time = time.time()
    sample_count = 0
    detection_count = 0
    last_print_time = start_time
    
    try:
        while (time.time() - start_time) < duration:
            # Get EEG data
            data = receiver.get_data()
            
            # Handle end of file for virtual mode
            if mode == 'virtual' and (data is None or data.shape[1] == 0):
                print("\n   End of recording reached, restarting...")
                receiver.current_index = 0
                continue
            
            if data is not None and data.shape[1] > 0:
                sample_count += data.shape[1]
                
                # Detect ERD
                detected, erd_values = detector.detect_erd(data)
                
                # Print status every second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    # Create status line
                    time_str = f"[{int(current_time - start_time):3d}s]"
                    
                    # ERD values
                    erd_str = ""
                    for ch, erd in erd_values.items():
                        erd_str += f" {ch}:{erd:5.1f}%"
                    
                    # Detection status
                    if detected:
                        status_str = " [DETECTED!]"
                        detection_count += 1
                        receiver.use_classification(1)
                    else:
                        status_str = " [--------]"
                    
                    # Rest indicator
                    rest_str = " (Rest)" if detector.is_resting else ""
                    
                    print(f"{time_str}{erd_str}{status_str}{rest_str}")
                    last_print_time = current_time
            
            # Small delay for virtual mode
            if mode == 'virtual':
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Step 6: Summary
    duration_actual = time.time() - start_time
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print(f"  Mode: {mode}")
    print(f"  Duration: {duration_actual:.1f} seconds")
    print(f"  Samples processed: {sample_count}")
    print(f"  Sampling rate: {sample_count/duration_actual:.1f} Hz")
    print(f"  ERD detections: {detection_count}")
    print(f"  Detection rate: {detection_count/duration_actual*60:.1f} per minute")
    print(f"  Baseline method: {detector.adaptation_method}")
    
    # Cleanup
    receiver.disconnect()
    print("\nDemo completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Simple ERD Detection Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python simple_erd_demo.py                # Run with virtual EEG data
  python simple_erd_demo.py --real         # Run with real hardware
  python simple_erd_demo.py --duration 60  # Run for 60 seconds
        '''
    )
    
    parser.add_argument('--real', action='store_true',
                       help='Use real hardware instead of virtual data')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    mode = 'real' if args.real else 'virtual'
    run_erd_demo(mode=mode, duration=args.duration)


if __name__ == "__main__":
    main()