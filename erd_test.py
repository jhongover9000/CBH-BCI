'''
Complete working example with all fixes applied
This should detect ERD properly with your virtual data
'''

import numpy as np
import time
from datetime import datetime


def run_fixed_erd_system():
    """Run ERD detection with all fixes applied"""
    
    print("Starting Fixed ERD Detection System")
    print("=" * 50)
    
    # Import modules
    try:
        # Try enhanced emulator first
        from receivers.virtual_receiver import Emulator
        print("Using enhanced virtual receiver")
    except ImportError:
        # Fall back to original with patches
        from receivers.virtual_receiver import Emulator
        from erd_quick_fix import patch_virtual_receiver
        print("Using original virtual receiver with patches")
    
    from erd_detection_gui import AdaptiveERDDetector
    
    # Create and initialize emulator
    print("\n1. Initializing virtual EEG stream...")
    emulator = Emulator(fileName="MIT33")
    
    # Apply patch if using original emulator
    if hasattr(emulator, 'auto_scale'):
        emulator.auto_scale = True
        emulator.verbose = True
    else:
        # Patch the original emulator
        emulator = patch_virtual_receiver(emulator)
    
    # Initialize connection
    fs, ch_names, n_channels, _ = emulator.initialize_connection()
    print(f"✓ Connected: {n_channels} channels at {fs} Hz")
    
    # Create detector
    print("\n2. Setting up ERD detector...")
    detector = AdaptiveERDDetector(sampling_freq=fs)
    
    # Apply detector fixes
    from erd_quick_fix import fix_erd_detector
    detector = fix_erd_detector(detector)
    
    # Find and set motor channels
    motor_channels = ['C3', 'C4', 'Cz']
    selected_indices = []
    
    for ch in motor_channels:
        if ch in ch_names:
            idx = ch_names.index(ch)
            selected_indices.append(idx)
            print(f"✓ Found channel {ch} at index {idx}")
    
    if not selected_indices:
        # Try alternative channel names
        alt_names = ['EEG C3', 'EEG C4', 'EEG Cz', 'C3-REF', 'C4-REF', 'Cz-REF']
        for ch in ch_names:
            for alt in alt_names:
                if alt in ch and len(selected_indices) < 3:
                    idx = ch_names.index(ch)
                    selected_indices.append(idx)
                    print(f"✓ Found alternative channel {ch} at index {idx}")
    
    if not selected_indices:
        print("⚠️  No motor channels found, using first 3 channels")
        selected_indices = [0, 1, 2]
    
    # Configure detector
    detector.set_channels(ch_names, selected_indices)
    detector.update_parameters(
        band='mu',
        threshold=20.0,
        adaptation_method='hybrid'
    )
    
    # Lower the minimum power threshold for better detection
    detector.min_valid_power = 1e-9
    
    print("\n3. Processing data...")
    print("   Collecting baseline (2 seconds)...")
    print("\n" + "-" * 50)
    print("Time  | ERD Values              | Status")
    print("-" * 50)
    
    # Process data
    start_time = time.time()
    sample_count = 0
    detection_count = 0
    baseline_set = False
    
    # Store some data for diagnostics
    first_chunk = None
    erd_history = []
    
    for i in range(500):  # Process up to 10 seconds
        # Get data
        data = emulator.get_data()
        
        if data is None or data.shape[1] == 0:
            # Reset to beginning
            emulator.current_index = 0
            continue
        
        # Store first chunk for diagnostics
        if first_chunk is None:
            first_chunk = data.copy()
        
        sample_count += data.shape[1]
        
        # Detect ERD
        detected, erd_values = detector.detect_erd(data)
        
        # Check if baseline was just set
        if not baseline_set and detector.is_baseline_set:
            baseline_set = True
            print(f"\n✓ Baseline set at {sample_count/fs:.1f}s")
            
            # Run diagnostics
            from erd_quick_fix import diagnose_erd_issue
            diagnose_erd_issue(detector, first_chunk)
            
            print("\nContinuing ERD detection...")
            print("-" * 50)
        
        # Display results every 0.5 seconds
        if i % 25 == 0 and erd_values:
            time_str = f"{sample_count/fs:5.1f}s"
            
            # Format ERD values
            erd_str = ""
            for ch, erd in list(erd_values.items())[:3]:
                erd_str += f"{ch}:{erd:6.1f}% "
            
            # Detection status
            status = "DETECTED!" if detected else "--------"
            
            print(f"{time_str} | {erd_str:23s} | {status}")
            
            # Store for analysis
            avg_erd = np.mean(list(erd_values.values()))
            erd_history.append(avg_erd)
            
            if detected:
                detection_count += 1
                emulator.use_classification(1)
        
        # Stop after 10 seconds
        if sample_count > 10 * fs:
            break
    
    # Summary
    duration = time.time() - start_time
    print("\n" + "=" * 50)
    print("Session Summary:")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Samples processed: {sample_count}")
    print(f"  ERD detections: {detection_count}")
    
    if erd_history:
        print(f"  ERD range: {min(erd_history):.1f}% to {max(erd_history):.1f}%")
        print(f"  ERD mean: {np.mean(erd_history):.1f}%")
        print(f"  ERD std: {np.std(erd_history):.1f}%")
        
        if max(erd_history) > 10:
            print("\n✅ SUCCESS: ERD detection is working!")
        else:
            print("\n⚠️  WARNING: ERD values seem low")
            print("   Try adjusting the threshold or checking the data")
    else:
        print("\n❌ ERROR: No ERD values were calculated")
    
    # Cleanup
    emulator.disconnect()
    
    return erd_history


def simple_diagnostic():
    """Run a simple diagnostic to check data"""
    print("\nRunning Simple Data Diagnostic")
    print("=" * 50)
    
    from receivers.virtual_receiver import Emulator
    import matplotlib.pyplot as plt
    from scipy import signal
    
    # Load data
    emulator = Emulator(fileName="MIT33")
    fs, ch_names, _, _ = emulator.initialize_connection()
    
    # Read 5 seconds
    print("Reading 5 seconds of data...")
    data_chunks = []
    for i in range(250):  # 5 seconds at 50 Hz chunks
        chunk = emulator.get_data()
        if chunk is not None:
            data_chunks.append(chunk)
    
    data = np.hstack(data_chunks)
    print(f"Data shape: {data.shape}")
    
    # Find C3 channel
    c3_idx = None
    for i, ch in enumerate(ch_names):
        if 'C3' in ch:
            c3_idx = i
            break
    
    if c3_idx is None:
        c3_idx = 0
        print(f"C3 not found, using channel {ch_names[0]}")
    
    # Analyze
    ch_data = data[c3_idx, :]
    print(f"\nChannel {ch_names[c3_idx]} statistics:")
    print(f"  Mean: {np.mean(ch_data):.2f}")
    print(f"  Std:  {np.std(ch_data):.2f}")
    print(f"  Min:  {np.min(ch_data):.2f}")
    print(f"  Max:  {np.max(ch_data):.2f}")
    
    # Check if scaling needed
    if np.std(ch_data) > 1000:
        print("\n⚠️  Data needs scaling down (probably raw ADC values)")
        ch_data = ch_data * 0.1
    elif np.std(ch_data) < 0.1:
        print("\n⚠️  Data needs scaling up (probably in volts)")
        ch_data = ch_data * 1e6
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    time_vec = np.arange(len(ch_data)) / fs
    ax1.plot(time_vec, ch_data)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'EEG Data - {ch_names[c3_idx]}')
    ax1.grid(True, alpha=0.3)
    
    # Power spectrum
    freqs, psd = signal.welch(ch_data - np.mean(ch_data), fs, nperseg=min(len(ch_data), fs))
    ax2.semilogy(freqs, psd)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Power Spectrum')
    ax2.set_xlim([0, 50])
    ax2.axvspan(8, 12, alpha=0.2, color='red', label='Mu band')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Check mu band power
    mu_mask = (freqs >= 8) & (freqs <= 12)
    mu_power_percent = (np.sum(psd[mu_mask]) / np.sum(psd)) * 100
    print(f"\nMu band (8-12 Hz) contains {mu_power_percent:.1f}% of total power")
    
    if mu_power_percent < 5:
        print("⚠️  Low mu band power - ERD detection might be difficult")
    else:
        print("✅ Good mu band power for ERD detection")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--diagnose':
        # Run diagnostic
        simple_diagnostic()
    else:
        # Run fixed system
        erd_history = run_fixed_erd_system()
        
        # Optional: plot results
        if erd_history:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(erd_history, 'b-', linewidth=2)
                plt.axhline(y=20, color='r', linestyle='--', label='Threshold')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.xlabel('Time (samples)')
                plt.ylabel('ERD (%)')
                plt.title('ERD Detection Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            except ImportError:
                pass