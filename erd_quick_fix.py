'''
Quick fix for ERD detection issues
Add this to your existing code or run as a patch
'''

import numpy as np
from scipy import signal


def fix_erd_detector(detector):
    """
    Apply fixes to existing ERD detector for common issues
    """
    # Override the calculate_band_power method with enhanced version
    original_calculate_band_power = detector.calculate_band_power
    
    def enhanced_calculate_band_power(data, band='mu'):
        """Enhanced band power calculation with debugging"""
        
        # Remove DC offset FIRST
        data = np.array(data)
        data = data - np.mean(data)
        
        # Check if data is too small (likely wrong scaling)
        if np.std(data) < 0.1:
            print(f"Warning: Very low data variance ({np.std(data):.6f}), scaling up by 100")
            data = data * 100
        
        # Check if data is too large (likely raw ADC values)
        elif np.std(data) > 1000:
            print(f"Warning: Very high data variance ({np.std(data):.1f}), scaling down")
            data = data * 0.1
        
        # Call original method with cleaned data
        power = original_calculate_band_power(data, band)
        
        # Ensure minimum power to avoid division issues
        min_power = 1e-6
        if power < min_power:
            print(f"Warning: Power too low ({power:.2e}), setting to minimum")
            power = min_power
            
        return power
    
    # Replace the method
    detector.calculate_band_power = enhanced_calculate_band_power
    
    # Also ensure minimum valid power is reasonable
    detector.min_valid_power = 1e-6
    
    print("ERD detector patched with fixes")
    return detector


def diagnose_erd_issue(detector, data_chunk):
    """
    Diagnose why ERD might be 0%
    """
    print("\nDiagnosing ERD calculation...")
    
    # Check if baseline is set
    if not detector.is_baseline_set:
        print("❌ Baseline not yet set")
        return
    
    # Check baseline values
    print("\nBaseline power values:")
    for idx in detector.selected_indices:
        if idx in detector.baseline_power:
            power = detector.baseline_power[idx]
            ch_name = detector.channel_names[idx] if idx < len(detector.channel_names) else f"Ch{idx}"
            print(f"  {ch_name}: {power:.6e}")
            
            if power < 1e-10:
                print(f"    ⚠️  Baseline power too low!")
            elif power > 1e6:
                print(f"    ⚠️  Baseline power too high!")
    
    # Check current data
    print("\nCurrent data statistics:")
    for i, idx in enumerate(detector.selected_indices):
        if idx < data_chunk.shape[0]:
            ch_data = data_chunk[idx, :]
            ch_name = detector.channel_names[idx] if idx < len(detector.channel_names) else f"Ch{idx}"
            print(f"  {ch_name}:")
            print(f"    Mean: {np.mean(ch_data):.6f}")
            print(f"    Std:  {np.std(ch_data):.6f}")
            print(f"    Range: {np.ptp(ch_data):.6f}")
    
    # Try manual ERD calculation
    print("\nManual ERD calculation test:")
    for idx in detector.selected_indices[:1]:  # Just first channel
        if idx in detector.baseline_power and idx < data_chunk.shape[0]:
            # Get baseline
            baseline_power = detector.baseline_power[idx]
            
            # Calculate current power manually
            ch_data = data_chunk[idx, -int(0.5 * detector.fs):]  # Last 0.5 seconds
            ch_data = ch_data - np.mean(ch_data)  # Remove DC
            
            # Simple power calculation
            current_power = np.mean(ch_data ** 2)
            
            # ERD calculation
            if baseline_power > 0:
                erd = ((baseline_power - current_power) / baseline_power) * 100
                print(f"  Baseline: {baseline_power:.6e}")
                print(f"  Current:  {current_power:.6e}")
                print(f"  ERD:      {erd:.1f}%")
            else:
                print(f"  Cannot calculate - baseline is {baseline_power}")


# Standalone test to verify ERD calculation
def test_erd_calculation():
    """Test ERD calculation with known synthetic data"""
    from erd_detection_system import AdaptiveERDDetector
    
    print("Testing ERD with synthetic data...")
    
    # Create detector
    fs = 1000
    detector = AdaptiveERDDetector(sampling_freq=fs)
    
    # Create synthetic channels
    n_channels = 3
    n_samples = 5 * fs  # 5 seconds
    
    # Generate data with clear ERD
    data = np.zeros((n_channels, n_samples))
    t = np.arange(n_samples) / fs
    
    for i in range(n_channels):
        # First 2 seconds: strong 10 Hz oscillation (baseline)
        data[i, :2*fs] = 20 * np.sin(2 * np.pi * 10 * t[:2*fs])
        # Last 3 seconds: weak 10 Hz oscillation (ERD)
        data[i, 2*fs:] = 5 * np.sin(2 * np.pi * 10 * t[2*fs:])
        # Add noise
        data[i, :] += np.random.randn(n_samples) * 2
    
    # Setup detector
    detector.set_channels(['Ch1', 'Ch2', 'Ch3'], [0, 1, 2])
    
    # Process data in chunks
    chunk_size = 100
    results = []
    
    for i in range(0, n_samples, chunk_size):
        chunk = data[:, i:i+chunk_size]
        detected, erd_values = detector.detect_erd(chunk)
        
        if erd_values:
            avg_erd = np.mean(list(erd_values.values()))
            results.append({
                'time': i/fs,
                'erd': avg_erd,
                'detected': detected
            })
    
    # Check results
    print(f"\nResults summary:")
    if results:
        early_erds = [r['erd'] for r in results if r['time'] < 2.5]
        late_erds = [r['erd'] for r in results if r['time'] > 2.5]
        
        if early_erds:
            print(f"  Early ERD (baseline): {np.mean(early_erds):.1f}%")
        if late_erds:
            print(f"  Late ERD (desync):    {np.mean(late_erds):.1f}%")
            print(f"  Expected: ~75% ERD")
            
        if late_erds and np.mean(late_erds) > 50:
            print("✅ ERD calculation working correctly!")
        else:
            print("❌ ERD calculation may have issues")
    else:
        print("❌ No ERD values calculated")


# Quick patch function to use with existing code
def patch_virtual_receiver(emulator):
    """
    Quick patch for existing virtual receiver
    """
    # Store original get_data
    original_get_data = emulator.get_data
    
    def patched_get_data():
        # Get original data
        data = original_get_data()
        
        if data is not None:
            # Check scaling
            data_std = np.std(data)
            
            # Auto-scale if needed
            if data_std > 1000:
                # Likely raw ADC values
                scale_factor = 50.0 / data_std  # Target 50 µV std
                data = data * scale_factor
                print(f"Auto-scaled data by {scale_factor:.6f}")
                
            elif data_std < 0.1:
                # Likely in volts
                data = data * 1e4
                pass
                # print("Converted from volts to microvolts")
            
            # Remove DC offset
            for i in range(data.shape[0]):
                data[i, :] -= np.mean(data[i, :])
        
        return data
    
    # Replace method
    emulator.get_data = patched_get_data
    print("Virtual receiver patched with auto-scaling")
    
    return emulator


if __name__ == "__main__":
    print("ERD Quick Fix Utilities")
    print("=" * 50)
    
    # Run synthetic test
    test_erd_calculation()
    
    print("\n" + "=" * 50)
    print("\nTo use these fixes in your code:")
    print("\n1. For ERD detector:")
    print("   from erd_quick_fix import fix_erd_detector")
    print("   detector = fix_erd_detector(detector)")
    
    print("\n2. For virtual receiver:")
    print("   from erd_quick_fix import patch_virtual_receiver")
    print("   emulator = patch_virtual_receiver(emulator)")
    
    print("\n3. To diagnose issues:")
    print("   from erd_quick_fix import diagnose_erd_issue")
    print("   diagnose_erd_issue(detector, data_chunk)")