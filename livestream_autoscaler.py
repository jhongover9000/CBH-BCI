'''
Adaptive Auto-Scaler for Live EEG Streaming
Handles real-time scaling without knowing future data
'''

import numpy as np
from collections import deque
import warnings


class LivestreamAutoScaler:
    """
    Adaptive scaler for real-time EEG data
    Learns appropriate scaling from initial data
    """
    
    def __init__(self, target_std=50.0, learning_duration=2.0, fs=1000):
        """
        Args:
            target_std: Target standard deviation in microvolts (default 50 µV)
            learning_duration: How long to observe before determining scaling (seconds)
            fs: Sampling frequency
        """
        self.target_std = target_std
        self.learning_duration = learning_duration
        self.fs = fs
        
        # State
        self.scaling_factor = 1.0
        self.is_calibrated = False
        self.calibration_buffer = []
        self.samples_needed = int(learning_duration * fs)
        self.samples_collected = 0
        
        # Statistics tracking
        self.stats_buffer = deque(maxlen=int(fs * 10))  # 10 second rolling window
        self.dc_offsets = None
        self.artifact_threshold = 500.0  # µV, for artifact detection
        
        # Scaling detection
        self.scaling_history = deque(maxlen=10)
        self.scaling_stable = False
        
    def process(self, data):
        """
        Process incoming data chunk and apply appropriate scaling
        
        Args:
            data: numpy array (n_channels, n_samples)
            
        Returns:
            scaled_data: properly scaled data in microvolts
        """
        
        if not self.is_calibrated:
            # During calibration phase
            scaled_data = self._calibration_phase(data)
        else:
            # Normal operation
            scaled_data = self._apply_scaling(data)
            
        return scaled_data
    
    def _calibration_phase(self, data):
        """
        Learn appropriate scaling during initial period
        """
        # Store raw data
        self.calibration_buffer.append(data.copy())
        self.samples_collected += data.shape[1]
        
        # Apply current scaling estimate
        scaled_data = data * self.scaling_factor
        
        # Check if we have enough data
        if self.samples_collected >= self.samples_needed:
            # Analyze all calibration data
            all_data = np.hstack(self.calibration_buffer)
            self._determine_scaling(all_data)
            self.is_calibrated = True
            
            print(f"\n✓ Auto-scaling calibrated:")
            print(f"  Scaling factor: {self.scaling_factor:.6f}")
            print(f"  DC offsets removed: {self.dc_offsets is not None}")
            
            # Clear buffer to save memory
            self.calibration_buffer = []
            
        return scaled_data
    
    def _determine_scaling(self, data):
        """
        Determine appropriate scaling factor from calibration data
        """
        n_channels = data.shape[0]
        
        # Calculate statistics per channel
        channel_stds = np.std(data, axis=1)
        channel_means = np.mean(data, axis=1)
        
        # Remove outlier channels (bad electrodes)
        median_std = np.median(channel_stds)
        valid_channels = np.abs(channel_stds - median_std) < 3 * median_std
        
        if np.sum(valid_channels) < n_channels * 0.5:
            warnings.warn("More than 50% of channels appear to be bad!")
        
        # Calculate scaling based on valid channels
        valid_stds = channel_stds[valid_channels]
        avg_std = np.mean(valid_stds)
        
        # Determine scaling factor
        if avg_std > 1000:
            # Likely raw ADC units
            self.scaling_factor = self.target_std / avg_std
            print(f"  Detected raw ADC units (std={avg_std:.1f})")
            
        elif avg_std < 0.1:
            # Likely in volts
            self.scaling_factor = 1e6  # Convert to microvolts
            print(f"  Detected volts (std={avg_std:.6f})")
            
        elif avg_std < 10:
            # Might be in millivolts
            self.scaling_factor = 1000  # Convert to microvolts
            print(f"  Detected millivolts (std={avg_std:.3f})")
            
        else:
            # Assume already in microvolts but adjust if needed
            self.scaling_factor = self.target_std / avg_std
            print(f"  Already in microvolts (std={avg_std:.1f})")
        
        # Check for DC offsets
        if np.any(np.abs(channel_means) > 10):
            self.dc_offsets = channel_means
            print(f"  DC offset detected (mean={np.mean(np.abs(channel_means)):.1f} µV)")
    
    def _apply_scaling(self, data):
        """
        Apply calibrated scaling to new data
        """
        # Apply scaling
        scaled_data = data * self.scaling_factor
        
        # Remove DC offset if detected during calibration
        if self.dc_offsets is not None:
            for i in range(scaled_data.shape[0]):
                if i < len(self.dc_offsets):
                    scaled_data[i, :] -= self.dc_offsets[i]
        
        # Update statistics
        self.stats_buffer.extend(scaled_data.flatten())
        
        # Check for artifacts or scaling drift
        if len(self.stats_buffer) > 1000:
            current_std = np.std(list(self.stats_buffer)[-1000:])
            
            # Detect if scaling has drifted significantly
            if current_std > self.target_std * 3 or current_std < self.target_std * 0.3:
                if not self.scaling_stable:
                    warnings.warn(f"Scaling may have drifted (current std={current_std:.1f} µV)")
        
        return scaled_data
    
    def check_data_quality(self, data):
        """
        Check for common data quality issues
        """
        issues = []
        
        # Check for flat channels
        channel_stds = np.std(data, axis=1)
        flat_channels = np.where(channel_stds < 0.01)[0]
        if len(flat_channels) > 0:
            issues.append(f"Flat channels detected: {flat_channels}")
        
        # Check for excessive artifacts
        max_values = np.max(np.abs(data), axis=1)
        artifact_channels = np.where(max_values > self.artifact_threshold)[0]
        if len(artifact_channels) > 0:
            issues.append(f"Artifact on channels: {artifact_channels}")
        
        # Check for clipping
        clip_high = np.sum(data == np.max(data))
        clip_low = np.sum(data == np.min(data))
        if clip_high > data.size * 0.01 or clip_low > data.size * 0.01:
            issues.append("Possible clipping detected")
        
        return issues
    
    def reset(self):
        """Reset calibration"""
        self.scaling_factor = 1.0
        self.is_calibrated = False
        self.calibration_buffer = []
        self.samples_collected = 0
        self.dc_offsets = None
        self.stats_buffer.clear()


# Wrapper for livestream receiver with auto-scaling
class ScaledLivestreamReceiver:
    """
    Wrapper around LivestreamReceiver that adds auto-scaling
    """
    
    def __init__(self, address="169.254.1.147", port=51244, broadcast=False,
                 auto_scale=True, target_std=50.0):
        # Create original receiver
        from receivers.livestream_receiver import LivestreamReceiver
        self.receiver = LivestreamReceiver(address, port, broadcast)
        
        # Auto-scaling
        self.auto_scale = auto_scale
        self.scaler = None
        self.target_std = target_std
        
    def initialize_connection(self):
        """Initialize connection and scaler"""
        # Initialize original connection
        fs, ch_names, n_channels, data = self.receiver.initialize_connection()
        
        # Create scaler if enabled
        if self.auto_scale:
            self.scaler = LivestreamAutoScaler(
                target_std=self.target_std,
                learning_duration=2.0,
                fs=fs
            )
            print("Auto-scaling enabled (2 second calibration)")
        
        return fs, ch_names, n_channels, data
    
    def get_data(self):
        """Get scaled data"""
        # Get raw data
        raw_data = self.receiver.get_data()
        
        if raw_data is None:
            return None
        
        # Apply scaling if enabled
        if self.auto_scale and self.scaler:
            scaled_data = self.scaler.process(raw_data)
            
            # Check data quality periodically
            if np.random.rand() < 0.01:  # 1% of chunks
                issues = self.scaler.check_data_quality(scaled_data)
                if issues:
                    print(f"Data quality issues: {issues}")
            
            return scaled_data
        else:
            return raw_data
    
    def use_classification(self, prediction):
        """Pass through to original receiver"""
        return self.receiver.use_classification(prediction)
    
    def disconnect(self):
        """Disconnect"""
        return self.receiver.disconnect()
    
    # Pass through other attributes
    def __getattr__(self, name):
        return getattr(self.receiver, name)


# Example usage for testing
def test_livestream_scaling():
    """Test scaling with simulated livestream data"""
    print("Testing Livestream Auto-Scaler")
    print("=" * 50)
    
    # Create scaler
    scaler = LivestreamAutoScaler(target_std=50.0, learning_duration=1.0, fs=1000)
    
    # Simulate different data scenarios
    scenarios = [
        ("Raw ADC", 5000, 1000),    # std=5000, offset=1000
        ("Microvolts", 50, 10),      # std=50, offset=10
        ("Volts", 0.00005, 0.001),   # std=50µV, offset=1mV in volts
    ]
    
    for name, std, offset in scenarios:
        print(f"\nTesting {name} data:")
        scaler.reset()
        
        # Generate test data
        for i in range(25):  # 2.5 seconds at 100ms chunks
            n_channels = 64
            n_samples = 100
            
            # Create data with known statistics
            data = np.random.randn(n_channels, n_samples) * std + offset
            
            # Process
            scaled = scaler.process(data)
            
            if i % 10 == 0:
                print(f"  Chunk {i}: in_std={np.std(data):.2f}, "
                      f"out_std={np.std(scaled):.2f}, "
                      f"calibrated={scaler.is_calibrated}")
        
        print(f"  Final scaling factor: {scaler.scaling_factor:.6f}")


if __name__ == "__main__":
    test_livestream_scaling()