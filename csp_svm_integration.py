'''
csp_svm_integration.py
---------------------
CSP+SVM integration module for enhanced ERD detection
Includes robust baseline and multi-band analysis
'''

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from collections import deque


class SimpleCSP:
    """Simplified CSP implementation that actually works"""
    
    def __init__(self, n_components=4, autotrain=False):
        self.n_components = n_components
        self.filters_ = None
        self.mean_ = None
        self.std_ = None
        self.autotrain = autotrain
        
    def fit(self, X_rest, X_mi):
        """
        Fit CSP using rest and motor imagery data
        X_rest, X_mi: (n_trials, n_channels, n_samples)
        """
        # Calculate normalized covariance matrices
        cov_rest = self._compute_covariance(X_rest)
        cov_mi = self._compute_covariance(X_mi)
        
        # Composite covariance
        cov_combined = cov_rest + cov_mi
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(cov_rest, cov_combined)
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select filters (first and last n_components/2)
        if self.n_components < eigenvectors.shape[1]:
            # Take extreme eigenvalues (best for each class)
            n_pairs = self.n_components // 2
            indices = np.concatenate([np.arange(n_pairs), 
                                    np.arange(-n_pairs, 0)])
        else:
            indices = np.arange(eigenvectors.shape[1])
            
        self.filters_ = eigenvectors[:, indices].T
        
        # Compute normalization parameters on training data
        features_rest = self.transform(X_rest)
        features_mi = self.transform(X_mi)
        all_features = np.vstack([features_rest, features_mi])
        self.mean_ = np.mean(all_features, axis=0)
        self.std_ = np.std(all_features, axis=0) + 1e-8
        
        return self
    
    def _compute_covariance(self, X):
        """Compute average covariance matrix"""
        n_trials = X.shape[0]
        n_channels = X.shape[1]
        
        cov_sum = np.zeros((n_channels, n_channels))
        for trial in X:
            # Normalize by trace
            cov = np.cov(trial)
            cov_sum += cov / np.trace(cov)
            
        return cov_sum / n_trials
    
    def transform(self, X):
        """Transform data using CSP filters"""
        if self.filters_ is None:
            raise ValueError("CSP not fitted yet")
            
        n_trials = X.shape[0]
        n_components = self.filters_.shape[0]
        
        features = np.zeros((n_trials, n_components))
        
        for i, trial in enumerate(X):
            # Apply spatial filters
            filtered = self.filters_ @ trial
            
            # Compute log-variance features
            features[i] = np.log(np.var(filtered, axis=1) + 1e-8)
        
        # Normalize if parameters are available
        if self.mean_ is not None:
            features = (features - self.mean_) / self.std_
            
        return features


class MultiFrequencyCSP:
    """CSP applied to multiple frequency bands"""
    
    def __init__(self, bands, fs, n_components=4):
        self.bands = bands  # Dict of {name: (low, high)}
        self.fs = fs
        self.n_components = n_components
        self.csp_models = {}
        self.filters = {}
        
    def fit(self, X_rest, X_mi):
        """Fit CSP for each frequency band"""
        for band_name, (low, high) in self.bands.items():
            print(f"  Fitting CSP for {band_name} band ({low}-{high} Hz)...")
            
            # Filter data to this band
            X_rest_filt = self._bandpass_filter(X_rest, low, high)
            X_mi_filt = self._bandpass_filter(X_mi, low, high)
            
            # Fit CSP for this band
            csp = SimpleCSP(n_components=self.n_components)
            csp.fit(X_rest_filt, X_mi_filt)
            
            self.csp_models[band_name] = csp
            
        return self
    
    def transform(self, X):
        """Extract features from all bands"""
        all_features = []
        
        for band_name, (low, high) in self.bands.items():
            # Filter to band
            X_filt = self._bandpass_filter(X, low, high)
            
            # Apply CSP
            features = self.csp_models[band_name].transform(X_filt)
            all_features.append(features)
        
        # Concatenate all band features
        return np.hstack(all_features)
    
    def _bandpass_filter(self, X, low, high):
        """Apply bandpass filter to data"""
        nyquist = self.fs / 2
        low_norm = max(low / nyquist, 0.01)
        high_norm = min(high / nyquist, 0.99)
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        # Filter each trial and channel
        X_filt = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filt[i, j, :] = filtfilt(b, a, X[i, j, :])
                
        return X_filt


class CSPSVMDetector:
    """Complete CSP+SVM detection system"""
    
    def __init__(self, fs, n_channels, use_multiband=True, autotrain = False):
        self.fs = fs
        self.n_channels = n_channels
        self.use_multiband = use_multiband
        self.autotrain = autotrain
        
        # Define frequency bands
        if use_multiband:
            self.bands = {
                'mu': (8, 12),
                'beta_low': (13, 20),
                'beta_high': (20, 30)
            }
            self.csp = MultiFrequencyCSP(self.bands, fs, n_components=4)
        else:
            # Single band CSP
            self.csp = SimpleCSP(n_components=6)
            
        # Classifier
        self.svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        
        # Training data storage
        self.training_data = {
            'rest': deque(maxlen=200),  # Store windows
            'mi': deque(maxlen=200)
        }
        
        self.is_trained = False
        
    def collect_training_data(self, window_data, label):
        """Collect training data (0=rest, 1=MI)"""
        if label == 0:
            self.training_data['rest'].append(window_data)
        else:
            self.training_data['mi'].append(window_data)
            
        # Check if we have enough data to train
        # if self.autotrain:
        #     if (len(self.training_data['rest']) >= 20 and 
        #         len(self.training_data['mi']) >= 20 and 
        #         not self.is_trained):
        #         print("\n🎯 Sufficient training data collected. Training CSP+SVM...")
        #         self.train()
    
    def train(self):
        """Train CSP and SVM on collected data"""
        # Convert to arrays
        X_rest = np.array(self.training_data['rest'])
        X_mi = np.array(self.training_data['mi'])
        
        print(f"  Training with {len(X_rest)} rest and {len(X_mi)} MI windows")
        
        try:
            # Fit CSP
            if self.use_multiband:
                # Apply bandpass and fit CSP for each band
                self.csp.fit(X_rest, X_mi)
            else:
                # Single band - filter to mu+beta (8-30 Hz)
                X_rest_filt = self._bandpass_filter(X_rest, 8, 30)
                X_mi_filt = self._bandpass_filter(X_mi, 8, 30)
                self.csp.fit(X_rest_filt, X_mi_filt)
            
            # Extract features
            features_rest = self.csp.transform(X_rest)
            features_mi = self.csp.transform(X_mi)
            
            # Prepare training data
            X_train = np.vstack([features_rest, features_mi])
            y_train = np.hstack([np.zeros(len(features_rest)), 
                                np.ones(len(features_mi))])
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train SVM
            self.svm.fit(X_train_scaled, y_train)
            
            # Cross-validation score
            scores = cross_val_score(self.svm, X_train_scaled, y_train, cv=5)
            print(f"  ✓ CSP+SVM trained! CV accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
            
            self.is_trained = True
            
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            self.is_trained = False

        return self.is_trained
    
    def predict(self, window_data):
        """Predict using CSP+SVM"""
        if not self.is_trained:
            return None, 0.0
            
        try:
            # Ensure correct shape
            if window_data.ndim == 2:
                window_data = window_data[np.newaxis, :, :]
            
            # Extract CSP features
            features = self.csp.transform(window_data)
            
            # Scale
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.svm.predict(features_scaled)[0]
            probability = self.svm.predict_proba(features_scaled)[0, 1]  # P(MI)
            
            return int(prediction), float(probability)
            
        except Exception as e:
            print(f"CSP+SVM prediction error: {e}")
            return None, 0.0
    
    def _bandpass_filter(self, X, low, high):
        """Apply bandpass filter"""
        nyquist = self.fs / 2
        low_norm = max(low / nyquist, 0.01)
        high_norm = min(high / nyquist, 0.99)
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        X_filt = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filt[i, j, :] = filtfilt(b, a, X[i, j, :])
                
        return X_filt
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.is_trained:
            model_data = {
                'csp': self.csp,
                'svm': self.svm,
                'scaler': self.scaler,
                'bands': self.bands if self.use_multiband else None,
                'use_multiband': self.use_multiband
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.csp = model_data['csp']
            self.svm = model_data['svm']
            self.scaler = model_data['scaler']
            self.bands = model_data.get('bands')
            self.use_multiband = model_data.get('use_multiband', True)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model: {e}")


class RobustBaseline:
    """Robust baseline calculation methods"""
    
    @staticmethod
    def calculate_robust_baseline(data, method='trimmed_mean'):
        """
        Calculate robust baseline using various methods
        
        data: (n_channels, n_samples)
        method: 'trimmed_mean', 'median', 'winsorized', 'adaptive'
        """
        if method == 'trimmed_mean':
            # Remove top and bottom 10% before averaging
            return RobustBaseline._trimmed_mean(data, trim_percent=0.1)
            
        elif method == 'median':
            # Use median instead of mean
            return np.median(data, axis=1)
            
        elif method == 'winsorized':
            # Cap extreme values before averaging
            return RobustBaseline._winsorized_mean(data, limits=(0.05, 0.05))
            
        elif method == 'adaptive':
            # Adaptive method based on data distribution
            return RobustBaseline._adaptive_baseline(data)
            
        else:
            # Fallback to standard mean
            return np.mean(data, axis=1)
    
    @staticmethod
    def _trimmed_mean(data, trim_percent=0.1):
        """Calculate trimmed mean per channel"""
        n_samples = data.shape[1]
        trim_count = int(n_samples * trim_percent)
        
        baseline = np.zeros(data.shape[0])
        for ch in range(data.shape[0]):
            sorted_data = np.sort(data[ch, :])
            if trim_count > 0:
                trimmed = sorted_data[trim_count:-trim_count]
            else:
                trimmed = sorted_data
            baseline[ch] = np.mean(trimmed) if len(trimmed) > 0 else np.mean(sorted_data)
            
        return baseline
    
    @staticmethod
    def _winsorized_mean(data, limits=(0.05, 0.05)):
        """Calculate winsorized mean"""
        from scipy.stats import mstats
        
        baseline = np.zeros(data.shape[0])
        for ch in range(data.shape[0]):
            baseline[ch] = mstats.winsorize(data[ch, :], limits=limits).mean()
            
        return baseline
    
    @staticmethod
    def _adaptive_baseline(data):
        """Adaptive baseline based on distribution"""
        baseline = np.zeros(data.shape[0])
        
        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            
            # Check for outliers using IQR
            q1, q3 = np.percentile(channel_data, [25, 75])
            iqr = q3 - q1
            
            # If high variability, use robust method
            if iqr > np.std(channel_data) * 1.5:
                baseline[ch] = np.median(channel_data)
            else:
                baseline[ch] = np.mean(channel_data)
                
        return baseline


def integrate_csp_svm_with_erd(erd_system):
    """
    Integrate CSP+SVM with existing ERD detection system
    Call this after ERD system initialization
    """
    print("\n🔧 Integrating CSP+SVM with ERD detection...")
    
    # Create CSP+SVM detector
    csp_svm = CSPSVMDetector(
        fs=erd_system.fs,
        n_channels=len(erd_system.erd_channel_indices),
        use_multiband=True
    )
    
    # Add to ERD system
    erd_system.csp_svm_detector = csp_svm
    
    # Modify detection method to include CSP+SVM
    original_detect = erd_system.detect_erd
    
    def enhanced_detect_erd(self, data):
        # Original ERD detection
        erd_detected, erd_values, erd_confidence = original_detect(data)
        
        # CSP+SVM detection if trained
        csp_prediction = None
        csp_confidence = 0.0
        
        if hasattr(self, 'csp_svm_detector') and self.csp_svm_detector.is_trained:
            csp_prediction, csp_confidence = self.csp_svm_detector.predict(data)
        
        # Combine decisions
        if csp_prediction is not None:
            # Weighted combination
            combined_confidence = 0.7 * csp_confidence + 0.3 * (erd_confidence / 100.0)
            detected = combined_confidence > 0.5
            
            # Update values for display
            erd_values['csp_svm'] = csp_confidence * 100
            
            return detected, erd_values, combined_confidence
        else:
            # Fallback to original ERD
            return erd_detected, erd_values, erd_confidence
    
    # Replace method
    erd_system.detect_erd = lambda data: enhanced_detect_erd(erd_system, data)
    
    print("  ✓ CSP+SVM integration complete!")
    print("  - Collect training data by marking rest/MI periods")
    print("  - System will auto-train after 20 examples of each")
    
    return csp_svm


# Training data collection helper
class TrainingCollector:
    """Helper to collect labeled training data"""
    
    def __init__(self, csp_svm_detector):
        self.detector = csp_svm_detector
        self.collecting = False
        self.current_label = None
        
    def start_collecting(self, label):
        """Start collecting data with label (0=rest, 1=MI)"""
        self.collecting = True
        self.current_label = label
        label_name = "REST" if label == 0 else "MOTOR IMAGERY"
        print(f"\n📝 Collecting {label_name} training data...")
        
    def stop_collecting(self):
        """Stop collecting data"""
        self.collecting = False
        self.current_label = None
        print("  ✓ Stopped collecting")
        
    def add_window(self, window_data):
        """Add window if collecting"""
        if self.collecting and self.current_label is not None:
            self.detector.collect_training_data(window_data, self.current_label)


# Example usage function
def setup_enhanced_erd_with_csp(bci_system):
    """
    Complete setup for enhanced ERD with CSP+SVM
    Call this after BCIERDSystem initialization
    """
    # 1. Integrate CSP+SVM
    csp_svm = integrate_csp_svm_with_erd(bci_system.erd_detector)
    
    # 2. Create training collector
    collector = TrainingCollector(csp_svm)
    bci_system.training_collector = collector
    
    # 3. Add keyboard commands for training
    original_keyboard = bci_system.handle_keyboard_input
    
    def enhanced_keyboard_input(self):
        original_keyboard()
        
        try:
            import select
            import sys
            
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                
                if key == '0':
                    self.training_collector.start_collecting(0)  # Rest
                elif key == '1':
                    self.training_collector.start_collecting(1)  # MI
                elif key == '9':
                    self.training_collector.stop_collecting()
                elif key == 'p':
                    if hasattr(self.erd_detector, 'csp_svm_detector'):
                        detector = self.erd_detector.csp_svm_detector
                        print(f"\nTraining data collected:")
                        print(f"  Rest: {len(detector.training_data['rest'])} windows")
                        print(f"  MI: {len(detector.training_data['mi'])} windows")
                        print(f"  Trained: {detector.is_trained}")
                        
        except:
            pass
    
    bci_system.handle_keyboard_input = lambda: enhanced_keyboard_input(bci_system)
    
    print("\n📚 Enhanced ERD system ready!")
    print("Keyboard commands for training:")
    print("  0 - Start collecting REST data")
    print("  1 - Start collecting MOTOR IMAGERY data")  
    print("  9 - Stop collecting")
    print("  p - Show training status")
    print("\nSystem will auto-train CSP+SVM after 20 samples of each class")
    
    return collector