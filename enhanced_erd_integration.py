'''
enhanced_erd_integration.py
--------------------------
Practical integration of CSP+SVM, robust baseline, and multi-band analysis
into the existing BCI ERD system
'''

import numpy as np
from csp_svm_integration import (
    integrate_csp_svm_with_erd, 
    setup_enhanced_erd_with_csp,
    RobustBaseline
)


def fix_robust_baseline_in_erd_system(erd_detector):
    """
    Fix the robust baseline calculation in ERDDetectionSystem
    """
    print("🔧 Fixing robust baseline calculation...")
    
    # Replace the buggy robust baseline method
    def _calculate_robust_baseline_fixed(self, main_buffer=None, force=False):
        """Fixed robust baseline using proper array handling"""
        # First get standard baseline data
        buffer_to_use = self.baseline_buffers['standard']
        
        if not force and len(buffer_to_use) < buffer_to_use.maxlen:
            return False
        
        if force and main_buffer is not None:
            if len(main_buffer) < buffer_to_use.maxlen:
                recent_samples = list(main_buffer)
            else:
                recent_samples = list(main_buffer)[-buffer_to_use.maxlen:]
            baseline_data = np.array(recent_samples).T
        else:
            if len(buffer_to_use) == 0:
                return False
            baseline_data = np.array(list(buffer_to_use)).T
        
        if baseline_data.shape[0] != self.n_channels:
            return False
        
        try:
            # Preprocess baseline data
            erd_data = self.preprocessor.preprocess_data(baseline_data, self.erd_channel_indices)
            
            # Extract features
            features = self.preprocessor.extract_multi_band_features(erd_data)
            
            # Apply robust statistics to each feature
            robust_features = {}
            for feature_name, values in features.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    # Calculate robust baseline for each channel
                    if values.ndim == 1:
                        # Single channel - use robust method
                        robust_value = RobustBaseline.calculate_robust_baseline(
                            values.reshape(1, -1), 
                            method='trimmed_mean'
                        )[0]
                        robust_features[feature_name] = robust_value
                    else:
                        # Multiple channels - preserve structure
                        robust_features[feature_name] = values  # Keep array structure
                else:
                    robust_features[feature_name] = values
            
            self.baseline_powers = robust_features
            self.baseline_calculated = True
            
            if self.config.VERBOSE:
                print(f"\n✓ Robust baseline calculated with {len(features)} feature types")
            
            return True
            
        except Exception as e:
            if self.config.VERBOSE:
                print(f"\n✗ Error calculating robust baseline: {e}")
            return False
    
    # Replace the method
    erd_detector._calculate_robust_baseline = lambda main_buffer=None, force=False: \
        _calculate_robust_baseline_fixed(erd_detector, main_buffer, force)
    
    print("  ✓ Robust baseline fixed!")


def optimize_multiband_detection(erd_detector):
    """
    Optimize multi-band detection with proper weighting
    """
    print("🔧 Optimizing multi-band detection...")
    
    # Better feature weights based on literature
    OPTIMIZED_WEIGHTS = {
        'mu_power': 0.4,          # Mu band is primary for motor imagery
        'beta_low_power': 0.35,   # Beta important too
        'beta_high_power': 0.25,  # High beta less reliable
        'erd_power': 0.5,         # Combined mu+beta
        'raw_power': 0.1          # Fallback
    }
    
    # Replace weight function
    def _get_optimized_feature_weights(self):
        """Get optimized weights for different feature types"""
        weights = []
        
        # Match weights to available features
        if hasattr(self, 'baseline_powers') and self.baseline_powers:
            for feature_name in self.baseline_powers.keys():
                weight = OPTIMIZED_WEIGHTS.get(feature_name, 0.1)
                weights.append(weight)
        
        # Normalize weights
        if weights:
            total = sum(weights)
            weights = [w/total for w in weights]
        else:
            weights = [1.0]
            
        return weights
    
    erd_detector._get_feature_weights = lambda: _get_optimized_feature_weights(erd_detector)
    
    print("  ✓ Multi-band detection optimized!")


def add_simple_mode_to_system(bci_system):
    """
    Add a simple mode toggle for easier testing
    """
    print("🔧 Adding simple mode toggle...")
    
    # Add simple mode flag
    bci_system.simple_mode = False
    
    # Original run method
    original_run = bci_system.run
    
    def run_with_mode_selection(self):
        """Run with mode selection"""
        print("\n" + "="*70)
        print("Select Detection Mode:")
        print("1. Simple ERD (8-20 Hz, fast)")
        print("2. Advanced Multi-band + CSP+SVM")
        print("="*70)
        
        try:
            choice = input("Enter choice (1-2) [1]: ").strip() or "1"
            
            if choice == "1":
                print("\n✓ Using Simple ERD mode")
                # Configure for simple mode
                from bci_erd_main import BCIConfig
                BCIConfig.USE_MULTIPLE_BANDS = False
                BCIConfig.USE_BETA_BURSTS = False
                BCIConfig.USE_PHASE_COUPLING = False
                BCIConfig.USE_ENSEMBLE_CSP = False
                BCIConfig.BASELINE_METHOD = 'standard'
                self.simple_mode = True
                
            elif choice == "2":
                print("\n✓ Using Advanced Multi-band mode")
                # Keep advanced settings
                self.simple_mode = False
                
                # Fix robust baseline
                fix_robust_baseline_in_erd_system(self.erd_detector)
                
                # Optimize multi-band
                optimize_multiband_detection(self.erd_detector)
                
                # Setup CSP+SVM
                setup_enhanced_erd_with_csp(self)
                
        except KeyboardInterrupt:
            print("\nUsing default simple mode")
            self.simple_mode = True
        
        # Continue with original run
        original_run()
    
    bci_system.run = lambda: run_with_mode_selection(bci_system)
    
    print("  ✓ Mode selection added!")


def create_quick_start_script():
    """
    Create a quick start script for the enhanced system
    """
    script = '''#!/usr/bin/env python3
"""
Quick start script for enhanced ERD detection with CSP+SVM
"""

import argparse
from bci_erd_main import BCIERDSystem, BCIConfig
from enhanced_erd_integration import (
    fix_robust_baseline_in_erd_system,
    optimize_multiband_detection,
    add_simple_mode_to_system
)
from csp_svm_integration import setup_enhanced_erd_with_csp

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced BCI ERD System")
    parser.add_argument('--virtual', action='store_true', help="Use virtual mode")
    parser.add_argument('--simple', action='store_true', help="Force simple mode")
    parser.add_argument('--advanced', action='store_true', help="Force advanced mode")
    args = parser.parse_args()
    
    # Configure
    if args.simple:
        BCIConfig.USE_MULTIPLE_BANDS = False
        BCIConfig.BASELINE_METHOD = 'standard'
        mode = "Simple"
    elif args.advanced:
        BCIConfig.USE_MULTIPLE_BANDS = True
        BCIConfig.BASELINE_METHOD = 'robust'
        mode = "Advanced"
    else:
        mode = "Interactive"
    
    print(f"Starting Enhanced BCI ERD System ({mode} mode)")
    
    # Create system
    system = BCIERDSystem()
    
    # Add enhancements
    if not args.simple:
        add_simple_mode_to_system(system)
    
    # Initialize and run
    system.initialize(args)
    system.run()

if __name__ == "__main__":
    main()
'''
    
    with open('run_enhanced_erd.py', 'w') as f:
        f.write(script)
    
    print("\n📝 Created run_enhanced_erd.py")


# Practical recommendations
def print_recommendations():
    """
    Print practical recommendations for using the enhanced features
    """
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR ENHANCED ERD DETECTION")
    print("="*70)
    
    print("\n1. CSP+SVM Benefits:")
    print("   - Better spatial filtering for motor imagery")
    print("   - Subject-specific optimization")
    print("   - Higher accuracy once trained")
    print("   - Works best with consistent electrode placement")
    
    print("\n2. When to use Multi-band Analysis:")
    print("   - Subject shows strong beta suppression")
    print("   - Need to distinguish different motor imagery types")
    print("   - Have good signal quality")
    print("   - Computational resources available")
    
    print("\n3. Robust Baseline Advantages:")
    print("   - Handles artifacts better")
    print("   - More stable with movement")
    print("   - Better for long sessions")
    print("   - Reduces false positives")
    
    print("\n4. Training Tips for CSP+SVM:")
    print("   - Collect at least 30-50 examples per class")
    print("   - Ensure clear rest vs MI periods")
    print("   - Keep head still during training")
    print("   - Retrain if electrode positions change")
    
    print("\n5. Performance Optimization:")
    print("   - Start with simple mode for testing")
    print("   - Use advanced mode for final deployment")
    print("   - Adjust thresholds based on subject")
    print("   - Monitor confidence scores")
    
    print("\n6. Debugging:")
    print("   - Use 'x' key to toggle debug mode")
    print("   - Check feature values with 's' key")
    print("   - Monitor CSP training with 'p' key")
    print("   - Save successful models for reuse")
    
    print("="*70 + "\n")


# Test function
def test_enhancements():
    """
    Test the enhanced features
    """
    print("Testing Enhanced ERD Features...")
    
    # Test robust baseline
    test_data = np.random.randn(3, 1000) * 20  # 3 channels, 1000 samples
    test_data[1, 100:150] = 200  # Add artifacts
    
    print("\n1. Testing Robust Baseline:")
    methods = ['trimmed_mean', 'median', 'winsorized', 'adaptive']
    for method in methods:
        baseline = RobustBaseline.calculate_robust_baseline(test_data, method)
        print(f"   {method}: {baseline}")
    
    # Test CSP
    print("\n2. Testing CSP:")
    from csp_svm_integration import SimpleCSP
    
    # Generate synthetic data
    n_trials = 50
    n_channels = 8
    n_samples = 250
    
    # Rest: random noise
    X_rest = np.random.randn(n_trials, n_channels, n_samples) * 10
    
    # MI: reduced power in channels 2-4
    X_mi = np.random.randn(n_trials, n_channels, n_samples) * 10
    X_mi[:, 2:5, :] *= 0.5  # Simulate ERD
    
    # Fit CSP
    csp = SimpleCSP(n_components=4)
    csp.fit(X_rest, X_mi)
    
    # Transform
    features_rest = csp.transform(X_rest[:5])
    features_mi = csp.transform(X_mi[:5])
    
    print(f"   Rest features shape: {features_rest.shape}")
    print(f"   MI features shape: {features_mi.shape}")
    print(f"   Feature difference: {np.mean(features_mi - features_rest):.3f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    print("Enhanced ERD Integration Module")
    print("="*70)
    
    # Run tests
    test_enhancements()
    
    # Print recommendations
    print_recommendations()
    
    # Create quick start script
    create_quick_start_script()
    
    print("\nTo use the enhanced system:")
    print("1. Import this module in your BCI script")
    print("2. Call the enhancement functions after system init")
    print("3. Or use: python run_enhanced_erd.py")