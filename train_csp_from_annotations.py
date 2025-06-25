#!/usr/bin/env python3
'''
train_csp_from_annotations.py
----------------------------
Train CSP+SVM model using annotations in recorded EEG data
Supports both offline files and online livestream training
'''

import numpy as np
import mne
import time
import argparse
import joblib
from collections import defaultdict
from scipy import signal
import os

# Import BCI components
from bci_erd_main import BCIConfig, ERDDetectionSystem
from csp_svm_integration import CSPSVMDetector
from receivers import virtual_receiver, livestream_receiver


class AnnotationBasedTrainer:
    """Train CSP+SVM using annotation markers"""
    
    def __init__(self, fs=250, multiband=True, verbose=True):
        self.fs = fs
        self.multiband = multiband
        self.verbose = verbose
        
        # Annotation mappings
        self.rest_annotations = ['Stimulus/S  4', 'S  4']
        self.mi_annotations = ['Stimulus/S  3', 'S  3']
        
        # Training data storage
        self.training_data = {
            'rest': [],
            'mi': []
        }
        
        # Frequency bands for multiband analysis
        self.frequency_bands = {
            'mu': (8, 12),
            'beta': (13, 30),
            'low_beta': (13, 20),
            'high_beta': (20, 30),
            'wide': (8, 30)
        }
        
        # Window parameters
        self.window_duration = 1.0  # seconds
        self.window_samples = int(self.window_duration * self.fs)
        self.annotation_offset = 0.5  # seconds after annotation to start window
        
    def train_from_file(self, filename, channel_names=None):
        """Train from recorded EEG file with annotations"""
        print(f"Loading data from {filename}...")
        
        # Load data based on file type
        if filename.endswith('.fif'):
            raw = mne.io.read_raw_fif(filename, preload=True)
        elif filename.endswith('.edf'):
            raw = mne.io.read_raw_edf(filename, preload=True)
        elif filename.endswith('.vhdr'):
            raw = mne.io.read_raw_brainvision(filename, preload=True)
            raw = raw.resample(250)
        else:
            raise ValueError("Unsupported file format. Use .fif or .edf")
        
        # Get data and annotations
        data = raw.get_data()
        annotations = raw.annotations
        sfreq = raw.info['sfreq']
        
        if channel_names is None:
            channel_names = raw.ch_names
        
        # Find ERD channels
        erd_indices = self._find_erd_channels(channel_names)
        
        print(f"Found {len(annotations)} annotations")
        print(f"Using {len(erd_indices)} ERD channels")
        
        # Process annotations
        self._process_annotations(data, annotations, erd_indices, sfreq)
        
        # Train model
        if len(self.training_data['rest']) >= 10 and len(self.training_data['mi']) >= 10:
            return self._train_model(len(erd_indices))
        else:
            print("Insufficient training data")
            return None
    
    def train_from_livestream(self, duration=300, ip=None, port=None):
        """Train from livestream with real-time annotation detection"""
        print("Starting livestream training mode...")
        print(f"Duration: {duration} seconds")
        print("Annotations to collect:")
        print(f"  REST: {', '.join(self.rest_annotations)}")
        print(f"  MI: {', '.join(self.mi_annotations)}")
        print("-" * 60)
        
        # Initialize receiver
        if ip and port:
            receiver = livestream_receiver.LivestreamReceiver(
                address=ip, port=port, broadcast=False
            )
        else:
            print("Using virtual receiver for testing")
            receiver = virtual_receiver.Emulator(verbose=False)
        
        # Initialize connection
        fs, ch_names, n_channels, _ = receiver.initialize_connection()
        self.fs = fs
        self.window_samples = int(self.window_duration * self.fs)
        
        # Find ERD channels
        erd_indices = self._find_erd_channels(ch_names)
        print(f"ERD channels: {[ch_names[i] for i in erd_indices]}")
        
        # Create CSP+SVM detector
        detector = CSPSVMDetector(fs=fs, n_channels=len(erd_indices), 
                                 use_multiband=self.multiband)
        
        # Data buffer
        buffer = []
        last_annotation_time = 0
        start_time = time.time()
        
        print("\nCollecting training data...")
        print("Press Ctrl+C to stop and train model\n")
        
        try:
            while (time.time() - start_time) < duration:
                # Get data
                data = receiver.get_data()
                if data is None:
                    continue
                
                # Add to buffer
                for i in range(data.shape[1]):
                    buffer.append(data[:, i])
                
                # Check for annotations if virtual receiver
                if hasattr(receiver, 'annotation_onsets') and hasattr(receiver, 'current_index'):
                    current_time = receiver.current_index / fs
                    
                    # Process new annotations
                    for i, onset in enumerate(receiver.annotation_onsets):
                        if last_annotation_time < onset <= current_time:
                            desc = receiver.annotation_descriptions[i]
                            
                            # Check if we have enough data after annotation
                            offset_samples = int((onset + self.annotation_offset) * fs)
                            if offset_samples + self.window_samples <= len(buffer):
                                # Extract window
                                window_start = offset_samples
                                window_end = window_start + self.window_samples
                                window_data = np.array(buffer[window_start:window_end]).T
                                
                                # Get ERD channels
                                erd_window = window_data[erd_indices, :]
                                
                                # Classify annotation and collect
                                if any(ann in desc.lower() for ann in self.rest_annotations):
                                    detector.collect_training_data(erd_window, 0)
                                    self.training_data['rest'].append(erd_window)
                                    print(f"✓ REST collected at {onset:.1f}s ('{desc}')")
                                elif any(ann in desc.lower() for ann in self.mi_annotations):
                                    detector.collect_training_data(erd_window, 1)
                                    self.training_data['mi'].append(erd_window)
                                    print(f"✓ MI collected at {onset:.1f}s ('{desc}')")
                    
                    last_annotation_time = current_time
                
                # Trim buffer to prevent memory issues
                if len(buffer) > fs * 60:  # Keep last 60 seconds
                    buffer = buffer[-fs * 60:]
                
                # Show progress
                rest_count = len(detector.training_data['rest'])
                mi_count = len(detector.training_data['mi'])
                if (rest_count + mi_count) % 10 == 0 and (rest_count + mi_count) > 0:
                    print(f"Progress: REST={rest_count}, MI={mi_count}")
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        
        # Disconnect
        receiver.disconnect()
        
        # Train model
        print("\nTraining CSP+SVM model...")
        if detector.train():
            print("Model trained successfully!")
            return detector
        else:
            print("Training failed - insufficient data")
            return None
    
    def _find_erd_channels(self, channel_names):
        """Find ERD-relevant channel indices"""
        # Try to get ERD channels from config, with fallback
        try:
            erd_channels = BCIConfig.ERD_CHANNELS
            print(BCIConfig.ERD_CHANNELS)
        except:
            # Fallback if BCIConfig not available
            erd_channels = ['C3','FC3', 'CP3', 'C1', 'C5', 'FC5', 'FC1', 'CP5', 'CP1']
        
        indices = []

        print(len(channel_names))
        
        # First try exact matches
        for target in erd_channels:
            print(target)
            for i, ch_name in enumerate(channel_names):
                print(f"{i}:{ch_name}")
                if target == ch_name:
                    indices.append(i)
                    print(f"Appended {target}")
                    break
        
        # If still not enough, use first N channels
        if len(indices) == 0:
            print("Warning: No motor channels found, using first 9 channels")
            indices = list(range(min(9, len(channel_names))))
        
        print(indices)

        return indices
    
    def _process_annotations(self, data, annotations, erd_indices, sfreq):
        """Process annotations and extract training windows"""
        rest_collected = 0
        mi_collected = 0
        skipped = 0
        
        for ann in annotations:
            onset_sample = int(ann['onset'] * sfreq)
            desc = ann['description']
            
            # Add offset
            window_start = onset_sample + int(self.annotation_offset * sfreq)
            window_end = window_start + self.window_samples
            
            # Check bounds
            if window_end > data.shape[1]:
                skipped += 1
                continue
            
            # Extract window
            window_data = data[erd_indices, window_start:window_end]
            
            # Check for exact matches first (case-insensitive)
            desc_lower = desc.lower()
            desc_stripped = desc.strip()
            
            # Then check for partial matches
            is_rest = False
            is_mi = False
            
            # Check REST annotations
            for rest_ann in self.rest_annotations:
                if (rest_ann.lower() == desc_lower or 
                    rest_ann in desc or 
                    desc in rest_ann or
                    rest_ann.lower() in desc_lower):
                    is_rest = True
                    break
            
            # Check MI annotations
            if not is_rest:
                for mi_ann in self.mi_annotations:
                    if (mi_ann.lower() == desc_lower or 
                        mi_ann in desc or 
                        desc in mi_ann or
                        mi_ann.lower() in desc_lower):
                        is_mi = True
                        break
            
            # Collect data
            if is_rest:
                self.training_data['rest'].append(window_data)
                rest_collected += 1
                if self.verbose:
                    print(f"REST: '{desc}' at {ann['onset']:.1f}s")
            elif is_mi:
                self.training_data['mi'].append(window_data)
                mi_collected += 1
                if self.verbose:
                    print(f"MI: '{desc}' at {ann['onset']:.1f}s")
            else:
                if self.verbose:
                    print(f"SKIPPED: '{desc}' at {ann['onset']:.1f}s (unrecognized)")
        
        if skipped > 0:
            print(f"\nSkipped {skipped} annotations due to boundary issues")
    
    def _train_model(self, n_channels):
        """Train CSP+SVM model with collected data"""
        print(f"\nTraining with {len(self.training_data['rest'])} REST "
              f"and {len(self.training_data['mi'])} MI windows")
        
        # Create detector
        detector = CSPSVMDetector(fs=self.fs, n_channels=n_channels, 
                                 use_multiband=self.multiband)
        
        # Add training data
        for window in self.training_data['rest']:
            detector.collect_training_data(window, 0)
        for window in self.training_data['mi']:
            detector.collect_training_data(window, 1)
        
        # Train
        if detector.train():
            print("Training successful!")
            self._analyze_multiband_performance(detector)
            return detector
        else:
            print("Training failed")
            return None
    
    def _analyze_multiband_performance(self, detector):
        """Analyze performance across frequency bands"""
        if not self.multiband or not hasattr(detector, 'multiband_scores'):
            return
        
        print("\nMultiband Analysis:")
        print("-" * 40)
        
        # If we have access to individual band performances
        if hasattr(detector, 'band_performances'):
            for band_name, score in detector.band_performances.items():
                print(f"{band_name:12s}: {score:.3f} accuracy")
        
        # Show which bands contribute most
        print("\nRecommendation: Focus on bands with highest accuracy")


class LivestreamAnnotationTrainer:
    """Integrate annotation-based training into livestream system"""
    
    def __init__(self, bci_system):
        self.bci_system = bci_system
        self.enabled = False
        self.annotation_map = {
            'rest': ['S  1', 'S  3', 'rest', 'baseline'],
            'mi': ['S  2', 'S  4', 'motor', 'imagine']
        }
        self.min_samples = 20  # Minimum samples before auto-training
        
    def enable_auto_training(self):
        """Enable automatic training data collection"""
        self.enabled = True
        print("Auto-training enabled. Will collect data from annotations:")
        print(f"  REST: {', '.join(self.annotation_map['rest'])}")
        print(f"  MI: {', '.join(self.annotation_map['mi'])}")
    
    def process_annotation(self, annotation, window_data):
        """Process annotation and collect training data if relevant"""
        if not self.enabled:
            return
        
        if not self.bci_system.erd_detector.csp_svm_detector:
            return
        
        desc = annotation['description']
        detector = self.bci_system.erd_detector.csp_svm_detector
        
        # Check annotation type
        for ann in self.annotation_map['rest']:
            if ann in desc:
                detector.collect_training_data(window_data, 0)
                rest_count = len(detector.training_data['rest'])
                print(f"REST training sample collected ({rest_count} total)")
                self._check_auto_train()
                return
        
        for ann in self.annotation_map['mi']:
            if ann in desc:
                detector.collect_training_data(window_data, 1)
                mi_count = len(detector.training_data['mi'])
                print(f"MI training sample collected ({mi_count} total)")
                self._check_auto_train()
                return
    
    def _check_auto_train(self):
        """Check if we have enough data to auto-train"""
        detector = self.bci_system.erd_detector.csp_svm_detector
        rest_count = len(detector.training_data['rest'])
        mi_count = len(detector.training_data['mi'])
        
        if (rest_count >= self.min_samples and mi_count >= self.min_samples 
            and not detector.is_trained):
            print(f"\nAuto-training CSP+SVM with {rest_count} REST and {mi_count} MI samples...")
            if detector.train():
                print("Auto-training successful! CSP+SVM now active.")
            else:
                print("Auto-training failed.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train CSP+SVM from annotated EEG data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train from BrainVision file:
    python train_csp_from_annotations.py --file recording.vhdr --save model.pkl
    
  Train with custom annotations:
    python train_csp_from_annotations.py --file data.vhdr \\
        --rest-annotations "Rest" "Baseline" \\
        --mi-annotations "Movement" "Imagery"
        
  Debug annotations in file:
    python train_csp_from_annotations.py --file data.vhdr --debug-annotations
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['file', 'livestream'], default='file',
                       help="Training mode (default: file)")
    
    # File mode options
    parser.add_argument('--file', type=str,
                       help="EEG file with annotations (.vhdr, .fif, or .edf)")
    
    # Livestream mode options  
    parser.add_argument('--duration', type=int, default=300,
                       help="Livestream training duration in seconds (default: 300)")
    parser.add_argument('--ip', type=str,
                       help="Livestream IP address")
    parser.add_argument('--port', type=int,
                       help="Livestream port")
    
    # Training options
    parser.add_argument('--multiband', action='store_true', default=True,
                       help="Use multiband CSP (default: True)")
    parser.add_argument('--no-multiband', dest='multiband', action='store_false',
                       help="Disable multiband CSP")
    parser.add_argument('--save', type=str,
                       help="Save trained model to file")
    
    # Window parameters
    parser.add_argument('--window-duration', type=float, default=1.0,
                       help="Window duration in seconds (default: 1.0)")
    parser.add_argument('--offset', type=float, default=0.5,
                       help="Offset after annotation in seconds (default: 0.5)")
    
    # Annotation configuration
    parser.add_argument('--rest-annotations', nargs='+',
                       help="REST annotation markers to look for")
    parser.add_argument('--mi-annotations', nargs='+',
                       help="MI annotation markers to look for")
    parser.add_argument('--clear-defaults', action='store_true',
                       help="Clear default annotation mappings")
    
    # Debug options
    parser.add_argument('--debug-annotations', action='store_true',
                       help="Only show annotations in file without training")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AnnotationBasedTrainer(multiband=args.multiband, verbose=args.verbose)
    
    # Update window parameters
    trainer.window_duration = args.window_duration
    trainer.window_samples = int(trainer.window_duration * trainer.fs)
    trainer.annotation_offset = args.offset
    
    # Handle annotation configuration
    if args.clear_defaults:
        trainer.rest_annotations = []
        trainer.mi_annotations = []
    
    if args.rest_annotations:
        if args.clear_defaults:
            trainer.rest_annotations = args.rest_annotations
        else:
            trainer.rest_annotations.extend(args.rest_annotations)
    
    if args.mi_annotations:
        if args.clear_defaults:
            trainer.mi_annotations = args.mi_annotations
        else:
            trainer.mi_annotations.extend(args.mi_annotations)
    
    # Debug mode - just show annotations
    if args.debug_annotations and args.file:
        debug_annotations_in_file(args.file)
        return
    
    # Train based on mode
    if args.mode == 'file':
        if not args.file:
            print("Error: --file required for file mode")
            parser.print_help()
            return
        
        detector = trainer.train_from_file(args.file)
        
    else:  # livestream mode
        detector = trainer.train_from_livestream(
            duration=args.duration,
            ip=args.ip,
            port=args.port
        )
    
    # Save model if requested
    if detector and args.save:
        detector.save_model(args.save)
        print(f"\nModel saved to {args.save}")
        
        # Print usage instructions
        print("\nTo use this model:")
        print("1. Start BCI system: python bci_system_complete.py --csp-svm")
        print("2. Press 'l' to load model")
        print(f"3. Enter filename: {args.save}")


def debug_annotations_in_file(filename):
    """Debug function to show all annotations in a file"""
    print(f"Debugging annotations in {filename}")
    print("="*60)
    
    try:
        # Load file
        if filename.endswith('.vhdr'):
            raw = mne.io.read_raw_brainvision(filename, preload=False)
        elif filename.endswith('.fif'):
            raw = mne.io.read_raw_fif(filename, preload=False)
        elif filename.endswith('.edf'):
            raw = mne.io.read_raw_edf(filename, preload=False)
        else:
            print("Unsupported file format")
            return
        
        # Get annotations
        annotations = raw.annotations
        
        print(f"File info:")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")
        print(f"\nTotal annotations: {len(annotations)}")
        
        # Count types
        annotation_counts = defaultdict(int)
        for ann in annotations:
            annotation_counts[ann['description']] += 1
        
        print("\nAnnotation types:")
        for desc, count in sorted(annotation_counts.items()):
            print(f"  '{desc}': {count} occurrences")
        
        # Show first 10 annotations with timing
        print("\nFirst 10 annotations:")
        for i, ann in enumerate(annotations[:10]):
            print(f"  {i+1}. t={ann['onset']:.3f}s: '{ann['description']}'")
        
        if len(annotations) > 10:
            print(f"  ... and {len(annotations)-10} more")
            
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    main()