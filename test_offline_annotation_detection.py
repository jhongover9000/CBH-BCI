#!/usr/bin/env python3
'''
test_annotation_detection_combined.py
------------------------------------
Combined script for testing annotation-detection correlation
Works with both virtual/live mode and offline file playback
'''

import numpy as np
import time
import argparse
import json
import mne
import os
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

# Import BCI system
try:
    from bci_erd_main import BCISystem, BCIConfig
except ImportError:
    # Fallback for older installations
    from bci_erd_main import BCISystem, BCIConfig
    print("Note: Using bci_erd_main (older module name)")

# Import receivers for file playback
from receivers import virtual_receiver


class AnnotationDetectionTester:
    """Test annotation-detection correlation with REST/MI classification"""
    
    def __init__(self, pre_window=0.5, post_window=1.5, verbose=False):
        self.pre_window = pre_window  # seconds before annotation
        self.post_window = post_window  # seconds after annotation
        self.verbose = verbose
        
        # Annotation mappings
        self.rest_annotations = [
            'Stimulus/S  2', 'S  2', 'S2', 'S 2',
            'Stimulus/S  4', 'S  4', 'S4', 'S 4',
            'rest', 'Rest', 'REST',
            'baseline', 'Baseline', 'BASELINE',
            'relax', 'Relax', 'RELAX',
            'Response/R  2', 'R  2', 'R2'
        ]
        
        self.mi_annotations = [
            'Stimulus/S  3', 'S  3', 'S3', 'S 3',
            'Stimulus/S  1', 'S  1', 'S1', 'S 1',
            'motor', 'Motor', 'MOTOR',
            'imagery', 'Imagery', 'IMAGERY',
            'movement', 'Movement', 'MOVEMENT',
            'imagine', 'Imagine', 'IMAGINE',
            'mi', 'MI', 'Motor Imagery',
            'Response/R  1', 'R  1', 'R1'
        ]
        
        # Results storage
        self.results = {
            'rest': {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'specificity': 0,
                'details': []
            },
            'mi': {
                'total': 0,
                'correct': 0,
                'missed': 0,
                'sensitivity': 0,
                'response_times': [],
                'details': []
            },
            'unclassified': {
                'total': 0,
                'with_detection': 0,
                'annotations': []
            },
            'overall': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'detections_total': 0
            }
        }
        
        # Detection tracking
        self.all_annotations = []
        self.all_detections = []
        
    def run_virtual_test(self, duration=None, save_results=None):
        """Run test using virtual receiver (original method)"""
        print("Annotation Detection Test (Virtual Mode)")
        print("="*60)
        print(f"Detection window: -{self.pre_window}s to +{self.post_window}s")
        print("="*60 + "\n")
        
        # Configure system
        BCIConfig.VIRTUAL = True
        BCIConfig.GUI_ENABLED = False
        BCIConfig.VERBOSE = False
        BCIConfig.SHOW_ANNOTATIONS = True
        
        if duration:
            BCIConfig.SESSION_DURATION = duration
        
        # Create and initialize system
        system = BCISystem()
        system.initialize()
        
        # Hook into the system
        self._hook_system(system)
        
        # Run the system
        print("Running detection system...")
        system.run()
        
        # Analyze results
        self._analyze_results()
        
        # Save results if requested
        if save_results:
            self._save_results(save_results)
        
        return self.results
    
    def run_file_test(self, filename, duration=None, save_results=None, 
                      csp_svm_model=None):
        """Run test on recorded file by playing it through the system"""
        print(f"Annotation Detection Test (File Mode: {filename})")
        print("="*60)
        print(f"Detection window: -{self.pre_window}s to +{self.post_window}s")
        
        # First, load file to get annotations
        raw = self._load_file_annotations(filename)
        if raw is None:
            return None
            
        print("="*60 + "\n")
        
        # Configure system for file playback
        BCIConfig.VIRTUAL = True
        BCIConfig.GUI_ENABLED = False
        BCIConfig.VERBOSE = False
        BCIConfig.SHOW_ANNOTATIONS = False  # We'll handle annotations
        
        # Set duration to file length if not specified
        if duration is None:
            duration = raw.times[-1]
        BCIConfig.SESSION_DURATION = min(duration, raw.times[-1])
        
        # Create custom virtual receiver that plays the file
        # This is a workaround to feed file data through the system
        print("Creating file playback receiver...")
        
        # Modify virtual receiver to use file data
        original_receiver = virtual_receiver.Emulator
        
        class FilePlaybackReceiver(original_receiver):
            def __init__(self, file_data, file_annotations, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.file_data = file_data
                self.file_annotations = file_annotations
                self.file_index = 0
                
            def initialize_connection(self):
                """Override to use file parameters"""
                result = super().initialize_connection()
                # Update with file info
                self.fs = int(self.file_data['sfreq'])
                self.ch_names = self.file_data['ch_names']
                self.n_channels = len(self.ch_names)
                
                # Convert annotations to virtual receiver format
                self.annotation_onsets = []
                self.annotation_descriptions = []
                for ann in self.file_annotations:
                    self.annotation_onsets.append(ann['onset'])
                    self.annotation_descriptions.append(ann['description'])
                
                return self.fs, self.ch_names, self.n_channels, None
                
            def get_data(self):
                """Override to return file data"""
                if self.file_index >= self.file_data['data'].shape[1]:
                    return None
                    
                # Get chunk of data
                chunk_size = int(self.chunk_size * self.fs)
                end_index = min(self.file_index + chunk_size, 
                              self.file_data['data'].shape[1])
                
                data = self.file_data['data'][:, self.file_index:end_index]
                self.file_index = end_index
                self.current_index = self.file_index
                
                return data
        
        # Temporarily replace virtual receiver
        virtual_receiver.Emulator = lambda **kwargs: FilePlaybackReceiver(
            file_data={
                'data': raw.get_data(),
                'sfreq': raw.info['sfreq'],
                'ch_names': raw.ch_names
            },
            file_annotations=raw.annotations,
            **kwargs
        )
        
        try:
            # Create and initialize system
            system = BCISystem()
            system.initialize()
            
            # Load CSP+SVM model if provided
            if csp_svm_model and BCIConfig.USE_CSP_SVM:
                if hasattr(system, 'erd_detector') and hasattr(system.erd_detector, 'csp_svm_detector'):
                    try:
                        system.erd_detector.csp_svm_detector.load_model(csp_svm_model)
                        print(f"Loaded CSP+SVM model: {csp_svm_model}")
                    except Exception as e:
                        print(f"Warning: Could not load CSP+SVM model: {e}")
            
            # Hook into the system
            self._hook_system(system)
            
            # Run the system
            print("Processing file through detection system...")
            system.run()
            
            # Analyze results
            self._analyze_results()
            
            # Save results if requested
            if save_results:
                self._save_results(save_results)
                
        finally:
            # Restore original receiver
            virtual_receiver.Emulator = original_receiver
        
        return self.results
    
    def _load_file_annotations(self, filename):
        """Load file and extract annotations"""
        try:
            if filename.endswith('.vhdr'):
                raw = mne.io.read_raw_brainvision(filename, preload=False)
            elif filename.endswith('.fif'):
                raw = mne.io.read_raw_fif(filename, preload=False)
            elif filename.endswith('.edf'):
                raw = mne.io.read_raw_edf(filename, preload=False)
            else:
                print(f"Unsupported file format: {filename}")
                return None
                
            print(f"Loaded file: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
            print(f"Found {len(raw.annotations)} annotations")
            
            # Store annotations for later use
            self.file_annotations = []
            for ann in raw.annotations:
                self.file_annotations.append({
                    'time': ann['onset'],
                    'description': ann['description']
                })
                
            return raw
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def _hook_system(self, system):
        """Hook into system to track detections and annotations"""
        # Store original methods
        original_update = system._update_display
        
        # For file mode, we need to inject annotations
        if hasattr(self, 'file_annotations'):
            system.annotations = self.file_annotations
            self.all_annotations = self.file_annotations.copy()
        else:
            # For virtual mode, track annotations as they come
            original_check_annotations = system._check_for_annotations
            
            def hooked_check_annotations():
                original_check_annotations()
                if hasattr(system, 'annotations'):
                    self.all_annotations = system.annotations.copy()
            
            system._check_for_annotations = hooked_check_annotations
        
        # Track detections
        def hooked_update(detected, erd_values, confidence):
            runtime = time.time() - system.session_start_time
            
            # Get accurate time for virtual/file mode
            if hasattr(system.receiver, 'current_index'):
                runtime = system.receiver.current_index / system.fs
            
            # Store all detections with their details
            self.all_detections.append({
                'time': runtime,
                'detected': detected,
                'erd_values': erd_values.copy() if erd_values is not None else None,
                'confidence': confidence
            })
            
            # Call original
            original_update(detected, erd_values, confidence)
        
        system._update_display = hooked_update
    
    def _classify_annotation(self, description):
        """Classify annotation as rest, mi, or neither"""
        desc_lower = description.lower()
        
        # Check REST
        for rest_ann in self.rest_annotations:
            if (rest_ann.lower() == desc_lower or 
                rest_ann in description or 
                rest_ann.lower() in desc_lower):
                return 'rest'
                
        # Check MI
        for mi_ann in self.mi_annotations:
            if (mi_ann.lower() == desc_lower or 
                mi_ann in description or 
                mi_ann.lower() in desc_lower):
                return 'mi'
                
        return None
    
    def _analyze_results(self):
        """Analyze correlation between annotations and detections"""
        print("\n\nAnalyzing results...")
        print("-"*60)
        
        # Process each annotation
        for ann in self.all_annotations:
            ann_time = ann['time']
            ann_desc = ann['description']
            ann_type = self._classify_annotation(ann_desc)
            
            # Find detections in window
            window_start = ann_time - self.pre_window
            window_end = ann_time + self.post_window
            
            detections_in_window = []
            for det in self.all_detections:
                if window_start <= det['time'] <= window_end and det['detected']:
                    detections_in_window.append(det)
            
            # Process based on annotation type
            if ann_type == 'mi':
                self._process_mi_annotation(ann, detections_in_window)
            elif ann_type == 'rest':
                self._process_rest_annotation(ann, detections_in_window)
            else:
                # Unclassified annotation
                self.results['unclassified']['total'] += 1
                if detections_in_window:
                    self.results['unclassified']['with_detection'] += 1
                self.results['unclassified']['annotations'].append(ann_desc)
                
                if self.verbose:
                    print(f"? {ann_desc} at {ann_time:.3f}s (unclassified)")
        
        # Count total detections
        self.results['overall']['detections_total'] = sum(
            1 for det in self.all_detections if det['detected']
        )
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _process_mi_annotation(self, ann, detections):
        """Process MI annotation"""
        self.results['mi']['total'] += 1
        
        if detections:
            # Correct detection
            self.results['mi']['correct'] += 1
            self.results['overall']['true_positives'] += 1
            
            # Calculate response time
            response_time = detections[0]['time'] - ann['time']
            self.results['mi']['response_times'].append(response_time)
            
            # Store details
            self.results['mi']['details'].append({
                'annotation': ann,
                'detection': detections[0],
                'response_time': response_time,
                'result': 'correct'
            })
            
            if self.verbose:
                print(f"✓ MI at {ann['time']:.3f}s → Detected at {detections[0]['time']:.3f}s "
                      f"(RT={response_time:.3f}s, conf={detections[0]['confidence']:.1f}%)")
        else:
            # Missed detection
            self.results['mi']['missed'] += 1
            self.results['overall']['false_negatives'] += 1
            
            self.results['mi']['details'].append({
                'annotation': ann,
                'result': 'missed'
            })
            
            if self.verbose:
                print(f"✗ MI at {ann['time']:.3f}s → Not detected")
    
    def _process_rest_annotation(self, ann, detections):
        """Process REST annotation"""
        self.results['rest']['total'] += 1
        
        if not detections:
            # Correct non-detection
            self.results['rest']['correct'] += 1
            self.results['overall']['true_negatives'] += 1
            
            self.results['rest']['details'].append({
                'annotation': ann,
                'result': 'correct'
            })
            
            if self.verbose:
                print(f"✓ REST at {ann['time']:.3f}s → No detection (correct)")
        else:
            # False positive
            self.results['rest']['incorrect'] += 1
            self.results['overall']['false_positives'] += 1
            
            self.results['rest']['details'].append({
                'annotation': ann,
                'detections': detections,
                'result': 'incorrect'
            })
            
            if self.verbose:
                print(f"✗ REST at {ann['time']:.3f}s → False detection(s): {len(detections)}")
    
    def _calculate_statistics(self):
        """Calculate performance statistics"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        # Overall metrics
        total_annotations = self.results['mi']['total'] + self.results['rest']['total']
        if total_annotations > 0:
            total_correct = self.results['mi']['correct'] + self.results['rest']['correct']
            accuracy = total_correct / total_annotations
            
            print(f"\nOverall Performance:")
            print(f"  Total annotations: {total_annotations}")
            print(f"  Overall accuracy: {accuracy:.1%}")
            print(f"  Total detections: {self.results['overall']['detections_total']}")
        
        # MI results
        if self.results['mi']['total'] > 0:
            sensitivity = self.results['mi']['correct'] / self.results['mi']['total']
            self.results['mi']['sensitivity'] = sensitivity
            
            print(f"\nMotor Imagery (MI) Results:")
            print(f"  Total: {self.results['mi']['total']}")
            print(f"  Correct: {self.results['mi']['correct']} ({sensitivity:.1%})")
            print(f"  Missed: {self.results['mi']['missed']}")
            
            if self.results['mi']['response_times']:
                avg_rt = np.mean(self.results['mi']['response_times'])
                std_rt = np.std(self.results['mi']['response_times'])
                print(f"  Avg response time: {avg_rt:.3f}s ± {std_rt:.3f}s")
        
        # REST results
        if self.results['rest']['total'] > 0:
            specificity = self.results['rest']['correct'] / self.results['rest']['total']
            self.results['rest']['specificity'] = specificity
            
            print(f"\nRest Results:")
            print(f"  Total: {self.results['rest']['total']}")
            print(f"  Correct: {self.results['rest']['correct']} ({specificity:.1%})")
            print(f"  False positives: {self.results['rest']['incorrect']}")
        
        # Unclassified
        if self.results['unclassified']['total'] > 0:
            print(f"\nUnclassified Annotations:")
            print(f"  Total: {self.results['unclassified']['total']}")
            print(f"  With detection: {self.results['unclassified']['with_detection']}")
            
            # Show unique unclassified types
            unique_types = list(set(self.results['unclassified']['annotations']))
            print(f"  Types: {', '.join(unique_types[:5])}")
            if len(unique_types) > 5:
                print(f"         ... and {len(unique_types)-5} more")
        
        # Performance metrics
        tp = self.results['overall']['true_positives']
        fp = self.results['overall']['false_positives']
        tn = self.results['overall']['true_negatives']
        fn = self.results['overall']['false_negatives']
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        
        # Store in results
        self.results['overall']['accuracy'] = accuracy if total_annotations > 0 else 0
        self.results['overall']['precision'] = precision
        self.results['overall']['recall'] = recall
        self.results['overall']['f1_score'] = f1
        
        print("="*60)
    
    def _save_results(self, filename):
        """Save results to JSON file"""
        # Prepare results for JSON
        results_to_save = {
            'rest': self.results['rest'],
            'mi': self.results['mi'],
            'unclassified': {
                'total': self.results['unclassified']['total'],
                'with_detection': self.results['unclassified']['with_detection']
            },
            'overall': self.results['overall'],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'pre_window': self.pre_window,
                'post_window': self.post_window,
                'config': {
                    'threshold': BCIConfig.ERD_THRESHOLD,
                    'window_duration': BCIConfig.OPERATING_WINDOW_DURATION,
                    'overlap': BCIConfig.WINDOW_OVERLAP,
                    'use_ma': BCIConfig.USE_MOVING_AVERAGE,
                    'sliding_baseline': BCIConfig.SLIDING_BASELINE,
                    'use_csp_svm': BCIConfig.USE_CSP_SVM
                }
            }
        }
        
        # Remove detailed info for JSON (too large)
        for key in ['rest', 'mi']:
            if 'details' in results_to_save[key]:
                results_to_save[key]['details_count'] = len(results_to_save[key]['details'])
                del results_to_save[key]['details']
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def plot_results(self, save_plot=None):
        """Plot results visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Classification results bar chart
        categories = ['MI Correct', 'MI Missed', 'REST Correct', 'REST False Pos']
        values = [
            self.results['mi']['correct'],
            self.results['mi']['missed'],
            self.results['rest']['correct'],
            self.results['rest']['incorrect']
        ]
        colors = ['green', 'red', 'green', 'red']
        
        ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Count')
        ax1.set_title('Classification Results')
        ax1.grid(True, alpha=0.3)
        
        # 2. Response time histogram
        if self.results['mi']['response_times']:
            ax2.hist(self.results['mi']['response_times'], bins=20, 
                    alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', label='Annotation onset')
            ax2.set_xlabel('Response Time (s)')
            ax2.set_ylabel('Count')
            ax2.set_title('MI Detection Response Times')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No MI detections', ha='center', va='center')
            ax2.set_title('MI Detection Response Times')
        
        # 3. Performance metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [
            self.results['overall'].get('accuracy', 0),
            self.results['overall'].get('precision', 0),
            self.results['overall'].get('recall', 0),
            self.results['overall'].get('f1_score', 0)
        ]
        
        ax3.bar(metrics, values, alpha=0.7, color='purple')
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics')
        ax3.grid(True, alpha=0.3)
        
        # 4. Sensitivity/Specificity
        if self.results['mi']['total'] > 0 or self.results['rest']['total'] > 0:
            labels = []
            sizes = []
            colors_pie = []
            
            if self.results['mi']['total'] > 0:
                labels.extend(['MI Detected', 'MI Missed'])
                sizes.extend([self.results['mi']['correct'], self.results['mi']['missed']])
                colors_pie.extend(['lightgreen', 'lightcoral'])
                
            if self.results['rest']['total'] > 0:
                labels.extend(['REST Correct', 'REST False Pos'])
                sizes.extend([self.results['rest']['correct'], self.results['rest']['incorrect']])
                colors_pie.extend(['darkgreen', 'darkred'])
            
            ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Detection Distribution')
        else:
            ax4.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax4.set_title('Detection Distribution')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=150)
            print(f"Plot saved to {save_plot}")
        else:
            plt.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test annotation-detection correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Virtual mode test:
    python test_annotation_detection_combined.py --duration 300
    
  File mode test:
    python test_annotation_detection_combined.py --file recording.vhdr
    
  With CSP+SVM:
    python test_annotation_detection_combined.py --file recording.vhdr \\
        --csp-svm --model trained_model.pkl
        """
    )
    
    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--file', type=str,
                      help="Test using recorded EEG file")
    group.add_argument('--virtual', action='store_true',
                      help="Test using virtual receiver (default)")
    
    # Timing windows
    parser.add_argument('--pre-window', type=float, default=0.5,
                       help="Time before annotation to check (default: 0.5s)")
    parser.add_argument('--post-window', type=float, default=1.5,
                       help="Time after annotation to check (default: 1.5s)")
    
    # Test parameters
    parser.add_argument('--duration', type=int,
                       help="Test duration in seconds")
    parser.add_argument('--threshold', type=float, default=60,
                       help="ERD detection threshold")
    parser.add_argument('--window', type=float, default=1.0,
                       help="Detection window duration")
    parser.add_argument('--overlap', type=float, default=0.6,
                       help="Window overlap (0-1)")
    
    # CSP+SVM options
    parser.add_argument('--csp-svm', action='store_true',
                       help="Enable CSP+SVM detection")
    parser.add_argument('--model', type=str,
                       help="CSP+SVM model file to load")
    
    # Features
    parser.add_argument('--no-ma', action='store_true',
                       help="Disable moving average")
    parser.add_argument('--no-sliding', action='store_true',
                       help="Disable sliding baseline")
    
    # Annotations
    parser.add_argument('--rest-annotations', nargs='+',
                       help="Additional REST annotation markers")
    parser.add_argument('--mi-annotations', nargs='+',
                       help="Additional MI annotation markers")
    
    # Output
    parser.add_argument('--save', type=str,
                       help="Save results to JSON file")
    parser.add_argument('--plot', action='store_true',
                       help="Show results plot")
    parser.add_argument('--save-plot', type=str,
                       help="Save plot to file")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure system
    BCIConfig.ERD_THRESHOLD = args.threshold
    BCIConfig.OPERATING_WINDOW_DURATION = args.window
    BCIConfig.WINDOW_OVERLAP = args.overlap
    BCIConfig.USE_CSP_SVM = args.csp_svm
    
    if args.no_ma:
        BCIConfig.USE_MOVING_AVERAGE = False
    if args.no_sliding:
        BCIConfig.SLIDING_BASELINE = False
    
    # Create tester
    tester = AnnotationDetectionTester(
        pre_window=args.pre_window,
        post_window=args.post_window,
        verbose=args.verbose
    )
    
    # Add custom annotations if provided
    if args.rest_annotations:
        tester.rest_annotations.extend(args.rest_annotations)
    if args.mi_annotations:
        tester.mi_annotations.extend(args.mi_annotations)
    
    # Run appropriate test
    if args.file:
        # File mode
        results = tester.run_file_test(
            filename=args.file,
            duration=args.duration,
            save_results=args.save,
            csp_svm_model=args.model
        )
    else:
        # Virtual mode (default)
        results = tester.run_virtual_test(
            duration=args.duration,
            save_results=args.save
        )
    
    # Plot if requested
    if results and (args.plot or args.save_plot):
        tester.plot_results(save_plot=args.save_plot)


if __name__ == "__main__":
    main()