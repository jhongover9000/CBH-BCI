#!/usr/bin/env python3
'''
test_annotation_correlation.py
-----------------------------
Test script to analyze correlation between annotations and ERD detections
Checks for ERD detections within configurable time windows of annotation onsets
'''

import numpy as np
import time
import argparse
import json
from collections import defaultdict
from datetime import datetime

# Import BCI system
from bci_erd_main import BCISystem, BCIConfig


class AnnotationCorrelationTester:
    """Test annotation-ERD correlation with configurable parameters"""
    
    def __init__(self, time_window=2.0, annotation_types=None):
        self.time_window = time_window  # seconds to check after annotation
        self.annotation_types = annotation_types or []  # specific types to analyze
        
        # Results storage
        self.results = {
            'total_annotations': 0,
            'annotations_with_erd': 0,
            'false_positives': 0,
            'true_positives': 0,
            'response_times': [],
            'by_type': defaultdict(lambda: {
                'count': 0,
                'detected': 0,
                'response_times': []
            })
        }
        
        # Detection tracking
        self.all_annotations = []
        self.all_detections = []
        
    def run_test(self, duration=None, save_results=None):
        """Run the correlation test"""
        print(f"Annotation Correlation Test")
        print("="*60)
        print(f"Time window: {self.time_window}s after annotation")
        print(f"Annotation types: {self.annotation_types if self.annotation_types else 'All'}")
        print("="*60 + "\n")
        
        # Configure system
        BCIConfig.VIRTUAL = True  # Use virtual mode
        BCIConfig.GUI_ENABLED = False  # No GUI
        BCIConfig.VERBOSE = False
        BCIConfig.SHOW_ANNOTATIONS = True
        
        if duration:
            BCIConfig.SESSION_DURATION = duration
        
        # Create and initialize system
        system = BCISystem()
        system.initialize()
        
        # Hook into the system to track detections
        self._hook_system(system)
        
        # Run the system
        print("Running detection system...")
        system.run()
        
        # Analyze results
        self._analyze_results(system)
        
        # Save results if requested
        if save_results:
            self._save_results(save_results)
        
        return self.results
    
    def _hook_system(self, system):
        """Hook into system to track detections"""
        # Store original methods
        original_update = system._update_display
        original_check_annotations = system._check_for_annotations
        
        # Track annotations
        def hooked_check_annotations():
            original_check_annotations()
            # Store annotations for analysis
            if hasattr(system, 'annotations'):
                self.all_annotations = system.annotations.copy()
        
        # Track detections
        def hooked_update(detected, erd_values, confidence):
            runtime = time.time() - system.session_start_time
            if BCIConfig.VIRTUAL and hasattr(system.receiver, 'current_index'):
                runtime = system.receiver.current_index / system.fs
            
            if detected:
                self.all_detections.append({
                    'time': runtime,
                    'erd_values': erd_values.copy(),
                    'confidence': confidence
                })
            
            # Call original
            original_update(detected, erd_values, confidence)
        
        # Replace methods
        system._check_for_annotations = hooked_check_annotations
        system._update_display = hooked_update
    
    def _analyze_results(self, system):
        """Analyze correlation between annotations and detections"""
        print("\n\nAnalyzing results...")
        print("-"*60)
        
        # Filter annotations by type if specified
        annotations_to_analyze = []
        for ann in self.all_annotations:
            if not self.annotation_types or any(t in ann['description'] for t in self.annotation_types):
                annotations_to_analyze.append(ann)
        
        self.results['total_annotations'] = len(annotations_to_analyze)
        
        # Check each annotation
        for ann in annotations_to_analyze:
            ann_time = ann['time']
            ann_type = ann['description'].split('/')[-1] if '/' in ann['description'] else ann['description']
            
            # Find detections within time window
            detections_in_window = []
            for det in self.all_detections:
                if ann_time <= det['time'] <= ann_time + self.time_window:
                    detections_in_window.append(det)
            
            # Update results
            self.results['by_type'][ann_type]['count'] += 1
            
            if detections_in_window:
                self.results['annotations_with_erd'] += 1
                self.results['by_type'][ann_type]['detected'] += 1
                
                # Calculate response time (first detection)
                response_time = detections_in_window[0]['time'] - ann_time
                self.results['response_times'].append(response_time)
                self.results['by_type'][ann_type]['response_times'].append(response_time)
                
                # True positive
                self.results['true_positives'] += 1
                
                # Print detailed info
                print(f"✓ {ann_type} at {ann_time:.3f}s → ERD at {detections_in_window[0]['time']:.3f}s "
                      f"(Δ={response_time:.3f}s, conf={detections_in_window[0]['confidence']:.1f}%)")
            else:
                print(f"✗ {ann_type} at {ann_time:.3f}s → No ERD within {self.time_window}s")
        
        # Count false positives (detections not near annotations)
        for det in self.all_detections:
            near_annotation = False
            for ann in annotations_to_analyze:
                if ann['time'] - 1.0 <= det['time'] <= ann['time'] + self.time_window:
                    near_annotation = True
                    break
            
            if not near_annotation:
                self.results['false_positives'] += 1
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate performance statistics"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        # Overall statistics
        if self.results['total_annotations'] > 0:
            detection_rate = (self.results['annotations_with_erd'] / 
                            self.results['total_annotations'] * 100)
        else:
            detection_rate = 0
        
        print(f"\nOverall Performance:")
        print(f"  Total annotations analyzed: {self.results['total_annotations']}")
        print(f"  Annotations with ERD:       {self.results['annotations_with_erd']} ({detection_rate:.1f}%)")
        print(f"  False positives:            {self.results['false_positives']}")
        
        if self.results['response_times']:
            avg_response = np.mean(self.results['response_times'])
            std_response = np.std(self.results['response_times'])
            min_response = np.min(self.results['response_times'])
            max_response = np.max(self.results['response_times'])
            
            print(f"\nResponse Times:")
            print(f"  Average: {avg_response:.3f}s ± {std_response:.3f}s")
            print(f"  Range:   {min_response:.3f}s - {max_response:.3f}s")
        
        # By annotation type
        if len(self.results['by_type']) > 0:
            print(f"\nBy Annotation Type:")
            for ann_type, stats in self.results['by_type'].items():
                if stats['count'] > 0:
                    type_rate = stats['detected'] / stats['count'] * 100
                    print(f"\n  {ann_type}:")
                    print(f"    Count:     {stats['count']}")
                    print(f"    Detected:  {stats['detected']} ({type_rate:.1f}%)")
                    
                    if stats['response_times']:
                        avg_rt = np.mean(stats['response_times'])
                        print(f"    Avg RT:    {avg_rt:.3f}s")
        
        # Performance metrics
        if self.results['total_annotations'] > 0:
            sensitivity = self.results['true_positives'] / self.results['total_annotations']
            total_detections = self.results['true_positives'] + self.results['false_positives']
            
            if total_detections > 0:
                precision = self.results['true_positives'] / total_detections
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
                
                print(f"\nPerformance Metrics:")
                print(f"  Sensitivity: {sensitivity:.3f}")
                print(f"  Precision:   {precision:.3f}")
                print(f"  F1 Score:    {f1_score:.3f}")
        
        print("="*60)
    
    def _save_results(self, filename):
        """Save results to JSON file"""
        # Convert defaultdict to regular dict for JSON
        results_to_save = dict(self.results)
        results_to_save['by_type'] = dict(results_to_save['by_type'])
        
        # Add metadata
        results_to_save['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'time_window': self.time_window,
            'annotation_types': self.annotation_types,
            'config': {
                'threshold': BCIConfig.ERD_THRESHOLD,
                'window_duration': BCIConfig.OPERATING_WINDOW_DURATION,
                'overlap': BCIConfig.WINDOW_OVERLAP,
                'use_ma': BCIConfig.USE_MOVING_AVERAGE,
                'sliding_baseline': BCIConfig.SLIDING_BASELINE
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test annotation-ERD correlation"
    )
    
    parser.add_argument('--window', type=float, default=2.0,
                       help="Time window after annotation (seconds)")
    parser.add_argument('--types', nargs='+',
                       help="Specific annotation types to analyze")
    parser.add_argument('--duration', type=int,
                       help="Test duration in seconds")
    parser.add_argument('--threshold', type=float,
                       help="ERD detection threshold")
    parser.add_argument('--save', type=str,
                       help="Save results to JSON file")
    parser.add_argument('--no-ma', action='store_true',
                       help="Disable moving average")
    parser.add_argument('--no-sliding', action='store_true',
                       help="Disable sliding baseline")
    
    args = parser.parse_args()
    
    # Configure system if specified
    if args.threshold:
        BCIConfig.ERD_THRESHOLD = args.threshold
    if args.no_ma:
        BCIConfig.USE_MOVING_AVERAGE = False
    if args.no_sliding:
        BCIConfig.SLIDING_BASELINE = False
    
    # Create tester
    tester = AnnotationCorrelationTester(
        time_window=args.window,
        annotation_types=args.types
    )
    
    # Run test
    results = tester.run_test(
        duration=args.duration,
        save_results=args.save
    )


if __name__ == "__main__":
    main()