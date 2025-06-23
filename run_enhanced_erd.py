#!/usr/bin/env python3
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
