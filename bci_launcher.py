#!/usr/bin/env python3
'''
launch_bci_system.py
-------------------
Quick launcher for the integrated BCI ERD system
'''

import sys
import os

# Add the current directory to Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_bci_erd_system import main

if __name__ == "__main__":
    print("=" * 70)
    print("BCI ERD Detection System - Integrated Edition")
    print("=" * 70)
    print("\nFeatures:")
    print("- Simple and Advanced (CSP+SVM) detection modes")
    print("- GUI controls for all functions")
    print("- Real-time ERD trend plotting with annotations")
    print("- Model saving/loading")
    print("- Live metrics display")
    print("- Training data collection")
    print("- Annotation correlation analysis")
    print("\nStarting GUI...\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()