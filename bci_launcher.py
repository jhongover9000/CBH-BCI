'''
BCI ERD Launcher
Easy-to-use launcher for the BCI ERD system with presets
'''

import subprocess
import sys
import os


def launch_preset(preset_name):
    """Launch BCI ERD system with predefined settings"""
    
    presets = {
        'virtual_test': [
            '--virtual',
            '--channels', 'C3', 'C4', 'Cz',
            '--threshold', '15',
            '--duration', '60',
            '--verbose'
        ],
        
        'real_basic': [
            '--channels', 'C3', 'C4', 'Cz',
            '--threshold', '20',
            '--broadcast'
        ],
        
        'real_sensitive': [
            '--channels', 'C3', 'C4', 'Cz', 'FC3', 'FC4',
            '--threshold', '10',
            '--adaptation', 'kalman',
            '--broadcast'
        ],
        
        'debug': [
            '--virtual',
            '--verbose',
            '--channels', 'C3', 'C4',
            '--threshold', '15',
            '--duration', '30',
            '--no-gui'
        ],
        
        'performance': [
            '--no-gui',
            '--channels', 'C3', 'C4',
            '--adaptation', 'exponential',
            '--broadcast'
        ]
    }
    
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {', '.join(presets.keys())}")
        return
    
    # Build command
    cmd = [sys.executable, 'bci_erd.py'] + presets[preset_name]
    
    print(f"Launching BCI ERD with preset: {preset_name}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run
    subprocess.run(cmd)


def interactive_setup():
    """Interactive setup wizard"""
    print("BCI ERD System - Interactive Setup")
    print("=" * 60)
    
    # Mode selection
    print("\n1. Select Mode:")
    print("   1) Virtual (testing with recorded data)")
    print("   2) Real (live EEG hardware)")
    mode = input("Enter choice (1-2): ").strip()
    
    args = []
    if mode == '1':
        args.append('--virtual')
    
    # Channel selection
    print("\n2. Select Channels:")
    print("   1) Standard motor (C3, C4, Cz)")
    print("   2) Extended motor (C3, C4, Cz, FC3, FC4)")
    print("   3) Custom")
    ch_choice = input("Enter choice (1-3): ").strip()
    
    if ch_choice == '1':
        args.extend(['--channels', 'C3', 'C4', 'Cz'])
    elif ch_choice == '2':
        args.extend(['--channels', 'C3', 'C4', 'Cz', 'FC3', 'FC4'])
    elif ch_choice == '3':
        channels = input("Enter channel names separated by spaces: ").strip().split()
        args.extend(['--channels'] + channels)
    
    # Threshold
    print("\n3. ERD Threshold:")
    print("   1) High sensitivity (10%)")
    print("   2) Standard (20%)")
    print("   3) Low sensitivity (30%)")
    print("   4) Custom")
    th_choice = input("Enter choice (1-4): ").strip()
    
    thresholds = {'1': '10', '2': '20', '3': '30'}
    if th_choice in thresholds:
        args.extend(['--threshold', thresholds[th_choice]])
    elif th_choice == '4':
        threshold = input("Enter threshold (5-50): ").strip()
        args.extend(['--threshold', threshold])
    
    # Baseline adaptation
    print("\n4. Baseline Adaptation:")
    print("   1) Hybrid (recommended)")
    print("   2) Static (traditional)")
    print("   3) Adaptive only")
    adapt_choice = input("Enter choice (1-3): ").strip()
    
    adaptations = {'1': 'hybrid', '2': 'static', '3': 'exponential'}
    if adapt_choice in adaptations:
        args.extend(['--adaptation', adaptations[adapt_choice]])
    
    # Options
    print("\n5. Additional Options:")
    if input("   Enable GUI monitor? (y/n): ").lower() != 'y':
        args.append('--no-gui')
    if input("   Enable broadcasting? (y/n): ").lower() == 'y':
        args.append('--broadcast')
    if input("   Enable verbose output? (y/n): ").lower() == 'y':
        args.append('--verbose')
    
    # Duration
    duration = input("\n6. Session duration in seconds (default 600): ").strip()
    if duration:
        args.extend(['--duration', duration])
    
    # Launch
    print("\n" + "=" * 60)
    print("Launching BCI ERD System...")
    print("Command:", ' '.join(['python', 'bci_erd.py'] + args))
    print("=" * 60)
    
    subprocess.run([sys.executable, 'bci_erd.py'] + args)


def main():
    """Main launcher menu"""
    print("BCI ERD System Launcher")
    print("=" * 60)
    print("\nOptions:")
    print("1. Quick Start - Virtual Test")
    print("2. Quick Start - Real Hardware")
    print("3. Sensitive Detection Mode")
    print("4. Debug Mode (no GUI)")
    print("5. Performance Mode")
    print("6. Interactive Setup")
    print("7. Exit")
    
    choice = input("\nSelect option (1-7): ").strip()
    
    if choice == '1':
        launch_preset('virtual_test')
    elif choice == '2':
        launch_preset('real_basic')
    elif choice == '3':
        launch_preset('real_sensitive')
    elif choice == '4':
        launch_preset('debug')
    elif choice == '5':
        launch_preset('performance')
    elif choice == '6':
        interactive_setup()
    elif choice == '7':
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    # Check if preset is specified
    if len(sys.argv) > 1:
        preset = sys.argv[1]
        if preset == '--interactive':
            interactive_setup()
        else:
            launch_preset(preset)
    else:
        main()