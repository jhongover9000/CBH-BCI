'''
Helper script to save StandardScaler from training data
This should be run after training or integrated into the training script
'''

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

def save_scaler_from_training(data_path, output_dir="./saved_weights"):
    """
    Create and save StandardScaler from training data
    
    Args:
        data_path: Path to the training data NPZ file
        output_dir: Directory to save the scaler
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    X = data['X']
    subject_ids = data.get('subject_ids', None)
    
    print(f"Data shape: {X.shape}")
    
    if subject_ids is not None:
        # Per-subject scaling (as in training)
        unique_subjects = np.unique(subject_ids)
        print(f"Found {len(unique_subjects)} subjects")
        
        # Option 1: Save individual scalers per subject
        scalers = {}
        for subj in unique_subjects:
            subj_idx = np.where(subject_ids == subj)[0]
            if len(subj_idx) == 0:
                continue
                
            original_shape = X[subj_idx].shape
            X_subj = X[subj_idx].reshape(original_shape[0], -1)
            
            scaler = StandardScaler()
            scaler.fit(X_subj)
            scalers[subj] = scaler
            
            print(f"Subject {subj}: fitted scaler on {len(subj_idx)} samples")
        
        # Save per-subject scalers
        scaler_path = os.path.join(output_dir, "subject_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Saved per-subject scalers to: {scaler_path}")
        
        # Option 2: Create and save a global scaler (recommended for real-time)
        print("\nCreating global scaler from all data...")
        X_all = X.reshape(X.shape[0], -1)
        global_scaler = StandardScaler()
        global_scaler.fit(X_all)
        
        global_scaler_path = os.path.join(output_dir, "global_scaler.pkl")
        with open(global_scaler_path, 'wb') as f:
            pickle.dump(global_scaler, f)
        print(f"Saved global scaler to: {global_scaler_path}")
        
        # Print statistics
        print(f"\nGlobal scaler statistics:")
        print(f"  Mean shape: {global_scaler.mean_.shape}")
        print(f"  Scale shape: {global_scaler.scale_.shape}")
        print(f"  Features: {global_scaler.n_features_in_}")
        
    else:
        # No subject IDs, create global scaler only
        print("No subject IDs found, creating global scaler...")
        X_flat = X.reshape(X.shape[0], -1)
        scaler = StandardScaler()
        scaler.fit(X_flat)
        
        scaler_path = os.path.join(output_dir, "global_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved global scaler to: {scaler_path}")

def load_and_test_scaler(scaler_path, test_shape=(1, 22, 250)):
    """
    Load and test a saved scaler
    
    Args:
        scaler_path: Path to saved scaler
        test_shape: Shape of test data (batch, channels, samples)
    """
    print(f"\nTesting scaler from: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Test with random data
    test_data = np.random.randn(*test_shape)
    test_flat = test_data.reshape(test_shape[0], -1)
    
    try:
        scaled_data = scaler.transform(test_flat)
        print(f"✓ Scaler works correctly!")
        print(f"  Input shape: {test_data.shape}")
        print(f"  Flattened shape: {test_flat.shape}")
        print(f"  Scaled shape: {scaled_data.shape}")
        print(f"  Scaled mean: {scaled_data.mean():.6f} (should be ~0)")
        print(f"  Scaled std: {scaled_data.std():.6f} (should be ~1)")
    except Exception as e:
        print(f"✗ Error testing scaler: {e}")

if __name__ == "__main__":
    # Example usage
    data_path = "./data/bci_subject_data_v7.npz" 
    output_dir = "./saved_weights"
    
    # Save scalers
    save_scaler_from_training(data_path, output_dir)
    
    # Test the saved scaler
    load_and_test_scaler(os.path.join(output_dir, "global_scaler.pkl"))