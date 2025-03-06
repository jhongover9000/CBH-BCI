'''
K-FOLD CLASSIFICATION PIPELINE

Description: Classifciation pipeline using a selection of models for pre-existing dataset. Still in
process of being made dynamic, there may be some hardcoded parts.
Uses K-Fold cross validation, the number of splits is a variable that can be edited.

Joseph Hong
'''
# ==================================================================================================
# ==================================================================================================
# IMPORTS
import numpy as np
from models import atc, BFN
import keras
from mne.io import read_epochs_eeglab
from mne import Epochs, find_events
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import tensorflow as tf

# Train Model with 5 fold validation
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import keras
import numpy as np

from models.EEGModels import EEGNet,DeepConvNet,ShallowConvNet
from models.models import ATCNet_


# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ==================================================================================================
# ==================================================================================================
# VARIABLES

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)

data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"

data_version = 'v3'
data_filename = f"subject_data_{data_version}.npz"  # Specify your desired output file

# Model Variables
nb_classes = 2
n_splits = 5  # Number of folds
epochs = 75
batch_size = 32             # batch size
validation_split = 0.2      # ratio of validation split

# Load Data
data = np.load(data_dir + data_filename)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# Set timestamp for identifier
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')


# ==================================================================================================
# ==================================================================================================
# FUNCTIONS

# Fit Transform (for Training Data)
def scaler_fit_transform(X_train, X_test):
    """
    Fits a StandardScaler on the training data and transforms both training and test data.
    """
    n_channels = X_train.shape[1]
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)

    scaler = StandardScaler()

    for i in range(n_channels):  # Iterate over each channel
        scaler.fit(X_train[:, i, :])  # Fit scaler on training data for each channel
        X_train_scaled[:, i, :] = scaler.transform(X_train[:, i, :])  # Transform training data
        X_test_scaled[:, i, :] = scaler.transform(X_test[:, i, :])  # Transform test data

    return X_train_scaled, X_test_scaled

# Plot Training History
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}{timestamp}_training_curve.png")
    plt.show()



# ==================================================================================================
# ==================================================================================================
# EXECUTION
print("Starting")

# Assume X (EEG data) and y (labels) are already prepared
# X.shape: (n_samples, n_channels, n_times), y.shape: (n_samples,)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
samples, chans = X.shape[2], X.shape[1]

# Initialize lists to track metrics for each fold
accuracy_per_fold = []
loss_per_fold = []

fold_number = 1

lr = 0.00005
w_decay = 0.01

for train_index, test_index in skf.split(X, y):
    print(f"Processing Fold {fold_number}...")
    
    # Split the data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize the data
    X_train, X_test = scaler_fit_transform(X_train, X_test)

    # Expand dimensions for compatibility with the model
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])  # Swap axes for EEGNet
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])

    model = EEGNet(nb_classes, chans, samples)
    model.load_weights(ref_weights_dir + "EEGNet-8-2-weights.h5", by_name=True, skip_mismatch=True)

    # Define the ATCNet model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (n_channels, n_times)
    nb_classes = len(np.unique(y))
    model = atc.ATCNet(input_shape=input_shape, nb_classes=nb_classes)
    opt_atc = keras.optimizers.Adam(learning_rate = lr)
    # Compile the model
    model.compile(optimizer=opt_atc, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())
    

    # Callbacks for training
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f'{saved_weights_dir}{timestamp}_best_model_fold_{fold_number}.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f"Fold {fold_number} - Loss: {scores[0]}, Accuracy: {scores[1]}")

    # Track metrics
    loss_per_fold.append(scores[0])
    accuracy_per_fold.append(scores[1])

    # Increment fold number
    fold_number += 1

# Display average metrics across all folds
print('Average metrics across all folds:')
print(f"Average Accuracy: {np.mean(accuracy_per_fold) * 100:.2f}%")
print(f"Average Loss: {np.mean(loss_per_fold):.4f}")


# ==================================================================================================
# ==================================================================================================
# EVALUATION 

# Test evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Group 1', 'Group 2']))

# Save accuracies to a text file
results_filename = f"{results_dir}{timestamp}_accuracy_results_kfold.txt"
with open(results_filename, "w") as f:
    f.write(f"Splits: {n_splits}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Validation Split: {validation_split}\n")
    f.write("Per-Fold Accuracies:\n")
    for i, acc in enumerate(accuracy_per_fold):
        f.write(f"Fold {i+1}: {acc * 100:.2f}%\n")
    
    # Save the average accuracy
    avg_accuracy = np.mean(accuracy_per_fold) * 100
    f.write(f"\nAverage Accuracy: {avg_accuracy:.2f}%\n")

print(f"Accuracies saved to {results_filename}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Group 1', 'Group 2'], yticklabels=['Group 1', 'Group 2'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f"{results_dir}{timestamp}_con_matrix.png")
plt.show()

plot_training_history()