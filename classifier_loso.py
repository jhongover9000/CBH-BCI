'''
LOSO CLASSIFICATION PIPELINE

Description: Classifciation pipeline using a selection of models for pre-existing dataset. Still in
process of being made dynamic, there may be some hardcoded parts.
Uses Leave-One-Subject-Out cross validation, the number of splits is a variable that can be edited.

Joseph Hong
'''
# ==================================================================================================
# ==================================================================================================
# IMPORTS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import keras
from keras.models import Sequential
from collections import defaultdict
import pandas as pd
# import BFN
from datetime import datetime
import gc
import shap
from shap.explainers._deep import deep_tf
import pickle


from tensorflow.keras.models import Model
# from deepexplain.tensorflow import DeepExplain
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

from models.EEGModels import EEGNet,DeepConvNet,ShallowConvNet
from models.models import ATCNet_

# ==================================================================================================
# ==================================================================================================
# VARIABLES

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)

# Enable SHAP value recording
shap_on = False

data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"

data_version = 'v4'
data_filename = f"subject_data_{data_version}.npz"  # Specify your desired output file
weight_filename = f"{saved_weights_dir}20250305151050_best_model_fold_1.weights.h5"


# Load Data
data = np.load(data_dir + data_filename)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# LOSO Cross-Validation
n_splits = len(np.unique(subject_ids))  # Number of subjects
epochs = 70
batch_size = 16
learning_rate = 0.00005
weight_decay = 0.01
samples, chans = X.shape[2], X.shape[1]
nb_classes = 2

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

# Plot History
def plot_training_history(history, timestamp):
    """Plot the training and validation loss and accuracy."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    fig_file = f"curves/History_{timestamp}.png"
    plt.savefig(fig_file)
    plt.show()

# ==================================================================================================
# ==================================================================================================
# EXECUTION

# Initialize an accumulator for the confusion matrix
conf_matrix_accum = np.zeros((nb_classes, nb_classes))

# Initialize metrics tracking
accuracy_per_fold = []
loss_per_fold = []
scores_atc = []
history_list = []

shap_values_all = []
y_test_all = []
y_pred_all = []

# Initialize dictionaries to track misclassified trials and counts per subject
misclassified_trials_all = defaultdict(list)  # Aggregated misclassified trials across all folds
misclassification_stats_all = defaultdict(int)  # Aggregated counts across all folds
misclassified_trials_per_fold = []  # List of dictionaries for each fold

print("Starting Training...")

# Iterate through each subject (Leave-One-Subject-Out)
for subject in np.unique(subject_ids):
    print(f"Processing Subject {subject}...")

    # Initialize and compile the model

    # model = ATCNet_(nb_classes, chans, samples)
    # model.load_weights( ref_weights_dir + "subject-9.h5",  by_name=True, skip_mismatch=True)


    model = EEGNet(nb_classes, chans, samples)
    # model.load_weights(weight_filename, skip_mismatch=True)
    model.load_weights(ref_weights_dir + "EEGNet-8-2-weights.h5", by_name=True, skip_mismatch=True)

    # Initialize and compile the model
    # model = BFN.proposed(samples, chans, nb_classes)
    # model.load_weights(ref_weights_dir + "pretrained_VR.h5", by_name=True, skip_mismatch=True)

    # Define the train and test splits for this fold (leave one subject out)
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]
    
   # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize the data
    X_train, X_test = scaler_fit_transform(X_train, X_test)
    
    # Expand dimensions for compatibility with the model
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])  # Swap axes for EEGNet
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y[train_index]), y=y[train_index])
    class_weight_dict = dict(enumerate(class_weights))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f"{saved_weights_dir}{timestamp}_best_model_subject_{subject}.weights.h5", monitor='val_loss', save_weights_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    history_list.append(history)
    
    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    # Predict and calculate 
    y_pred = model.predict(X_test)
    acc_atc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    scores_atc.append(acc_atc)

    # Predict on the test set for the current fold
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Shapley Analysis
    if (shap_on):
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
        shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  
        shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["BatchMatMulV2"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        shap.explainers._deep.deep_tf.op_handlers["Neg"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
        # Set up 300 random points for shap
        background = np.array(X_train[np.random.choice(X_train.shape[0], 300, replace=False)])
        print(np.shape(X_test))
        # Create DeepExplainer model
        e = shap.DeepExplainer(model, background)
        print(e)
        shap_values = e.shap_values(X_test, check_additivity=False)
        shap_values_all.append(shap_values)

    # Append Accuracies    
    y_test_all.append(y_test.argmax(axis=-1))
    y_pred_all.append(y_pred)
    print(len(shap_values_all))
    print(len(y_test_all))
    print(len(y_pred_all))

    # Identify misclassified trials
    misclassified_indices = np.where(y_pred_classes != y_test_classes)[0]
    test_subjects = subject_ids[test_index]  # Map test indices to subject IDs

    # Track misclassified trials for the current subject
    misclassified_trials_current_fold = defaultdict(list)
    for idx in misclassified_indices:
        subject_id = test_subjects[idx]
        misclassified_trials_current_fold[subject_id].append(idx)
        misclassified_trials_all[subject_id].append(idx)
        misclassification_stats_all[subject_id] += 1

    misclassified_trials_per_fold.append(misclassified_trials_current_fold)

    # Compute confusion matrix for the current subject
    fold_conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=range(nb_classes))
    
    # Accumulate confusion matrices
    conf_matrix_accum += fold_conf_matrix
    
    print(f"Subject {subject} - Accuracy: {scores[1]:.4f}, Loss: {scores[0]:.4f}, ATC: {acc_atc:.4f}")

    # ==================================================================================================
    # Clear memory after each fold
    del model  # Delete model to free memory
    K.clear_session()  # Clear Keras session to free up resources
    gc.collect()  # Run garbage collection to clean up any residual memory usage

# ==================================================================================================
# ==================================================================================================
# EVALUATION

print(f"\nAverage Accuracy Across Subjects: {np.mean(accuracy_per_fold) * 100:.2f}%")
print(f"Average Loss Across Subjects: {np.mean(loss_per_fold):.4f}")
print(f"Average ATC Across Subjects: {np.mean(scores_atc) * 100:.2f}%")
print(accuracy_per_fold)

# Save subject-wise accuracies to a text file
results_filename = f"{results_dir}{timestamp}_accuracy_results_LOSO.txt"
with open(results_filename, "w") as f:
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write("Per-Subject Accuracies:\n")
    for subject, acc in zip(np.unique(subject_ids), accuracy_per_fold):
        f.write(f"Subject {subject}: {acc * 100:.2f}%\n")
    
    # Save the average accuracy
    avg_accuracy = np.mean(accuracy_per_fold) * 100
    f.write(f"\nAverage Accuracy Across Subjects: {avg_accuracy:.2f}%\n")

print(f"Subject accuracies saved to {results_filename}")

# Confusion Matrix and Heatmap
conf_matrix_percent = conf_matrix_accum.astype('float') / conf_matrix_accum.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f"{results_dir}{timestamp}_con_matrix.png")
plt.show()

# Misclassification Analysis
misclassification_df = pd.DataFrame(misclassification_stats_all.items(), columns=["Subject", "Misclassified Trials"])
misclassification_df.to_csv(f"{shap_dir}{timestamp}_misclassifications.csv", index=False)

# Shapley Analysis
if (shap_on):
    with open(f"{shap_dir}{timestamp}_shaps_values_all_LOSO", "wb") as fp:  # Pickling
        pickle.dump(shap_values_all, fp)

    with open(f"{shap_dir}{timestamp}_y_test_all_LOSO", "wb") as fp:  # Pickling
        pickle.dump(y_test_all, fp)

    with open(f"{shap_dir}{timestamp}_y_pred_all_LOSO", "wb") as fp:  # Pickling
        pickle.dump(y_pred_all, fp)