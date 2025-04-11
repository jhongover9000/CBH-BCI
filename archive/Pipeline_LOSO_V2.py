# Training Pipeline with Leave-One-Subject-Out Cross-Validation
# ==================================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import keras
from keras.models import Sequential
from collections import defaultdict
import pandas as pd
from datetime import datetime
import gc
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from Models.EEGModels import EEGNet, DeepConvNet, ShallowConvNet
from Models.models import ATCNet_
from Models import atc, BFN

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)

weights_dir = "./Weights/"

data_dir = './data/'
data_filename = "subject_data_v2.npz"

# Load Data
data = np.load(data_dir + data_filename)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# ==================================================================================================
# Define Helper Functions
def scaler_fit_transform(X_train, X_test):
    scaler = StandardScaler()
    for i in range(X_train.shape[1]):
        scaler.fit(X_train[:, i, :])
        X_train[:, i, :] = scaler.transform(X_train[:, i, :])
        X_test[:, i, :] = scaler.transform(X_test[:, i, :])
    return X_train, X_test

# ==================================================================================================
# LOSO Cross-Validation
n_splits = len(np.unique(subject_ids))
epochs = 70
batch_size = 16
learning_rate = 0.00005
weight_decay = 0.01
samples, chans = X.shape[2], X.shape[1]
nb_classes = 2
conf_matrix_accum = np.zeros((nb_classes, nb_classes))
accuracy_per_fold, loss_per_fold, scores_atc = [], [], []

# Fine-tuning percentage
fine_tune_ratio = 0.2

print("Starting Training...")
for subject in np.unique(subject_ids):
    print(f"Processing Subject {subject}...")
    
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_train, X_test = scaler_fit_transform(X_train, X_test)
    X_train, X_test = np.expand_dims(X_train, 1), np.expand_dims(X_test, 1)
    y_train, y_test = to_categorical(y_train, nb_classes), to_categorical(y_test, nb_classes)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y[train_index]), y=y[train_index])
    class_weight_dict = dict(enumerate(class_weights))

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(np.shape(input_shape))
    model = atc.ATCNet(input_shape = input_shape, nb_classes = nb_classes)
    model.load_weights(weights_dir + "subject-9.h5", by_name=True, skip_mismatch=True)
    
    # Fine-tuning with a subset of subject's data
    X_fine_tune, _, y_fine_tune, _ = train_test_split(X_test, y_test, test_size=1-fine_tune_ratio)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_fine_tune, y_fine_tune, batch_size=batch_size, epochs=10, verbose=1)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        batch_size=batch_size, epochs=epochs, class_weight=class_weight_dict, verbose=1)
    
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    y_pred = model.predict(X_test)
    acc_atc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    scores_atc.append(acc_atc)
    
    y_pred_classes, y_test_classes = np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)
    conf_matrix_accum += confusion_matrix(y_test_classes, y_pred_classes, labels=range(nb_classes))
    
    print(f"Subject {subject} - Accuracy: {scores[1]:.4f}, Loss: {scores[0]:.4f}, ATC: {acc_atc:.4f}")
    
    del model
    K.clear_session()
    gc.collect()

# ==================================================================================================
# Model Evaluation
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"\nAverage Accuracy Across Subjects: {np.mean(accuracy_per_fold) * 100:.2f}%")
print(f"Average Loss Across Subjects: {np.mean(loss_per_fold):.4f}")
print(f"Average ATC Across Subjects: {np.mean(scores_atc) * 100:.2f}%")

conf_matrix_percent = conf_matrix_accum.astype('float') / conf_matrix_accum.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f"matrix/con_matrix_{timestamp}.png")
plt.show()
