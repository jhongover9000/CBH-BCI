# ==================================================================================================
# IMPORTS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

from models.atcnet_new import ATCNet_

# ==================================================================================================
# VARIABLES

# Set Seed for Robustness Testing
SEED = 42  # Change this value to test robustness

# Number of channels in the original model (pre-trained)
original_channels = 19
# Number of channels in the new dataset
new_channels = 6

# Directories
data_dir = './data/'
ref_weights_dir = "./reference_weights/"
saved_weights_dir = "./saved_weights/"
results_dir = "./results/"
shap_dir = "./shap/"
ref_weight = 'ATC_NT.weights.h5'

# Data Configurations
data_version = 'v4'
data_type = 'xon'
if(data_type == 'mit'):
    data_filename = f"mit_subject_data_{data_version}.npz"
elif (data_type == 'xon'):
    data_filename = f"xon_subject_data_{data_version}.npz"
else:
    data_filename = f"subject_data_{data_version}.npz"


# Load New Data
data = np.load(data_dir + data_filename)
X_new = data['X']  # Shape (trials, new_channels, timepoints)
y_new = data['y']
subject_ids = data['subject_ids']
print(f"New Data loaded. X shape: {X_new.shape}, y shape: {y_new.shape}, Subject IDs: {subject_ids.shape}")

# Training Configurations
epochs = 90  # Fine-tuning for fewer epochs
batch_size = 8
learning_rate = 0.00001  # Lower LR for fine-tuning
nb_classes = 2
weight_decay = 0.01

# Number of layers to freeze (you might need to experiment with this value)
num_layers_to_freeze = 2


# Timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# ==================================================================================================
# LOAD PRE-TRAINED MODEL AND ADJUST INPUT SIZE

# Load the original ATCNet model (22 channels)
original_model = ATCNet_(nb_classes, original_channels, X_new.shape[2])

# Load pre-trained weights
pretrained_weights_path = f"{ref_weights_dir}{ref_weight}"
try:
    original_model.load_weights(pretrained_weights_path)
    print("✅ Loaded pre-trained weights from:", pretrained_weights_path)
except:
    print(f"⚠ No pre-trained weights found at {ref_weights_dir}{ref_weight}. Training from scratch.")
    exit()

# Create a new model with the adjusted number of channels (7)
new_model = ATCNet_(nb_classes, new_channels, X_new.shape[2])

# Map layer names from old model to new model
layer_mapping = {orig_layer.name: new_layer.name for orig_layer, new_layer in zip(original_model.layers, new_model.layers)}

for orig_layer_name, new_layer_name in layer_mapping.items():
    try:
        # Skip input-dependent layers
        # if "conv2d" in orig_layer_name or "depthwise_conv2d" in orig_layer_name:
        #     print(f"🚫 Skipping incompatible Conv layer: {orig_layer_name}")
        #     continue  

        # if "batch_normalization" in orig_layer_name:
        #     print(f"⚠ Adapting BatchNorm layer: {orig_layer_name}")
        #     orig_weights = original_model.get_layer(orig_layer_name).get_weights()
        #     if len(orig_weights) == 4:  # Standard BatchNorm has 4 params
        #         gamma, beta, mean, var = orig_weights
        #         new_model.get_layer(new_layer_name).set_weights([gamma[:new_channels], beta[:new_channels],
        #                                                          mean[:new_channels], var[:new_channels]])
        #     continue  

        # Transfer all other compatible layers
        new_model.get_layer(new_layer_name).set_weights(original_model.get_layer(orig_layer_name).get_weights())
        print(f"✅ Transferred weights for {orig_layer_name} → {new_layer_name}")

    except ValueError as e:
        print(f"⚠ Skipping {orig_layer_name} due to mismatch: {e}")

print("Weight transfer complete. Now fine-tuning.")

# ==================================================================================================
# FREEZE LAYERS

print(f"Freezing the first {num_layers_to_freeze} layers of the new model.")
for i, layer in enumerate(new_model.layers[:num_layers_to_freeze]):
    layer.trainable = False
    print(f"🔒 Layer {i}: {layer.name} is now frozen.")

# Verify which layers are trainable
print("\nTrainable layers in the new model:")
for layer in new_model.layers:
    print(f"- {layer.name}: {layer.trainable}")

# ==================================================================================================
# DATA PREPARATION

# Expand dimensions for ATCNet input (batch, 1, channels, time)
X_new = np.expand_dims(X_new, axis=1)

# Convert labels to categorical
y_new = to_categorical(y_new, nb_classes)

# ==================================================================================================
# ADJUST `n_windows` TO MATCH TIME DIMENSION

final_time_dim = X_new.shape[-1]  # Time dimension after input processing
n_windows = min(final_time_dim, 3)  # Ensure `n_windows` is valid
print(f"Updated `n_windows`: {n_windows}")

# ==================================================================================================
# COMPILE AND TRAIN

# Compile Model
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
    ModelCheckpoint(f"{saved_weights_dir}{timestamp}_fine_tuned_ATCNet.keras", monitor='val_loss', save_best_only=True)
]

# Train the model on the new data
history = new_model.fit(
    X_new, y_new,
    batch_size=batch_size,
    epochs=epochs,
    # validation_split=0.1, # Use 20% of new data for validation
    callbacks=callbacks,
    verbose=1
)

# Save the final fine-tuned model
new_model.save_weights(f"{saved_weights_dir}{timestamp}_final_finetuned_ATCNet.weights.h5")
print("✅ Fine-tuned model saved.")

# ==================================================================================================
# SAVE RESULTS

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
fig_file = f"{results_dir}{timestamp}_FineTuning_History.png"
plt.savefig(fig_file)
plt.show()

print(f"✅ Fine-tuning complete. Training history saved as {fig_file}.")