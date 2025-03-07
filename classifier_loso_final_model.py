import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from models.EEGModels import EEGNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

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


# Load pre-trained EEGNet model
pretrained_model_path = "pretrained_eegnet.h5"
pretrained_model = load_model(pretrained_model_path)

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

# Define dataset and model settings
new_channels = 32  # Adjust based on new EEG setup
samples = 400  
nb_classes = 2  
batch_size = 16  
epochs = 20  # LOSO fine-tuning epochs
final_epochs = 50  # Final full-dataset model training


# Load EEG dataset
data = np.load('./data/subject_data_v2.npz')
X = data['X']
y = data['y']
subject_ids = data['subject_ids']

# Standardization function
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    return X_train, X_test

# LOSO Cross-Validation Training
accuracy_per_subject = []

for subject in np.unique(subject_ids):
    print(f"Training on Subject {subject} (Leaving them out)...")

    # Create a new EEGNet model
    model = EEGNet(nb_classes, new_channels, samples)

    # Load pre-trained weights, skipping mismatched layers
    for layer in pretrained_model.layers:
        try:
            model.get_layer(layer.name).set_weights(layer.get_weights())
            print(f"Loaded weights for layer: {layer.name}")
        except ValueError:
            print(f"Skipping layer: {layer.name} (Shape mismatch)")

    # LOSO Data Split
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize data
    X_train, X_test = standardize_data(X_train, X_test)

    # Reshape for EEGNet
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    # Save model for this subject (trained without them)
    model.save_weights(f"model_subject_{subject}.h5")

    # Evaluate on test data
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_subject.append(scores[1])
    print(f"Subject {subject} - Accuracy: {scores[1] * 100:.2f}%")

# Save subject-wise accuracies
with open("subject_accuracy_results.txt", "w") as f:
    f.write("Per-Subject Accuracies:\n")
    for subject, acc in zip(np.unique(subject_ids), accuracy_per_subject):
        f.write(f"Subject {subject}: {acc * 100:.2f}%\n")
    
    avg_accuracy = np.mean(accuracy_per_subject) * 100
    f.write(f"\nAverage Accuracy Across Subjects: {avg_accuracy:.2f}%\n")

print("Subject-specific models saved. Now training a final model on all subjects...")

# ============================
# **Train Final Model on All Data**
# ============================

# Create a new EEGNet model
final_model = EEGNet(nb_classes, new_channels, samples)

# Load pre-trained weights (except the first layer)
for layer in pretrained_model.layers:
    try:
        final_model.get_layer(layer.name).set_weights(layer.get_weights())
        print(f"Loaded weights for layer: {layer.name}")
    except ValueError:
        print(f"Skipping layer: {layer.name} (Shape mismatch)")

# Standardize all data
X_train, X_test = standardize_data(X, X)

# Reshape for EEGNet
X_train = np.expand_dims(X_train, 1)
X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y, nb_classes)

# Compile model
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model on full dataset
final_model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=final_epochs,
    verbose=1
)

# Save final trained model
final_model.save_weights("final_model_all_subjects.h5")
print("Final model trained and saved.")

# ============================
# **Ensemble Prediction Function (Optional)**
# ============================
def ensemble_predict(models, X_test):
    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    return predictions
