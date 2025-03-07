import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from Models.EEGModels import EEGNet  # Ensure this is your EEGNet implementation

# Define your new EEG configuration
new_channels = 7  # Change this based on your new dataset
samples = 200  # Keep the same number of time points
nb_classes = 2  # Keep the same output classes

# Load pre-trained EEGNet model (Make sure you have the right path)
pretrained_model_path = "pretrained_eegnet.h5"
pretrained_model = load_model(pretrained_model_path)

# Create a new EEGNet model with the updated channel count
new_model = EEGNet(nb_classes, new_channels, samples)

# Load weights while skipping mismatched layers
for layer in pretrained_model.layers:
    try:
        new_model.get_layer(layer.name).set_weights(layer.get_weights())
        print(f"Loaded weights for layer: {layer.name}")
    except ValueError:
        print(f"Skipping layer: {layer.name} (Shape mismatch)")

print("Pre-trained weights loaded where applicable.")

# Compile the model (ensure same optimizer settings as before)
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Load your dataset
data = np.load('./data/subject_data_v2.npz')
X = data['X']
y = data['y']
subject_ids = data['subject_ids']

# Ensure data is standardized
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    return X_train, X_test

# LOSO Cross-Validation
accuracy_per_subject = []

for subject in np.unique(subject_ids):
    print(f"Training on Subject {subject} (Leaving them out)...")

    # LOSO Data Split
    test_index = np.where(subject_ids == subject)[0]
    train_index = np.where(subject_ids != subject)[0]

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize
    X_train, X_test = standardize_data(X_train, X_test)

    # Expand dimensions for EEGNet
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])  # Swap axes for EEGNet
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    # Train the model
    history = new_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=16,
        epochs=20,  # Fine-tune for fewer epochs
        verbose=1
    )

    # Evaluate on test data
    scores = new_model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_subject.append(scores[1])
    print(f"Subject {subject} - Accuracy: {scores[1] * 100:.2f}%")

# Save subject-wise accuracies
with open("subject_accuracy_results.txt", "w") as f:
    f.write("Per-Subject Accuracies:\n")
    for subject, acc in zip(np.unique(subject_ids), accuracy_per_subject):
        f.write(f"Subject {subject}: {acc * 100:.2f}%\n")
    
    avg_accuracy = np.mean(accuracy_per_subject) * 100
    f.write(f"\nAverage Accuracy Across Subjects: {avg_accuracy:.2f}%\n")

print("Training complete. Subject accuracies saved to 'subject_accuracy_results.txt'.")
