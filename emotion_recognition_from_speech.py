# -*- coding: utf-8 -*-

"""# Task
Recognize emotions from the provided speech audio files "/content/03-01-01-01-01-01-24.wav", "/content/03-01-01-01-01-02-24.wav", "/content/03-01-01-01-02-01-24.wav", "/content/03-01-01-01-02-02-24.wav", and "/content/03-01-02-01-01-02-24.wav" using deep learning and speech signal processing techniques, including MFCC feature extraction and a CNN, RNN, or LSTM model.

## Install necessary libraries

### Subtask:
Install libraries for audio processing, feature extraction, and deep learning.

**Reasoning**:
Install the required libraries for audio processing, feature extraction, and deep learning.
"""

!pip install librosa tensorflow scikit-learn

"""## Load and preprocess data

### Subtask:
Load the audio files and their corresponding emotion labels.

**Reasoning**:
Load the audio files and extract the emotion labels, storing them in a list of dictionaries.
"""

import librosa
import os

audio_files = [
    "/content/03-01-01-01-01-01-24.wav",
    "/content/03-01-01-01-01-02-24.wav",
    "/content/03-01-01-01-02-01-24.wav",
    "/content/03-01-01-01-02-02-24.wav",
    "/content/03-01-02-01-01-02-24.wav",
]

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

audio_data = []

for file_path in audio_files:
    filename = os.path.basename(file_path)
    emotion_code = filename.split('-')[2]
    emotion_label = emotion_map.get(emotion_code, 'unknown')

    y, sr = librosa.load(file_path)

    audio_data.append({
        'file_path': file_path,
        'emotion': emotion_label,
        'audio_time_series': y,
        'sampling_rate': sr
    })

print(f"Loaded {len(audio_data)} audio files.")

"""## Extract features

### Subtask:
Extract relevant features like MFCCs from the audio data.

**Reasoning**:
Extract MFCC features from the loaded audio data and store them with their corresponding emotion labels.
"""

import librosa.feature
import numpy as np

extracted_features = []
max_pad_len = 0

for entry in audio_data:
    mfccs = librosa.feature.mfcc(y=entry['audio_time_series'], sr=entry['sampling_rate'], n_mfcc=40)
    extracted_features.append({
        'emotion': entry['emotion'],
        'mfccs': mfccs
    })
    if mfccs.shape[1] > max_pad_len:
        max_pad_len = mfccs.shape[1]

# Pad or truncate MFCC sequences
processed_features = []
for entry in extracted_features:
    mfccs = entry['mfccs']
    padded_mfccs = np.pad(mfccs, ((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
    processed_features.append({
        'emotion': entry['emotion'],
        'mfccs': padded_mfccs
    })

print(f"Extracted and processed features for {len(processed_features)} audio files.")
# Display the shape of the first processed feature as an example
if processed_features:
    print(f"Shape of the first processed MFCC feature: {processed_features[0]['mfccs'].shape}")

"""## Prepare data for model

### Subtask:
Split the data into training and testing sets and format it for the deep learning model.

**Reasoning**:
Extract the features and labels from the processed data, encode the labels, and split the data into training and testing sets.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# 1. Extract features and labels
features = [entry['mfccs'] for entry in processed_features]
labels = [entry['emotion'] for entry in processed_features]

# Convert features list to a numpy array
features = np.array(features)

# 2. Encode the categorical emotion labels
# Use LabelEncoder to convert string labels to integers
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)

# Use OneHotEncoder to convert integer labels to one-hot vectors
# Reshape is needed because OneHotEncoder expects a 2D array
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded_labels.reshape(-1, 1))

# 3. Split the data into training and testing sets
# For this small dataset, stratification is not strictly necessary but good practice
X_train, X_test, y_train, y_test = train_test_split(
    features,
    onehot_encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=onehot_encoded_labels if len(np.unique(labels)) > 1 else None # Stratify only if there's more than one class
)

# 4. Reshape features for CNN input (add channel dimension)
# Assuming the model expects input shape (batch_size, n_mfcc, n_frames, n_channels)
# where n_channels is typically 1 for grayscale-like features
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Data splitting and encoding complete.")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# 1. Extract features and labels
features = [entry['mfccs'] for entry in processed_features]
labels = [entry['emotion'] for entry in processed_features]

# Convert features list to a numpy array
features = np.array(features)

# 2. Encode the categorical emotion labels
# Use LabelEncoder to convert string labels to integers
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)

# Use OneHotEncoder to convert integer labels to one-hot vectors
# Reshape is needed because OneHotEncoder expects a 2D array
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded_labels.reshape(-1, 1))

# 3. Split the data into training and testing sets
# Due to the small dataset size and potential for classes with only one sample,
# stratification is removed.
X_train, X_test, y_train, y_test = train_test_split(
    features,
    onehot_encoded_labels,
    test_size=0.2,
    random_state=42
)

# 4. Reshape features for CNN input (add channel dimension)
# Assuming the model expects input shape (batch_size, n_mfcc, n_frames, n_channels)
# where n_channels is typically 1 for grayscale-like features
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print("Data splitting and encoding complete.")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

"""## Build and train model

### Subtask:
Design and train a deep learning model (e.g., CNN, RNN, or LSTM) for emotion recognition.

**Reasoning**:
Design, compile, and train a deep learning model for emotion recognition using the prepared training data.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten

# 1. Define the model architecture
model = Sequential()

# Adding Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Reshaping the output for the LSTM layer
# The output of Conv2D is (batch_size, height, width, channels)
# We need to reshape it to (batch_size, timesteps, features) for LSTM
# We can treat the flattened spatial dimensions as features over time (height or width)
# Let's flatten the last two dimensions and treat height as timesteps
# The shape before flatten: (batch_size, new_height, new_width, new_channels)
# For simplicity, let's flatten and then reshape to (batch_size, new_height, new_width * new_channels)
model.add(Flatten())
# Calculate the dimensions after flattening
flattened_shape = model.output_shape[1]
# Determine the number of timesteps. We can choose one of the original spatial dimensions.
# Let's assume the original height (40) is the timestep dimension after some pooling.
# This requires careful consideration of how the pooling layers affect the dimensions.
# A simpler approach for a mixed CNN-RNN is to apply CNN, flatten, and then feed to RNN.
# However, the instruction suggests a combination, implying feeding CNN output into RNN.
# Let's try reshaping the output of the last Conv/Pool block before flattening.
# Need to get the shape of the output of the last MaxPooling2D layer.
# Let's rebuild the model structure to include the reshape before LSTM.

# Re-defining the model for CNN -> Reshape -> LSTM -> Dense
model_cnn_lstm = Sequential()

# Adding Convolutional layers
model_cnn_lstm.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model_cnn_lstm.add(MaxPooling2D((2, 2)))
model_cnn_lstm.add(Dropout(0.25))

model_cnn_lstm.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn_lstm.add(MaxPooling2D((2, 2)))
model_cnn_lstm.add(Dropout(0.25))

# Calculate the output shape of the last pooling layer dynamically
# This is tricky without running model.summary() or calculating manually.
# Let's calculate manually based on the input shape (40, 160, 1) and kernel/pool sizes
# After first Conv2D(32, (3,3)): shape becomes (38, 158, 32) (assuming padding='valid')
# After first MaxPooling2D((2,2)): shape becomes (19, 79, 32)
# After second Conv2D(64, (3,3)): shape becomes (17, 77, 64)
# After second MaxPooling2D((2,2)): shape becomes (8, 38, 64) (integer division)

# Reshape the output to be suitable for LSTM (batch_size, timesteps, features)
# Let's use 8 as timesteps and 38*64 as features
timesteps = 8
features_per_timestep = 38 * 64
model_cnn_lstm.add(tf.keras.layers.Reshape((timesteps, features_per_timestep)))

# Adding LSTM layer
model_cnn_lstm.add(LSTM(64, return_sequences=False)) # return_sequences=False for classification
model_cnn_lstm.add(Dropout(0.25))

# Adding Dense layers
model_cnn_lstm.add(Dense(32, activation='relu'))
model_cnn_lstm.add(Dropout(0.25))

# Output layer
num_classes = y_train.shape[1]
model_cnn_lstm.add(Dense(num_classes, activation='softmax'))

# 2. Compile the model
model_cnn_lstm.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

model_cnn_lstm.summary()

# 3. Train the model
# With only 4 training samples, training will likely not be effective or generalize well.
# Setting a small number of epochs for demonstration.
epochs = 10
batch_size = 1 # With only 4 samples, batch size must be small

history = model_cnn_lstm.fit(X_train, y_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             verbose=1) # Set verbose to 1 to see training progress

# 4. Store the trained model (optional, but good practice)
# model_cnn_lstm.save('emotion_recognition_model.h5') # Save the entire model
# Or save just the weights
# model_cnn_lstm.save_weights('emotion_recognition_weights.h5')

print("Model training complete.")

"""## Evaluate model

### Subtask:
Evaluate the trained model's performance on the test set.

**Reasoning**:
Evaluate the trained model on the test set to determine its performance.
"""

# Evaluate the model on the test set
test_loss, test_accuracy = model_cnn_lstm.evaluate(X_test, y_test, verbose=0)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

"""## Predict emotion

### Subtask:
Use the trained model to predict emotions from new audio data.

**Reasoning**:
Select audio files, preprocess them, predict emotions using the trained model, and print the results.
"""

import librosa
import numpy as np

# Select audio files for prediction
audio_files_to_predict = [
    "/content/03-01-01-01-01-01-24.wav",
    "/content/03-01-01-01-01-02-24.wav",
    "/content/03-01-01-01-02-01-24.wav",
]

predicted_emotions = {}

for file_path in audio_files_to_predict:
    # Preprocess the audio file
    y, sr_load = librosa.load(file_path, sr=sr) # Use the same sampling rate as training

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr_load, n_mfcc=40)

    # Pad or truncate MFCC sequences to max_pad_len
    if mfccs.shape[1] < max_pad_len:
        padded_mfccs = np.pad(mfccs, ((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
    else:
        padded_mfccs = mfccs[:, :max_pad_len]

    # Reshape for model input (add batch and channel dimensions)
    # Model expects (batch_size, n_mfcc, n_frames, n_channels)
    input_features = np.expand_dims(padded_mfccs, axis=0) # Add batch dimension
    input_features = np.expand_dims(input_features, axis=-1) # Add channel dimension

    # Use the trained model to predict the emotion
    predictions = model_cnn_lstm.predict(input_features)

    # Convert prediction (probability distribution) to emotion label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # Inverse transform using the one-hot encoder and label encoder
    # The onehot_encoder's categories are the unique integer labels
    # The label_encoder can inverse transform the integer label
    predicted_integer_label = onehot_encoder.categories_[0][predicted_class_index]
    predicted_emotion_label = label_encoder.inverse_transform([predicted_integer_label])[0]

    predicted_emotions[file_path] = predicted_emotion_label

# Print the predicted emotions
print("Predicted emotions:")
for file_path, emotion in predicted_emotions.items():
    print(f"{file_path}: {emotion}")

"""## Summary:

### Data Analysis Key Findings

*   The process successfully loaded and preprocessed 5 audio files, extracting emotion labels based on the RAVDESS filename convention.
*   Mel-Frequency Cepstral Coefficients (MFCCs) were extracted from each audio file, with 40 MFCCs per frame.
*   MFCC sequences were padded to a uniform length of 160 frames, resulting in a feature shape of (40, 160).
*   Due to the small dataset size (5 samples), stratification was removed during the train-test split. The data was split into 4 training samples and 1 testing sample.
*   A CNN-LSTM model architecture was built and compiled for emotion recognition.
*   The model was trained on the small training dataset for 10 epochs with a batch size of 1.
*   The model achieved a test accuracy of 1.0000 on the single test sample.
*   Predictions were made on three selected audio files, and the model predicted the emotion as "neutral" for all three.

### Insights or Next Steps

*   The perfect test accuracy is likely an artifact of the extremely small test set size (n=1) and does not indicate good model performance or generalization.
*   To build a robust emotion recognition model, a significantly larger and more diverse dataset is required for training and evaluation. Future steps should focus on acquiring a larger dataset or using techniques like data augmentation or transfer learning if a larger dataset is not available.

"""
