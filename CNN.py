import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from tabulate import tabulate
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import resampy

# Function to extract MFCC features and resample audio to 16 kHz
def extract_mfcc(audio_file_path, num_mfcc=44, target_sample_rate=16000, desired_length = 2048):
    audio_data, sample_rate = librosa.load(audio_file_path, sr=target_sample_rate, duration=1.0, res_type='kaiser_fast')

    # Zero-pad the audio signal if it's shorter than desired_length
    if len(audio_data) < desired_length:
        shortage = desired_length - len(audio_data)
        padding = shortage // 2
        audio_data = np.pad(audio_data, (padding, shortage - padding), 'constant')

    # Resample the audio to the target sample rate
    audio_data = resampy.resample(audio_data, sample_rate, target_sample_rate)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=num_mfcc)

    return mfccs

# Defining batch size and create data generators
batch_size = 32

def data_generator(audio_paths, labels, batch_size, num_mfcc, max_length):
    num_samples = len(audio_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_audio_paths = audio_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]

            batch_mfcc_sequences = []
            for audio_path in batch_audio_paths:
                mfccs = extract_mfcc(audio_path, num_mfcc=num_mfcc)

                # Pad or truncate the MFCC sequence to the maximum length
                if mfccs.shape[1] < max_length:
                    mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
                else:
                    mfccs = mfccs[:, :max_length]

                    # Reshape MFCC sequence to match model's input shape
                mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
                batch_mfcc_sequences.append(mfccs)

            yield np.array(batch_mfcc_sequences), np.array(batch_labels)

audio_folder_path = 'input'

# Setting a fixed number of MFCC coefficients for all audio files
num_mfcc = 20

# Setting a smaller maximum length for padding/truncating MFCC sequences
max_length = 36

# List to store the audio file paths and corresponding labels
audio_paths = []
labels = []

# Defining a mapping of folder names to labels
class_mapping = {}

# Loop through the audio files in the folder and collect audio file paths and labels:
for i, folder_name in enumerate(os.listdir(audio_folder_path)):
    folder_path = os.path.join(audio_folder_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    class_mapping[folder_name] = i

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            audio_file_path = os.path.join(folder_path, filename)
            audio_paths.append(audio_file_path)

            # Assign a label based on the folder name
            label = class_mapping[folder_name]
            labels.append(label)

# Converting the lists to numpy arrays
audio_paths = np.array(audio_paths)
labels = np.array(labels)

# Splitting the data into train and test sets (80% train, 20% test)
train_paths, test_paths, train_labels, test_labels = train_test_split(audio_paths, labels, test_size=0.2, random_state=42)

# Creating data generators for train and test data
train_generator = data_generator(train_paths, train_labels, batch_size, num_mfcc, max_length)
test_generator = data_generator(test_paths, test_labels, batch_size, num_mfcc, max_length)

# Defining the number of classes
num_classes = len(class_mapping)  # 16

if os.path.exists('trained_model.h5'):
    # Load the saved model
    loaded_model = tf.keras.models.load_model('trained_model.h5')
else:
    # CNN model with 7 convolutional layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(max_length, num_mfcc, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compiling the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Printing model summary
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Training the model using the data generators
    epochs = 20
    steps_per_epoch = len(train_paths) // batch_size
    validation_steps = len(test_paths) // batch_size
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=test_generator, validation_steps=validation_steps, callbacks=[early_stopping])

    # Saving the trained model
    model.save('trained_model.h5')
    loaded_model = model

# Evaluating the loaded model on the test data
test_predictions = []
test_labels = []
for batch in test_generator:
    batch_mfccs, batch_labels = batch

    # Reshape and transpose batch_mfccs to match the input shape of the model
    batch_mfccs = np.array(batch_mfccs)
    batch_mfccs = np.transpose(batch_mfccs, (0, 2, 1, 3))

    # Make predictions with the loaded model
    predictions = loaded_model.predict(batch_mfccs)
    batch_predictions = np.argmax(predictions, axis=-1)

    test_predictions.extend(batch_predictions)
    test_labels.extend(batch_labels)

    if len(test_predictions) >= len(test_paths):
        break

test_predictions = np.array(test_predictions)
test_labels = np.array(test_labels)

# Computing the confusion matrix
confusion_matrix = metrics.confusion_matrix(test_labels, test_predictions)

print("Confusion Matrix:")
print(confusion_matrix)
# Defining a reverse mapping for class labels to class names
class_mapping_inv = {v: k for k, v in class_mapping.items()}
#
#Computing the original confusion matrix
confusion_mat = sklearn_confusion_matrix(test_labels, test_predictions, labels=np.arange(num_classes))

class_names = list(class_mapping.keys()) #13

# Plotting the original confusion matrix using Seaborn and Matplotlib
plt.figure(figsize=(10, 8))
sns.set(font_scale=1)
sns.heatmap(confusion_mat, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Original Confusion Matrix")
plt.show()
