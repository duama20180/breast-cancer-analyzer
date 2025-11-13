import os
import cv2
import numpy as np
import tensorflow as tf
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print("Failed to import Keras:", e)
    exit(1)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(dataset_path, max_images=None):
    images = []
    labels = []
    image_count = 0
    patient_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    for patient_folder in tqdm(patient_folders, desc="Processing patient folders"):
        patient_path = os.path.join(dataset_path, patient_folder)
        for label_folder in ['0', '1']:
            label_path = os.path.join(patient_path, label_folder)
            if not os.path.isdir(label_path):
                continue
            label = int(label_folder)
            for file in os.listdir(label_path):
                if file.endswith('.png'):
                    img_path = os.path.join(label_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    img = cv2.resize(img, (50, 50))
                    images.append(img)
                    labels.append(label)
                    image_count += 1
                    if max_images and image_count >= max_images:
                        break
            if max_images and image_count >= max_images:
                break
        if max_images and image_count >= max_images:
            break
    return np.array(images), np.array(labels)

dataset_path = 'IDC_regular_ps50_idx5'
X, y = load_data(dataset_path, max_images=20000)
X = X / 255.0

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_val)
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

model.save('breast_cancer_cnn.h5')
