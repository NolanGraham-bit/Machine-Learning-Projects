#************************************************************************************
# Nolan Graham
# ML – HW#5 (Fast Version for Testing & Submission)
# Filename: cnn_crack_detection_fast.py
# Due: March 26, 2025
#
# Objective:
# Lightweight CNN version that runs faster but still generates required outputs.
#*************************************************************************************

import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for LEAP/headless systems
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Directory setup
BASE_DIR = os.getcwd()
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training', shuffle=True, seed=42
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='validation', shuffle=True, seed=42
)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compiling
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    steps_per_epoch=100,
    validation_steps=20,
    callbacks=[early_stop]
)

# Evaluation
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions
y_pred = (model.predict(test_gen) > 0.5).astype("int32")
y_true = test_gen.classes

# Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Metrics Report
report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
with open('metrics_report.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

# Training/Validation Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png')
plt.close()

print("Fast version complete — metrics and plots saved.")
