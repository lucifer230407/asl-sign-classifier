import pandas as pd
import numpy as np
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tf_keras.utils import to_categorical
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("model", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("data/sign_mnist_train.csv")
test_df  = pd.read_csv("data/sign_mnist_test.csv")

y_train = train_df['label']
X_train = train_df.drop('label', axis=1)
y_test  = test_df['label']
X_test  = test_df.drop('label', axis=1)

# ── Normalize & reshape ──────────────────────────────────────────────────────
X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
X_test  = X_test.values.reshape(-1, 28, 28, 1) / 255.0

# ── Remove J (9) and Z (25) ──────────────────────────────────────────────────
excluded = [9, 25]
X_train = X_train[~y_train.isin(excluded)]
y_train = y_train[~y_train.isin(excluded)]
X_test  = X_test[~y_test.isin(excluded)]
y_test  = y_test[~y_test.isin(excluded)]

# ── Remap labels 0-23 ────────────────────────────────────────────────────────
unique_labels = sorted(y_train.unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = y_train.map(label_map)
y_test  = y_test.map(label_map)

y_train = to_categorical(y_train, num_classes=24)
y_test  = to_categorical(y_test,  num_classes=24)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── Data augmentation ────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# ── Build model ──────────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(24, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ── Train ────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("model/asl_model.h5", save_best_only=True, verbose=1)
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("model/training_plot.png")
print("✅ Model saved to model/asl_model.h5")
print("✅ Plot saved to model/training_plot.png")
