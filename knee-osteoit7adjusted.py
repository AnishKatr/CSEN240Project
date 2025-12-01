#!/usr/bin/env python3

import os
import cv2
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
# Switched to DenseNet121 - Better for texture/medical x-rays
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import legacy as legacy_optimizers

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
IMG_SIZE = 256  # Slightly smaller to allow larger batch size/faster texture learning
BATCH_SIZE = 32
DATA_PATH = "."
TRAIN_DIR = os.path.join(DATA_PATH, "train")
MODEL_PATH = "best_densenet_knee_fixed.h5"

# --- 1. DATA PREPARATION ---
categories = ["Normal", "Osteopenia", "Osteoporosis"]
image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(TRAIN_DIR, category)
    if os.path.exists(category_path):
        for image_name in os.listdir(category_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(category_path, image_name))
                labels.append(category)

df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(f"Total images: {len(df)}")

# Encode labels
label_encoder = LabelEncoder()
df["category_encoded_int"] = label_encoder.fit_transform(df["label"])
df["category_encoded"] = df["category_encoded_int"].astype(str)

# Stratified Split
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["category_encoded"]
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["category_encoded"]
)

print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

# --- 2. CALCULATE CLASS WEIGHTS ---
# This fixes the "Class 2 Collapse" by making rare classes more 'expensive' to miss
class_weights_arr = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['category_encoded_int']),
    y=train_df['category_encoded_int']
)
class_weights = dict(enumerate(class_weights_arr))
print("Class Weights applied:", class_weights)

# --- 3. CUSTOM PREPROCESSING (CLAHE) ---
def apply_clahe(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance bone trabeculae texture.
    Expects img input to be 0-255 range.
    """
    # Convert RGB to Lab to process only the 'Lightness' channel
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge back and convert to RGB
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Normalize to 0-1 range for the model
    return final.astype(np.float32) / 255.0

# --- 4. GENERATORS ---
# Note: We remove 'rescale' here because our function handles normalization
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe, 
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_test_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="category_encoded",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True
)

valid_gen = valid_test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col="image_path",
    y_col="category_encoded",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

test_gen = valid_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="category_encoded",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# --- 5. MODEL BUILD (DenseNet121) ---
def build_densenet_model():
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Phase 1: Freeze Base
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Batch Norm helps stabilize the inputs to the dense layer
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x) # Reduced dropout to keep more info
    outputs = Dense(3, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet_Knee")
    return model, base_model

model, base_model = build_densenet_model()

# --- 6. TRAINING PHASE 1 (Head Only) ---
print("\n--- PHASE 1: Training Head ---")
optimizer = legacy_optimizers.Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

history1 = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=15,
    class_weight=class_weights, # Crucial for fixing imbalance
    callbacks=callbacks
)

# --- 7. TRAINING PHASE 2 (Fine Tuning) ---
print("\n--- PHASE 2: Fine Tuning ---")

# Unfreeze the last block of DenseNet
base_model.trainable = True
total_layers = len(base_model.layers)
# Freeze first 85% of layers, train last 15%
fine_tune_at = int(total_layers * 0.85)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# CRITICAL: Keep BatchNormalization layers frozen during fine-tuning
# Otherwise they learn mean/var of the small batch and destroy the model
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

optimizer_ft = legacy_optimizers.Adam(learning_rate=1e-5) # Very slow learning rate

model.compile(
    optimizer=optimizer_ft,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_ft = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

history2 = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks_ft
)

# --- 8. EVALUATION ---
print("\n--- EVALUATION ---")
best_model = load_model(MODEL_PATH)

test_gen.reset()
preds = best_model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_df["category_encoded_int"].values

class_labels = list(test_gen.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot Confusion Matrix if desired
try:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_fixed.png')
    print("Confusion matrix saved.")
except Exception as e:
    print(f"Could not plot confusion matrix: {e}")