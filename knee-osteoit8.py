#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
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
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

# --- GPU SETUP (Cluster Safe) ---
# This prevents the "CUDNN_STATUS_INTERNAL_ERROR" by allowing memory to grow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- CONFIGURATION ---
IMG_SIZE = 320  # INCREASED: Better visibility of bone texture
BATCH_SIZE = 32 # V100 can handle 32 at this res easily
DATA_PATH = "."
TRAIN_DIR = os.path.join(DATA_PATH, "train")
MODEL_PATH = "best_densenet_knee_it8.keras" # Using .keras format (standard for new TF)

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

# Stratified Split
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

# --- 2. CALCULATE CLASS WEIGHTS ---
# We need to map labels to integers first for weight calculation
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_df['label'])
class_weights_arr = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels_encoded),
    y=train_labels_encoded
)
class_weights = dict(enumerate(class_weights_arr))
print("Class Weights applied:", class_weights)

# --- 3. CUSTOM PREPROCESSING (CLAHE) ---
def apply_clahe(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float32) / 255.0

# --- 4. GENERATORS (With Improved Augmentation) ---
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rotation_range=20,        # Increased from 15
    width_shift_range=0.15,   # Increased slightly
    height_shift_range=0.15,
    zoom_range=0.2,           # Increased variability
    brightness_range=[0.8, 1.2], # NEW: Simulates different X-ray exposures
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_test_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe
)

# IMPORTANT: We switch class_mode to 'categorical' (one-hot)
# This is required for Label Smoothing to work properly
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_gen = valid_test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = valid_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --- 5. MODEL BUILD (DenseNet121) ---
def build_densenet_model():
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x) # Increased Dropout (was 0.3) to fight overfitting
    outputs = Dense(3, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet_Knee_IT8")
    return model, base_model

model, base_model = build_densenet_model()

# --- 6. TRAINING PHASE 1 (Head Only) ---
print("\n--- PHASE 1: Training Head ---")

# NEW: Label Smoothing in Loss Function
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=loss_fn,
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
    epochs=12, # Slightly fewer epochs for head training
    class_weight=class_weights,
    callbacks=callbacks
)

# --- 7. TRAINING PHASE 2 (Fine Tuning) ---
print("\n--- PHASE 2: Deep Fine Tuning ---")

base_model.trainable = True
total_layers = len(base_model.layers)

# Unfreeze MORE layers.
# DenseNet is deep. Unfreezing only 15% was conservative.
# We go deeper to let the model learn texture features better.
fine_tune_at = int(total_layers * 0.70) # Unfreeze top 30%

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Keep BatchNorm frozen (standard practice for transfer learning)
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5), # Low LR
    loss=loss_fn, # Keep label smoothing
    metrics=["accuracy"]
)

callbacks_ft = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

history2 = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=40,
    class_weight=class_weights,
    callbacks=callbacks_ft
)

# --- 8. EVALUATION ---
print("\n--- EVALUATION ---")
# Reload best model to ensure we test the absolute best state
best_model = load_model(MODEL_PATH)

test_gen.reset()
preds = best_model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes # Get true labels from generator since we used categorical

# Get class labels from the generator map
labels_map = (test_gen.class_indices)
labels_map = dict((v,k) for k,v in labels_map.items())
target_names = [labels_map[k] for k in sorted(labels_map.keys())]

print(classification_report(y_true, y_pred, target_names=target_names))

# --- 9. CONFUSION MATRIX PLOT ---
try:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (DenseNet IT8)')
    plt.savefig('confusion_matrix_it8.png')
    print("Confusion matrix saved.")
except Exception as e:
    print(f"Could not plot confusion matrix: {e}")