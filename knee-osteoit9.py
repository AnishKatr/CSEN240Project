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
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, 
    Reshape, Multiply, Conv2D, Concatenate, Activation, Add, GlobalMaxPooling2D
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

# --- GPU SETUP ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- CONFIGURATION ---
IMG_SIZE = 320
BATCH_SIZE = 32
DATA_PATH = "."
TRAIN_DIR = os.path.join(DATA_PATH, "train")
MODEL_PATH = "best_densenet_attention_it9.keras"

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
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

# --- 2. CALCULATE CLASS WEIGHTS ---
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_df['label'])
class_weights_arr = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(train_labels_encoded), y=train_labels_encoded
)
class_weights = dict(enumerate(class_weights_arr))

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

# --- 4. GENERATORS ---
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col="image_path", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)

valid_gen = valid_test_datagen.flow_from_dataframe(
    dataframe=valid_df, x_col="image_path", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

test_gen = valid_test_datagen.flow_from_dataframe(
    dataframe=test_df, x_col="image_path", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

# --- 5. ATTENTION BLOCKS (CBAM) ---
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    return Multiply()([input_feature, cbam_feature])

def cbam_block(input_feature, ratio=8):
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x)
    return x

# --- 6. MODEL BUILD ---
def build_model():
    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze base
    base_model.trainable = False
    
    x = base_model.output
    
    # === INSERT ATTENTION HERE ===
    # This forces the model to look at the important texture areas
    x = cbam_block(x)
    # =============================
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(3, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet_Attention")
    return model, base_model

model, base_model = build_model()

# --- 7. TRAINING ---
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Phase 1: Head
print("\n--- PHASE 1: Training Head ---")
model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss_fn, metrics=["accuracy"])
model.fit(
    train_gen, validation_data=valid_gen, epochs=20, 
    class_weight=class_weights, 
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=2)]
)

# Phase 2: Fine Tuning
print("\n--- PHASE 2: Deep Fine Tuning ---")
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.60) # Train top 40%

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])

callbacks_ft = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

model.fit(train_gen, validation_data=valid_gen, epochs=100, class_weight=class_weights, callbacks=callbacks_ft)

# --- 8. EVALUATION & TTA ---
print("\n--- EVALUATION (Standard) ---")
best_model = load_model(MODEL_PATH)
test_gen.reset()
preds = best_model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
print(classification_report(test_gen.classes, y_pred, target_names=list(test_gen.class_indices.keys())))

# --- TTA EVALUATION (The Booster) ---
print("\n--- EVALUATION (TTA) ---")
tta_steps = 5
final_preds = []
true_labels = []

# Manual TTA Loop
for index, row in df.iloc[test_df.index].iterrows():
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Ensure correct size
    
    # 5 augmentations
    aug_preds = []
    img_batch = np.expand_dims(img, axis=0)
    aug_iter = train_datagen.flow(img_batch, batch_size=5) # Use train_datagen for augmentations
    
    for _ in range(tta_steps):
        aug_img = next(aug_iter)
        aug_preds.append(best_model.predict(aug_img, verbose=0)[0])
        
    final_preds.append(np.argmax(np.mean(aug_preds, axis=0)))
    true_labels.append(test_gen.class_indices[row["label"]])

print(classification_report(true_labels, final_preds, target_names=list(test_gen.class_indices.keys())))