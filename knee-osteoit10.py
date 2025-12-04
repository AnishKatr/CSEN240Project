#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, 
    Reshape, Multiply, Conv2D, Concatenate, Activation, Add, GlobalMaxPooling2D
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

# CONFIGURATION
# unzipped directory
DATA_PATH = "."  
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VALID_DIR = os.path.join(DATA_PATH, "test")

IMG_SIZE = 320
BATCH_SIZE = 32
MODEL_PATH = "final_retrained_model.keras"

#GPU CONFIG
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# LOAD DATA
def load_data(directory):
    paths = []
    labels = []
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return pd.DataFrame(), []
    
    classes = os.listdir(directory)
    for category in classes:
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(category_path, image_name))
                    labels.append(category)
    
    print(f"Found {len(paths)} images in {directory}")
    return pd.DataFrame({"image_path": paths, "label": labels})

print("--- Loading Datasets ---")
train_df = load_data(TRAIN_DIR)
valid_df = load_data(VALID_DIR)

if len(train_df) == 0 or len(valid_df) == 0:
    print("CRITICAL ERROR: Data missing. Ensure 'train' and 'valid' folders exist.")
    exit(1)

#CLASS WEIGHTS (From Training Data)
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_df['label'])
class_weights_arr = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(train_labels_encoded), y=train_labels_encoded
)
class_weights = dict(enumerate(class_weights_arr))
print(f"Class Weights applied: {class_weights}")

# PREPROCESSING (CLAHE)
def apply_clahe(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float32) / 255.0

#GENERATORS
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
    zoom_range=0.2, brightness_range=[0.8, 1.2],
    horizontal_flip=True, fill_mode="nearest"
)
valid_datagen = ImageDataGenerator(preprocessing_function=apply_clahe)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col="image_path", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)

valid_gen = valid_datagen.flow_from_dataframe(
    dataframe=valid_df, x_col="image_path", y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False 
)

#CBAM + DenseNet121
def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    # Channel Attention
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
    x = Multiply()([input_feature, cbam_feature])
    
    # Spatial Attention
    kernel_size = 7
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    return Multiply()([x, cbam_feature])

def build_model():
    base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Start Frozen
    base_model.trainable = False
    
    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(3, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet_Final_Retrain")
    return model, base_model

model, base_model = build_model()
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# TRAIN HEAD
print("\n--- PHASE 1: Training Head (15 Epochs) ---")
model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss_fn, metrics=["accuracy"])

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
]

model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks_phase1
)

#PHASE 2: FINE TUNING
print("\n--- PHASE 2: Fine Tuning (Up to 50 Epochs) ---")
base_model.trainable = True
total_layers = len(base_model.layers)
# Unfreeze 
fine_tune_at = int(total_layers * 0.40) 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
# Keep BatchNorm frozen
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
]

model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=50,
    class_weight=class_weights,
    callbacks=callbacks_phase2
)

# FINAL EVALUATION
print("\n--- FINAL EVALUATION (TTA) ---")
best_model = load_model(MODEL_PATH)

tta_steps = 5
final_preds = []
true_labels = []

# Use dataframe to iterate so order is preserved matches prediction order logic
for index, row in valid_df.iterrows():
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Create batch of 1 image
    img_batch = np.expand_dims(img, axis=0)
    
    # Create TTA batch (augments the single image 5 times)
    aug_iter = train_datagen.flow(img_batch, batch_size=5)
    
    aug_preds = []
    for _ in range(tta_steps):
        aug_img = next(aug_iter)
        # Predict on augmented image
        pred = best_model.predict(aug_img, verbose=0)[0]
        aug_preds.append(pred)
        
    # Average predictions
    final_preds.append(np.argmax(np.mean(aug_preds, axis=0)))
    true_labels.append(valid_gen.class_indices[row["label"]])

# Reports
target_names = list(valid_gen.class_indices.keys())
print(classification_report(true_labels, final_preds, target_names=target_names))

cm = confusion_matrix(true_labels, final_preds)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Final Validation Confusion Matrix')
plt.savefig('final_validation_matrix.png')
print("Plot saved to final_validation_matrix.png")