#!/usr/bin/env python3

import os
import math
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Reshape,
    Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.applications import EfficientNetB0

warnings.filterwarnings("ignore")

data_path = "."
train_dir = os.path.join(data_path, "train")

categories = ["Normal", "Osteopenia", "Osteoporosis"]

image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(train_dir, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image_paths.append(image_path)
        labels.append(category)

df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.info())
print("Unique labels: {}".format(df["label"].unique()))
print("Label counts: {}".format(df["label"].value_counts()))

# Encode labels, then convert to strings for flow_from_dataframe with sparse mode
label_encoder = LabelEncoder()
df["category_encoded_int"] = label_encoder.fit_transform(df["label"])
df["category_encoded"] = df["category_encoded_int"].astype(str)

print("Encoded label distribution (int):")
print(df["category_encoded_int"].value_counts())
print("Encoded label distribution (str):")
print(df["category_encoded"].value_counts())

# Plot label balance for sanity
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x="label", ax=ax)
ax.set_title("Label distribution")
plt.tight_layout()
plt.savefig("label_distribution_it7.png")
plt.close(fig)

# Stratified split: train / temp, then temp -> valid / test
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["category_encoded"]
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["category_encoded"]
)

print("Train shape:", train_df.shape)
print("Temp shape:", temp_df.shape)
print("Valid shape:", valid_df.shape)
print("Test shape:", test_df.shape)

print("Train label counts:")
print(train_df["category_encoded"].value_counts())
print("Valid label counts:")
print(valid_df["category_encoded"].value_counts())
print("Test label counts:")
print(test_df["category_encoded"].value_counts())

print("check")

# ImageDataGenerator with moderate augmentation
IMG_SIZE = 299  # keep same size as Xception setup
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_gen_new = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="category_encoded",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True
)

valid_gen_new = valid_test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col="image_path",
    y_col="category_encoded",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

test_gen_new = valid_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="category_encoded",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

train_steps_per_epoch = math.ceil(len(train_df) / BATCH_SIZE)
valid_steps_per_epoch = math.ceil(len(valid_df) / BATCH_SIZE)
test_steps_per_epoch = math.ceil(len(test_df) / BATCH_SIZE)

print("Train steps per epoch:", train_steps_per_epoch)
print("Valid steps per epoch:", valid_steps_per_epoch)
print("Test steps per epoch:", test_steps_per_epoch)

NUM_CLASSES = len(categories)
MODEL_PATH = "best_efficientnet_knee_it7.h5"

def se_block(x, reduction=16):
    """Simple squeeze and excitation block for channel attention."""
    channels = int(x.shape[-1])
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // reduction, activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    return Multiply()([x, se])

def build_efficientnet_knee(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape, name="Input_Layer")

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Freeze base for phase 1
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = se_block(x, reduction=16)
    x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
    x = Dropout(0.5, name="Dropout_1")(x)
    x = Dense(256, activation="relu", name="FC_256")(x)
    x = BatchNormalization(name="BatchNorm_1")(x)
    x = Dropout(0.5, name="Dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="Output_Layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_knee_it7")
    return model, base_model

model, base_model = build_efficientnet_knee()
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Phase 1: train new head with frozen EfficientNet
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase1 = model.fit(
    train_gen_new,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=valid_gen_new,
    validation_steps=valid_steps_per_epoch,
    epochs=20,
    callbacks=[early_stop, lr_plateau, checkpoint]
)

# Phase 2: unfreeze top part of EfficientNet for fine tuning
total_layers = len(base_model.layers)
fine_tune_start = int(total_layers * 0.6)

print("Total layers in base model:", total_layers)
print("Fine tuning from layer index:", fine_tune_start)

for i, layer in enumerate(base_model.layers):
    if i < fine_tune_start:
        layer.trainable = False
    else:
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Reset callbacks for phase 2 (fresh patience on early stopping)
early_stop_ft = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_plateau_ft = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

checkpoint_ft = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history_phase2 = model.fit(
    train_gen_new,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=valid_gen_new,
    validation_steps=valid_steps_per_epoch,
    epochs=50,
    callbacks=[early_stop_ft, lr_plateau_ft, checkpoint_ft]
)

# Load best weights and evaluate on test set
best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

test_gen_new.reset()
preds = best_model.predict(test_gen_new, steps=test_steps_per_epoch, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_df["category_encoded_int"].values[: len(y_pred)]

print("Class indices:", test_gen_new.class_indices)
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[str(c) for c in sorted(test_gen_new.class_indices.keys())]
    )
)