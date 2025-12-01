# %%
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Set data path relative to where you submit the job
data_path = "."

categories = ["Normal", "Osteopenia", "Osteoporosis"]

image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(data_path, "train", category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image_paths.append(image_path)
        labels.append(category)

# %%
df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.shape)

# Basic checks
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.info())
print("Unique labels: {}".format(df["label"].unique()))
print("Label counts: {}".format(df["label"].value_counts()))

# %%
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x="label", palette="viridis", ax=ax)
ax.set_title("Distribution of Tumor Types", fontsize=14, fontweight="bold")
ax.set_xlabel("Tumor Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )
# plt.show()

# %%
label_counts = df["label"].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
colors = sns.color_palette("viridis", len(label_counts))
ax.pie(
    label_counts,
    labels=label_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops={"fontsize": 12, "weight": "bold"},
    wedgeprops={"edgecolor": "black", "linewidth": 1},
)
ax.set_title("Distribution of Tumor Types - Pie Chart", fontsize=14, fontweight="bold")
# plt.show()

# %%
# Show a few sample images per class
num_images = 5
plt.figure(figsize=(15, 12))
for i, category in enumerate(categories):
    category_images = df[df["label"] == category]["image_path"].iloc[:num_images]
    for j, img_path in enumerate(category_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(len(categories), num_images, i * num_images + j + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(category)

plt.tight_layout()
plt.show(block=False)
plt.pause(5)
plt.close()

# %%
# Encode labels as integers, then cast to string for Keras on WAVE
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["label"])
print("Encoded label distribution (int):")
print(df["category_encoded"].value_counts())

# Keras on this cluster expects string labels when class_mode="sparse"
df["category_encoded"] = df["category_encoded"].astype(str)
df = df[["image_path", "category_encoded"]]

print("Encoded label distribution (str):")
print(df["category_encoded"].value_counts())

# %%
# Train / val / test split on the original balanced dataframe (no oversampling)
from sklearn.model_selection import train_test_split

train_df_new, temp_df_new = train_split = train_test_split(
    df,
    train_size=0.8,
    shuffle=True,
    random_state=42,
    stratify=df["category_encoded"],
)
print("Train shape:", train_df_new.shape)
print("Temp shape:", temp_df_new.shape)

valid_df_new, test_df_new = train_test_split(
    temp_df_new,
    test_size=0.5,
    shuffle=True,
    random_state=42,
    stratify=temp_df_new["category_encoded"],
)
print("Valid shape:", valid_df_new.shape)
print("Test shape:", test_df_new.shape)

print("Train label counts:")
print(train_df_new["category_encoded"].value_counts())
print("Valid label counts:")
print(valid_df_new["category_encoded"].value_counts())
print("Test label counts:")
print(test_df_new["category_encoded"].value_counts())

# %%
import time
import shutil
import pathlib
import itertools
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings("ignore")
print("check")

# %%
# Hyperparameters
batch_size = 32
img_size = (299, 299)  # native size for Xception
channels = 3
input_shape = (img_size[0], img_size[1], channels)

from tensorflow.keras.applications.xception import preprocess_input

# Use Xception's preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen_new = train_datagen.flow_from_dataframe(
    train_df_new,
    x_col="image_path",
    y_col="category_encoded",
    target_size=img_size,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size,
)

valid_gen_new = valid_datagen.flow_from_dataframe(
    valid_df_new,
    x_col="image_path",
    y_col="category_encoded",
    target_size=img_size,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size,
)

test_gen_new = test_datagen.flow_from_dataframe(
    test_df_new,
    x_col="image_path",
    y_col="category_encoded",
    target_size=img_size,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size,
)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print("Train steps per epoch:", len(train_gen_new))
print("Valid steps per epoch:", len(valid_gen_new))
print("Test steps per epoch:", len(test_gen_new))

# %%
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    min_delta=1e-3,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    "best_xception_knee_it6.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

# %%
def build_xception_model(input_shape, num_classes=3, base_trainable=False):
    inputs = Input(shape=input_shape, name="Input_Layer")

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )
    base_model.trainable = base_trainable

    x = base_model.output
    x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)

    # Lighter but still strong head
    x = Dropout(0.3, name="Dropout_1")(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(5e-5),
        name="FC_256",
    )(x)
    x = BatchNormalization(name="BatchNorm_1")(x)
    x = Dropout(0.2, name="Dropout_2")(x)

    outputs = Dense(num_classes, activation="softmax", name="Output_Layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Xception_knee_it6")

    loss_fn = SparseCategoricalCrossentropy()

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model, base_model

cnn_model, base_model = build_xception_model(
    input_shape=input_shape,
    num_classes=3,
    base_trainable=False,  # first phase: frozen backbone
)

cnn_model.summary()

# %%
# Phase 1: train only the new head
epochs_phase1 = 20

history_phase1 = cnn_model.fit(
    train_gen_new,
    validation_data=valid_gen_new,
    epochs=epochs_phase1,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
)

# Load best weights from phase 1 before fine tuning
cnn_model.load_weights("best_xception_knee_it6.h5")

# %%
# Phase 2: unfreeze more of Xception and fine tune with lower learning rate
base_model.trainable = True

# Freeze first 40 percent of layers instead of 60 percent
total_layers = len(base_model.layers)
fine_tune_at = int(total_layers * 0.4)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print("Total layers in base model:", total_layers)
print("Fine tuning from layer index:", fine_tune_at)

# Recompile with lower learning rate
loss_fn = SparseCategoricalCrossentropy()
cnn_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=["accuracy"],
)

epochs_phase2 = 30
initial_epoch = len(history_phase1.history["loss"])

history_phase2 = cnn_model.fit(
    train_gen_new,
    validation_data=valid_gen_new,
    epochs=initial_epoch + epochs_phase2,
    initial_epoch=initial_epoch,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
)

# Load best overall weights
cnn_model.load_weights("best_xception_knee_it6.h5")

# %%
# Accuracy and loss plots for both phases
all_train_acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
all_val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
all_train_loss = history_phase1.history["loss"] + history_phase2.history["loss"]
all_val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]

plt.figure()
plt.plot(all_train_acc)
plt.plot(all_val_acc)
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show(block=False)
plt.pause(5)
plt.close()

plt.figure()
plt.plot(all_train_loss)
plt.plot(all_val_loss)
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show(block=False)
plt.pause(5)
plt.close()

# %%
# Test set evaluation
test_labels = test_gen_new.classes
predictions = cnn_model.predict(test_gen_new)
predicted_classes = np.argmax(predictions, axis=1)

target_names = list(test_gen_new.class_indices.keys())
print("Class indices:", test_gen_new.class_indices)

report = classification_report(
    test_labels,
    predicted_classes,
    target_names=target_names,
    digits=4,
)
print(report)

conf_matrix = confusion_matrix(test_labels, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show(block=False)
plt.pause(5)
plt.close()