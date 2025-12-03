#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Paths and settings
DATA_PATH = "."
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VAL_DIR = os.path.join(DATA_PATH, "valid")
TEST_DIR = os.path.join(DATA_PATH, "test")

IMG_SIZE = 320
BATCH_SIZE = 32
MODEL_PATH = "final_retrained_model.keras"


def load_split_df(directory):
    """Return a DataFrame with columns: image_path, label."""
    image_paths = []
    labels = []

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, fname)
                # label is the immediate parent directory name
                label = os.path.basename(os.path.dirname(full_path))
                image_paths.append(full_path)
                labels.append(label)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {directory}")

    return pd.DataFrame({"image_path": image_paths, "label": labels})


def make_eval_generator(df, target_size, batch_size):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="image_path",
        y_col="label",
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return gen


def eval_split(model, split_name, split_dir):
    print(f"\n===== Evaluating on {split_name} set from {split_dir} =====")
    df = load_split_df(split_dir)
    gen = make_eval_generator(df, IMG_SIZE, BATCH_SIZE)

    # Predict
    preds = model.predict(gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = gen.classes

    # Map indices to class names
    idx_to_class = {v: k for k, v in gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix per split
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    png_name = f"cm_{split_name}.png"
    plt.savefig(png_name)
    print(f"Saved confusion matrix to {png_name}")
    plt.close()


def main():
    print(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    eval_split(model, "train", TRAIN_DIR)
    eval_split(model, "val", VAL_DIR)
    eval_split(model, "test", TEST_DIR)


if __name__ == "__main__":
    main()