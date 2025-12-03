#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "."
TRAIN_DIR = os.path.join(DATA_PATH, "train")
VAL_DIR   = os.path.join(DATA_PATH, "val")
TEST_DIR  = os.path.join(DATA_PATH, "test")

IMG_SIZE = 320
BATCH_SIZE = 32
MODEL_PATH = "final_retrained_model.keras"

def apply_clahe(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float32) / 255.0

def load_split_df(directory):
    paths = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, fname)
                label = os.path.basename(os.path.dirname(full_path))
                paths.append(full_path)
                labels.append(label)
    if not paths:
        raise RuntimeError(f"No images found in {directory}")
    return pd.DataFrame({"image_path": paths, "label": labels})

def make_generator(df):
    datagen = ImageDataGenerator(preprocessing_function=apply_clahe)
    gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return gen

def eval_split(model, split_name, split_dir):
    print(f"\n===== Evaluating on {split_name} set from {split_dir} =====")
    df = load_split_df(split_dir)
    gen = make_generator(df)

    preds = model.predict(gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = gen.classes

    idx_to_class = {v: k for k, v in gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

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
    png_name = f"cm_{split_name}_clahe.png"
    plt.title(f"{split_name.capitalize()} Confusion Matrix (CLAHE)")
    plt.savefig(png_name)
    plt.close()
    print(f"Saved confusion matrix to {png_name}")

def main():
    print(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    eval_split(model, "train", TRAIN_DIR)
    eval_split(model, "val",   VAL_DIR)
    eval_split(model, "test",  TEST_DIR)

if __name__ == "__main__":
    main()
