import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm  # Progress bar

# --- CONFIG ---
IMG_SIZE = 256
BATCH_SIZE = 32
DATA_PATH = "."
TRAIN_DIR = os.path.join(DATA_PATH, "train")
MODEL_PATH = "best_densenet_knee_fixed.h5"

# --- REBUILD DATAFRAME (Same as before) ---
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
label_encoder = LabelEncoder()
df["category_encoded_int"] = label_encoder.fit_transform(df["label"])
df["category_encoded"] = df["category_encoded_int"].astype(str)

# Stratified Split (To get the exact same test set)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category_encoded"])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["category_encoded"])

print(f"Test Set Size: {len(test_df)}")

# --- PREPROCESSING (CLAHE) ---
def apply_clahe(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float32) / 255.0

# --- TTA GENERATOR ---
# Notice: We ENABLE augmentation for the test set here
tta_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    rotation_range=15,       # Slight rotation
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

# --- LOAD MODEL ---
model = load_model(MODEL_PATH)
print("Model loaded.")

# --- PERFORM TTA ---
print("Running Test Time Augmentation (5x per image)...")

# We act manually here instead of flow_from_dataframe to control the averaging
tta_steps = 5
final_preds = []
true_labels = []

# Iterate over every image in test set
for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    img_path = row["image_path"]
    true_label = row["category_encoded_int"]
    true_labels.append(true_label)
    
    # Load Image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Create a batch of the SAME image, repeated
    # We rely on the generator to augment it randomly each time we call flow
    img_batch = np.expand_dims(img, axis=0) # Shape (1, H, W, 3)
    
    # Generate 5 augmented versions
    aug_iter = tta_datagen.flow(img_batch, batch_size=tta_steps)
    
    preds_for_this_image = []
    for _ in range(tta_steps):
        aug_img = next(aug_iter) # Get one augmented batch (size 1)
        pred = model.predict(aug_img, verbose=0)
        preds_for_this_image.append(pred[0])
        
    # Average the 5 predictions
    avg_pred = np.mean(preds_for_this_image, axis=0)
    final_preds.append(np.argmax(avg_pred))

# --- REPORT ---
print("\n--- TTA RESULTS ---")
print(classification_report(true_labels, final_preds, target_names=categories))