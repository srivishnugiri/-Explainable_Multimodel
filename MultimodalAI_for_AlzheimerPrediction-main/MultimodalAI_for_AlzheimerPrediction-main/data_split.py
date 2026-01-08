import os
import shutil
import random

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "dataset_split"
TRAIN_RATIO = 0.8  # 80% train, 20% test

# Create output structure
train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Go through each category folder
for category in os.listdir(DATA_DIR):
    category_path = os.path.join(DATA_DIR, category)
    if not os.path.isdir(category_path):
        continue

    # Create subfolders for train/test
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Get all files in that category
    files = os.listdir(category_path)
    random.shuffle(files)

    # Split
    split_index = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_index]
    test_files = files[split_index:]

    # Move or copy files
    for f in train_files:
        shutil.copy(os.path.join(category_path, f), os.path.join(train_dir, category, f))
    for f in test_files:
        shutil.copy(os.path.join(category_path, f), os.path.join(test_dir, category, f))

print("✅ Dataset split completed successfully!")
