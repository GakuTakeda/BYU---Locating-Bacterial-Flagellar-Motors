# %%
import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments
from sklearn.model_selection import StratifiedKFold, KFold
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define Kaggle paths
data_path = "../input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Define YOLO dataset structure
yolo_dataset_dir = "yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Define constants
TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# Image processing functions
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)
    
    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    
    return np.uint8(normalized)
def add_val(fold, mode, trust=TRUST):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] == 0].copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    unique_tomos = tomo_df['tomo_id'].unique()
    splits_0 = list(kf.split(unique_tomos))
    # 2) 今回のfold用のタプルをアンパック
    train_0_index, val_0_index = splits_0[fold]
    
    print(f"Found {len(unique_tomos)} unique tomograms with motors")

    def process_tomogram_set_0(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):

            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)

            z_start = random.randint(0, 292)
            
            # Process each slice in the range
            for z in range(z_start, z_start+8):
                # Create slice filename
                slice_filename = f"slice_{z:04d}.jpg"
                
                # Source path for the slice
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load and normalize the slice
                img = Image.open(src_path)
                img_array = np.array(img)
                
                # Normalize the image
                normalized_img = normalize_slice(img_array)
                
                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)

                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w'): 
                    pass
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)

    if mode == "train":
        index = train_0_index[fold]
    else:
        index = val_0_index[fold]
    
    process_tomogram_set_0(unique_tomos[index], yolo_images_val, yolo_labels_val, "validation")


def prepare_yolo_fold(labels_df, unique_tomos, train_tomos, val_tomos, fold, trust):
    # フォルダ名をfoldごとに分ける
    dataset_dir      = f"yolo_dataset_{fold}"
    images_train_dir = os.path.join(dataset_dir, "images", "train")
    images_val_dir   = os.path.join(dataset_dir, "images", "val")
    labels_train_dir = os.path.join(dataset_dir, "labels", "train")
    labels_val_dir   = os.path.join(dataset_dir, "labels", "val")

    # ディレクトリ作成
    for p in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(p, exist_ok=True)

    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):

            if z_center == -1:
                continue
            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)
            
            # Process each slice in the range
            for z in range(z_min, z_max + 1):
                # Create slice filename
                slice_filename = f"slice_{z:04d}.jpg"
                
                # Source path for the slice
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load and normalize the slice
                img = Image.open(src_path)
                img_array = np.array(img)
                
                # Normalize the image
                normalized_img = normalize_slice(img_array)
                
                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Create YOLO format label
                # YOLO format: <class> <x_center> <y_center> <width> <height>
                # Values are normalized to [0, 1]
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height
                
                # Write label file
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)

    # process_tomogram_set は元のまま再利用
    train_slices, train_motors = process_tomogram_set(
        train_tomos, images_train_dir, labels_train_dir, f"fold{fold}-train"
    )
    val_slices,   val_motors   = process_tomogram_set(
        val_tomos,   images_val_dir,   labels_val_dir,   f"fold{fold}-val"
    )

    # dataset.yaml も書き出し
    yaml_content = {
        'path': dataset_dir,
        'train': 'images/train',
        'val':   'images/val',
        'names': {0: 'motor'}
    }
    with open(os.path.join(dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    add_val(fold, mode="train", trust=TRUST)
    add_val(fold, mode="val", trust=TRUST)

    print(f"[fold {fold}] 完了: {train_motors}+{val_motors} motors, {train_slices}+{val_slices} slices")


def main():
    labels_df    = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    unique_tomos = labels_df.loc[labels_df['Number of motors'] > 0, 'tomo_id'].unique()

    counts = labels_df.groupby('tomo_id')['Number of motors'].sum()
    # unique_tomos の順で取り出す
    y = counts.reindex(unique_tomos).values

    # KFold で5分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(unique_tomos, y)):
        train_tomos = unique_tomos[train_idx]
        val_tomos   = unique_tomos[val_idx]
        prepare_yolo_fold(labels_df, unique_tomos, train_tomos, val_tomos, fold, TRUST)


if __name__ == "__main__":
    main()