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
from sklearn.model_selection import train_test_split
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define Kaggle paths
data_path = "../input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Define YOLO dataset structure
yolo_dataset_dir = "dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Define constants
TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
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

def prepare_train_dataset(trust=TRUST):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] > 1].copy()
    unique_tomos = tomo_df['tomo_id'].unique()

    # Function to process a set of tomograms
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
                     int(motor['Array shape (axis 0)']),
                     int(motor['Array shape (axis 1)']),
                     int(motor['Array shape (axis 2)']),
                     int(motor['Voxel spacing'])
                ))
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max, y_max, x_max, space in tqdm(motor_counts, desc=f"Processing {set_name} motors"):

            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)

            x_space = 255 * space / x_max 
            y_space = 255 * space / y_max 
            
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
                x_space_norm = x_space / img_width
                y_space_norm = y_space / img_height
                
                # Write label file
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"x_center y_center x_space y_space\n")
                    f.write(f"{x_center_norm} {y_center_norm} {x_space_norm} {y_space_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    process_tomogram_set(unique_tomos, yolo_images_train, yolo_labels_train, "training")

def prepare_val_dataset(trust=TRUST):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df_0 = labels_df[labels_df['Number of motors'] == 0].copy()
    tomo_df_1 = labels_df[labels_df['Number of motors'] == 1].copy()
    tomos   = tomo_df_1['tomo_id'].unique()
    unique_tomos_0 = tomo_df_0['tomo_id'].unique()
    unique_tomos_1 = tomo_df_1['tomo_id'].unique()

    np.random.shuffle(tomos)
    split = int(len(tomos) * 0.7)
    train_tomos, val_tomos = tomos[:split], tomos[split:]

    val_0_tomos = tomo_df_0['tomo_id'].unique()
    np.random.shuffle(val_0_tomos)

    yolo_list = os.path.join('dataset', 'yolo_list')
    with open(yolo_list, 'w') as f:                # ← ここで一度だけ 'w'
        for name in val_0_tomos:
            f.write(f"{name}\n")
        for name in val_tomos:
            f.write(f"{name}\n")
    # Function to process a set of tomograms
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
                     int(motor['Array shape (axis 0)']),
                     int(motor['Array shape (axis 1)']),
                     int(motor['Array shape (axis 2)']),
                     int(motor['Voxel spacing'])
                ))
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max, y_max, x_max, space in tqdm(motor_counts, desc=f"Processing {set_name} motors"):

            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)

            x_space = 255 * space / x_max 
            y_space = 255 * space / y_max
            
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
                x_space_norm = x_space / img_width
                y_space_norm = y_space / img_height
                dif_from_center = np.abs(z_center - z)
                
                # Write label file
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"x_center y_center x_space y_space dif_from_center\n")
                    f.write(f"{x_center_norm} {y_center_norm} {x_space_norm} {y_space_norm} {dif_from_center}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    def process_0_tomogram(tomogram_ids, images_dir, labels_dir, n):
        per_number = int(len(tomogram_ids) / n)
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Array shape (axis 0)'])
                ))

        # Process each motor
        processed_slices = 0

        
        for tomo_id, z_max in tqdm(motor_counts, desc=f"Processing non motors"):

            start = random.randint(0, z_max - per_number)
   
            # Process each slice in the range
            for z in range(start,  start+per_number):
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
                dest_filename = f"{tomo_id}_z{z:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)

                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"x_center y_center x_space y_space dif_from_center\n")
                    f.write(f"{-1} {-1} {-1} {-1} {-1}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    _, n_train = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    _, n = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "val")
    _, k = process_0_tomogram(val_0_tomos, yolo_images_val, yolo_labels_val, n)
    print(f"trainingデータの数:{n_train}validationデータの数:{n+k}(zoro_motor{k})")
    
if __name__ == "__main__":
    prepare_train_dataset()
    prepare_val_dataset()