import os
import yaml
import git
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
import random
from PIL import Image
import timm
import torch
import torch.nn as nn
class TimmBackbone(nn.Module):
    def __init__(self, model_name, pretrained, out_indices):
        super().__init__()
        self.body = timm.create_model(
            model_name, pretrained=pretrained,
            features_only=True, out_indices=out_indices
        )
    def forward(self, x):
        return self.body(x)

def fix_yaml_paths(yaml_path, yolo_dataset_dir):
    """
    Fix the paths in the YAML file to match the actual Kaggle directories
    
    Args:
        yaml_path (str): Path to the original dataset YAML file
        
    Returns:
        str: Path to the fixed YAML file
    """
    print(f"Fixing YAML paths in {yaml_path}")
    
    # Read the original YAML
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Update paths to use actual dataset location
    if 'path' in yaml_data:
        yaml_data['path'] = yolo_dataset_dir
    
    # Create a new fixed YAML in the working directory
    fixed_yaml_path = os.path.join("output", "fixed_dataset.yaml")
    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")
    return fixed_yaml_path

def prepare_dataset(yolo_dataset_dir, output_dir):
    """
    Check if dataset exists and create a proper YAML if needed
    
    Returns:
        str: Path to the YAML file to use for training
    """
    # Check if images exist
    train_images_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
    val_images_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
    val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'val')
    
    # Print directory existence status
    print(f"Directory status:")
    print(f"- Train images dir exists: {os.path.exists(train_images_dir)}")
    print(f"- Val images dir exists: {os.path.exists(val_images_dir)}")
    print(f"- Train labels dir exists: {os.path.exists(train_labels_dir)}")
    print(f"- Val labels dir exists: {os.path.exists(val_labels_dir)}")
    
    # Check for original YAML file
    original_yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')
    
    if os.path.exists(original_yaml_path):
        print(f"Found original dataset.yaml at {original_yaml_path}")
        # Fix the paths in the YAML
        return original_yaml_path
    else:
        print(f"Original dataset.yaml not found, creating a new one")
        
        # Create a new YAML file
        yaml_data = {
            'path': yolo_dataset_dir,
            'train': 'images/train',
            'val': 'images/train' if not os.path.exists(val_images_dir) else 'images/val',
            'names': {0: 'motor'}
        }
        
        new_yaml_path = os.path.join(output_dir, "dataset.yaml")
        with open(new_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
            
        print(f"Created new YAML at {new_yaml_path}")
        return new_yaml_path
    
def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for train and validation, marking the best model
    
    Args:
        run_dir (str): Directory where the training results are stored
    """
    # Path to the results CSV file
    results_csv = os.path.join(run_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    # Read results CSV
    results_df = pd.read_csv(results_csv)
    
    # Check if DFL loss columns exist
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    # Find the epoch with the best validation loss
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    
    # Mark the best model with a vertical line
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    
    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot in the same directory as weights
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)
    
    # Also save it to the working directory for easier access
    plt.savefig(os.path.join('output','dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    # Return the best epoch info
    return best_epoch, best_val_loss

def train_yolo_model(
    yaml_path,
    pretrained_weights_path,
    project_name,
    run_name,
    backbone,      #追加した
    out_indices,   #追加した
    pre_trained=True,#追加した
    epochs=30,
    freeze_backbone_epochs=10,   
    batch_size=16,
    img_size=640,
    patience=40,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    dropout=0.1,
    weight_decay=0.0005,
    augment=True, mosaic=0, close_mosaic=0, mixup=0,
    degrees=0, shear=0, perspective=0, translate=0,
    scale=0, flipud=0, fliplr=0,
    cos_lr=False, amp=True,
    ):
    # モデル読み込み
    model = YOLO(pretrained_weights_path)
    custom_bb = TimmBackbone(model_name=backbone, pretrained=pre_trained, out_indices=out_indices)
    model.model.backbone = custom_bb

    results = model.train(
            data=yaml_path,
            save_period=1,
            epochs=epochs,
            freeze=freeze_backbone_epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project_name,
            name=run_name,
            exist_ok=True,
            patience=patience,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=warmup_epochs,
            dropout=dropout,
            weight_decay=weight_decay,
            augment=augment,
            mosaic=mosaic,
            close_mosaic=close_mosaic,
            mixup=mixup,
            degrees=degrees,
            shear=shear,
            perspective=perspective,
            translate=translate,
            scale=scale,
            flipud=flipud,
            fliplr=fliplr,
            cos_lr=cos_lr,
            amp=amp,
        )
    
    # Get the path to the run directory
    run_dir = os.path.join(os.getcwd(), "42", run_name)

    plot_dfl_loss_curve(run_dir)
    
    return model, results
def predict_on_samples(model, yolo_dataset_dir, num_samples=4):
    """
    Run predictions on random validation samples and display results
    
    Args:
        model: Trained YOLO model
        num_samples (int): Number of random samples to test
    """
    # Get validation images
    val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found at {val_dir}")
        # Try train directory instead if val doesn't exist
        val_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
        print(f"Using train directory for predictions instead: {val_dir}")
        
    if not os.path.exists(val_dir):
        print("No images directory found for predictions")
        return
    
    val_images = os.listdir(val_dir)
    
    if len(val_images) == 0:
        print("No images found for prediction")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_file in enumerate(samples):
        if i >= len(axes):
            break
            
        img_path = os.path.join(val_dir, img_file)
        
        # Run prediction
        results = model.predict(img_path, conf=0.25)[0]
        
        # Load and display the image
        img = Image.open(img_path)
        axes[i].imshow(np.array(img), cmap='gray')
        
        # Draw ground truth box if available (from filename)
        try:
            # This assumes your filenames contain coordinates in a specific format
            parts = img_file.split('_')
            y_part = [p for p in parts if p.startswith('y')]
            x_part = [p for p in parts if p.startswith('x')]
            
            if y_part and x_part:
                y_gt = int(y_part[0][1:])
                x_gt = int(x_part[0][1:].split('.')[0])
                
                box_size = 24
                rect_gt = Rectangle((x_gt - box_size//2, y_gt - box_size//2), 
                              box_size, box_size, 
                              linewidth=1, edgecolor='g', facecolor='none')
                axes[i].add_patch(rect_gt)
        except:
            pass  # Skip ground truth if parsing fails
        
        # Draw predicted boxes (red)
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
                axes[i].text(x1, y1-5, f'{conf:.2f}', color='red')
        
        axes[i].set_title(f"Image: {img_file}\nGround Truth (green) vs Prediction (red)")
    
    plt.tight_layout()
    
    # Save the predictions plot
    plt.savefig(os.path.join('output', 'predictions.png'))
    plt.show()

def git_commits(rand):
    def func_decorator(my_func):
        print("experiment_name: ", rand)

        repo = git.Repo(str(Path(os.getcwd()).parents[0]))
        repo.git.diff("HEAD")
        repo.git.add(".")
        repo.index.commit(f"{rand}(before running)")

        def decorator_wrapper(*args, **kwargs):
            my_func(*args, **kwargs)

            repo.index.commit(f"{rand}(after running)")
            repo.git.push('origin', 'master')
        return decorator_wrapper

    return func_decorator