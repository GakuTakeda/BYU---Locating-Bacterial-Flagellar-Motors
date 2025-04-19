import os
import torch
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
import yaml
import pandas as pd
import json
import mlflow
import mlflow.pytorch
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# %%
# 実験パラメータ例
yolo_pretrained_weights = "yolo11l.pt"
epochs = 50
batch_size = 16 
img_size = 960
patience = 100
optimizer='AdamW'
lr0=1e-4
lrf=0.1
warmup_epochs=0
dropout=0.1
augment = True
mosaic=1.0 
close_mosaic=10   
mixup=0.4        
degrees=45       
shear=0         
perspective=0.0   
translate=0     
scale=0.25         
flipud=0.5        
fliplr=0.0         
run_name="yolo11l_train_1"
output_path = os.path.join("output", run_name)
os.makedirs(output_path, exist_ok=True)
# Define paths for Kaggle environment
yolo_dataset_dir =os.path.join("BYU---Locating-Bacterial-Flagellar-Motors/parse_data", "yolo_dataset")
yolo_weights_dir =os.path.join("output", run_name, "yolo_weights")

# %%
def main():
    print("Starting YOLO training process...")

    # Create weights directory if it doesn't exist
    os.makedirs(yolo_weights_dir, exist_ok=True)
    
    # ここでデータセットを準備するとして(例)
    yaml_path = prepare_dataset()
    print(f"Using YAML file: {yaml_path}")

    # Print YAML file contents
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    print(f"YAML file contents:\n{yaml_content}")

    # MLflowで実験管理開始
    mlflow.set_experiment("yolov11l_train_experiment")  # 存在しない場合は自動作成される
    with mlflow.start_run(run_name=run_name):
        # 実験パラメータを記録
        mlflow.log_param("pretrained_weights", yolo_pretrained_weights)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("patience", patience)
        mlflow.log_param("augment", augment)
        mlflow.log_param("mosaic", mosaic)
        mlflow.log_param("mixup", mixup)
        mlflow.log_param("degrees", degrees)
        mlflow.log_param("shear", shear)
        mlflow.log_param("perspective", perspective)
        mlflow.log_param("translate", translate)
        mlflow.log_param("scale", scale)
        mlflow.log_param("flipud", flipud)
        mlflow.log_param("fliplr", fliplr)


        print("\nStarting YOLO training...")
        model, results = train_yolo_model(
            yaml_path,
            pretrained_weights_path=yolo_pretrained_weights,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            patience=patience,
            optimizer='AdamW',
            lr0=1e-4,
            lrf=0.1,
            warmup_epochs=0,
            dropout=0.1,
            augment=augment,      # 全般的なオーグメンテーションを有効化
            mosaic=mosaic,        # mosaic を 70% の確率で適用
            mixup=mixup,        # mixup を 5% の確率で適用
            degrees=degrees,       # 回転の最大角度
            shear=shear,         # シアー(台形変換)の最大量
            perspective=perspective,   # 透視変換の量
            translate=translate,     # 平行移動の範囲 (画像サイズに対する比率)
            scale=scale,         # 拡大縮小の範囲 (0.9 => +/-10% ぐらいを目安)
            flipud=flipud,        # 上下反転の確率
            fliplr=fliplr         # 左右反転の確率 (50%)
        )
        print("\nTraining complete!")

        # ------------------
        # メトリクスの取得例
        # ------------------
        # 以下は想定例です。実際には results 内部にある形式に合わせて読み取ってください。
        # YOLOv8の場合、results.csv (または results ディレクトリ) 内に各Epochの指標が保存されます。
        # best epoch時のメトリクスを拾いたい場合は、csvをパースしたり、resultsオブジェクトから取得します。
        
        # 例: best epoch の val F1 と mAP をログに保存 (実際のキーは実装に合わせて調整)
        best_epoch_f1 = None
        best_epoch_map = None
        if hasattr(results, "metrics"):
            # 例： results.metrics に best_epoch の情報が格納されている仮定
            # 実際には "val/box_f1", "val/box_map50" などあるかもしれません。
            best_epoch_f1 = results.metrics.get("best_f1", None)
            best_epoch_map = results.metrics.get("best_map", None)
            if best_epoch_f1 is not None:
                mlflow.log_metric("best_val_f1", best_epoch_f1)
            if best_epoch_map is not None:
                mlflow.log_metric("best_val_map", best_epoch_map)

        # エポックごとの学習曲線を残したい場合
        # 例: results.history["train_f1"], results.history["val_f1"]などがあれば可視化
        train_f1_values = []
        val_f1_values = []
        train_map_values = []
        val_map_values = []

        if hasattr(results, "history"):
            # 例: results.history が list of dict で、
            # [{"epoch":1, "train_f1":..., "val_f1":..., "train_map":..., "val_map":...}, ...]
            for epoch_data in results.history:
                train_f1_values.append(epoch_data.get("train_f1", 0))
                val_f1_values.append(epoch_data.get("val_f1", 0))
                train_map_values.append(epoch_data.get("train_map", 0))
                val_map_values.append(epoch_data.get("val_map", 0))

            # 学習曲線を保存(例: F1)
            plot_learning_curve(train_f1_values, val_f1_values, "F1", save_path="f1_learning_curve.png")
            mlflow.log_artifact("f1_learning_curve.png")

            # 学習曲線を保存(例: mAP)
            plot_learning_curve(train_map_values, val_map_values, "mAP", save_path="map_learning_curve.png")
            mlflow.log_artifact("map_learning_curve.png")

        # bestモデルファイルを artifact として保存 (weightsディレクトリ等に出力されている想定)
        # 例: "runs/detect/train/weights/best.pt" など
        best_model_path = Path(os.path.join(yolo_weights_dir, "best.pt"))
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path))

        # ---- 推論の実行 ----
        print("\nRunning predictions on sample images...")
        predict_on_samples(model, num_samples=4)

        # 実行が終わったら適宜 mlflow.end_run() は不要(コンテキストマネージャー利用時)。
        print("All done!")

# %%
# Define paths for Kaggle environment
yolo_dataset_dir ="BYU---Locating-Bacterial-Flagellar-Motors/parse_data/yolo_dataset"
yolo_weights_dir ="output/yolo_weights"

# Create weights directory if it doesn't exist
os.makedirs(yolo_weights_dir, exist_ok=True)

def fix_yaml_paths(yaml_path):
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
    fixed_yaml_path = os.path.join("output", run_name, "fixed_dataset.yaml")
    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")
    return fixed_yaml_path

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
    plt.savefig(os.path.join('output',run_name, 'dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    # Return the best epoch info
    return best_epoch, best_val_loss

def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640, patience=40, optimizer='AdamW', lr0=1e-4, lrf=0.1, warmup_epochs=0, dropout=0.1,
                     augment=True, mosaic=0, close_mosaic=0, mixup=0, degrees=0, shear=0, perspective=0, translate=0, scale=0, flipud=0, fliplr=0):
    """
    Train a YOLO model on the prepared dataset
    
    Args:
        yaml_path (str): Path to the dataset YAML file
        pretrained_weights_path (str): Path to pre-downloaded weights file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Image size for training
    """
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    
    # Load a pre-trained YOLOv8 model
    model = YOLO(pretrained_weights_path)
    
    # Train the model with early stopping
    results = model.train(
    data=yaml_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    project=yolo_weights_dir,
    name='motor_detector',
    exist_ok=True,
    patience=patience,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.1,
    warmup_epochs=0,
    dropout=0.1,
    save_period=5,
    val=True,
    verbose=True,
    # --- 以下がオーグメンテーション用のパラメータ例 ---
    augment=augment,      # 全般的なオーグメンテーションを有効化
    mosaic=mosaic, 
    close_mosaic=close_mosaic,       # mosaic を 70% の確率で適用
    mixup=mixup,        # mixup を 5% の確率で適用
    degrees=degrees,       # 回転の最大角度
    shear=shear,         # シアー(台形変換)の最大量
    perspective=perspective,   # 透視変換の量
    translate=translate,     # 平行移動の範囲 (画像サイズに対する比率)
    scale=scale,         # 拡大縮小の範囲 (0.9 => +/-10% ぐらいを目安)
    flipud=flipud,        # 上下反転の確率
    fliplr=fliplr         # 左右反転の確率 (50%)
    )
    
    # Get the path to the run directory
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    
    # Plot and save the loss curve
    best_epoch_info = plot_dfl_loss_curve(run_dir)
    
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
    return model, results

def predict_on_samples(model, num_samples=4):
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
    plt.savefig(os.path.join('/kaggle/working', 'predictions.png'))
    plt.show()

# Check and create a dataset YAML if needed
def prepare_dataset():
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
        return fix_yaml_paths(original_yaml_path)
    else:
        print(f"Original dataset.yaml not found, creating a new one")
        
        # Create a new YAML file
        yaml_data = {
            'path': yolo_dataset_dir,
            'train': 'images/train',
            'val': 'images/train' if not os.path.exists(val_images_dir) else 'images/val',
            'names': {0: 'motor'}
        }
        
        new_yaml_path = "output/dataset.yaml"
        with open(new_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
            
        print(f"Created new YAML at {new_yaml_path}")
        return new_yaml_path

# Main execution
# def main():
#     print("Starting YOLO training process...")
    
#     # Prepare dataset and get YAML path
#     yaml_path = prepare_dataset()
#     print(f"Using YAML file: {yaml_path}")
    
#     # Print YAML file contents
#     with open(yaml_path, 'r') as f:
#         yaml_content = f.read()
#     print(f"YAML file contents:\n{yaml_content}")
    
#     # Train model
#     print("\nStarting YOLO training...")
#     model, results = train_yolo_model(
#         yaml_path,
#         pretrained_weights_path=yolo_pretrained_weights,
#         epochs=epochs,
#         batch_size=batch_size,
#         img_size=img_size,
#         patience=patience
#     )
    
#     print("\nTraining complete!")
    
#     # Run predictions
#     print("\nRunning predictions on sample images...")
#     predict_on_samples(model, num_samples=4)

# if __name__ == "__main__":
#     main()

# %%
def plot_learning_curve(train_metrics, val_metrics, metric_name, save_path="learning_curve.png"):
    """
    学習曲線を描画する簡単な例。
    train_metrics, val_metrics: 各epochごとのmetric値をリストや配列で持っている想定。
    metric_name: "f1", "map"など
    save_path: 保存先のファイル名
    """
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics, label=f"Train {metric_name}")
    plt.plot(epochs, val_metrics, label=f"Val {metric_name}")
    plt.title(f"{metric_name} curve")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# %%
if __name__ == "__main__":
    main()