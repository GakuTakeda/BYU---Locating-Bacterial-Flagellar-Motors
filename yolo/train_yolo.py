import os
import torch
import numpy as np
import random
from PIL import Image
import yaml
import pandas as pd
import json
import mlflow
import mlflow.pytorch
from pathlib import Path
import hydra
from omegaconf import DictConfig
from utils import prepare_dataset, train_yolo_model, predict_on_samples
from ultralytics import YOLO
import matplotlib.pyplot as plt

@hydra.main(config_path="config", config_name="config11l")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    np.random.seed(cfg.yolo.seed)
    random.seed(cfg.yolo.seed)
    torch.manual_seed(cfg.yolo.seed)

    # ── 2. パス周りを設定 ──
    run_name = cfg.yolo.run_name
    output_path     = os.path.join(orig_cwd, cfg.yolo.paths.output_path)
    yolo_weights_dir = os.path.join(orig_cwd, cfg.yolo.paths.yolo_weights_dir)
    os.makedirs(yolo_weights_dir, exist_ok=True)

    # ── 3. データセット準備 ──
    yaml_path = prepare_dataset(os.path.join("/mnt/c/Users/tkdgk/BYU---Locating-Bacterial-Flagellar-Motors", cfg.yolo.paths.yolo_dataset_dir), os.path.join(orig_cwd, "output"))
    print(f"Using YAML file: {yaml_path}")

    model, results, run_dir = train_yolo_model(
            yaml_path,
            pretrained_weights_path=cfg.yolo.pretrained_weights,
            project_name=str(cfg.yolo.seed),
            run_name=run_name,
            epochs=cfg.yolo.epochs,
            batch_size=cfg.yolo.batch_size,
            img_size=cfg.yolo.img_size,
            patience=cfg.yolo.patience,

            optimizer=cfg.yolo.optimizer,
            momentum=cfg.yolo.momentum,
            lr0=cfg.yolo.lr0,
            lrf=cfg.yolo.lrf,
            warmup_epochs=cfg.yolo.warmup_epochs,
            dropout=cfg.yolo.dropout,
            weight_decay=cfg.yolo.weight_decay,

            augment=cfg.yolo.augment,
            mosaic=cfg.yolo.mosaic,
            mixup=cfg.yolo.mixup,

            degrees=cfg.yolo.degrees,
            shear=cfg.yolo.shear,
            perspective=cfg.yolo.perspective,
            translate=cfg.yolo.translate,
            scale=cfg.yolo.scale,
            flipud=cfg.yolo.flipud,
            fliplr=cfg.yolo.fliplr,
            cos_lr=cfg.yolo.cos_lr,
            amp=cfg.yolo.amp
        )
    
    csv_path = os.path.join(run_dir, "results.csv")
    df = pd.read_csv(csv_path)

    p = df["metrics/precision(B)"]
    r = df["metrics/recall(B)"]
    df["F1"] = 2 * p * r / (p + r)
    df["F2"] = 5 * p * r / (4*p + r)
    best_f2_index = df["F2"].idxmax()
    best_f2_epoch = f"epoch{int(df.loc[best_f2_index, "epoch"])}.pt"   
    best_f2_model = YOLO(os.path.join(run_dir, "weights", best_f2_epoch))

    result = best_f2_model.val(
        data=yaml_path,
        batch=12,
        plots=True

    )
    
    best_epoch = df["F2"].idxmax()

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["F2"])
    plt.title("F2_plot")

    plt.axvline(x=df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(df.loc[best_epoch, "epoch"])}, F2: {df.loc[best_epoch, "F2"]:.4f})')

    plt.xlabel("epoch")
    plt.ylabel("F2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(run_dir, 'Plot_F2.png')
    plt.savefig(plot_path)
    plt.close()
    print("Training complete!")

if __name__ == "__main__":
    main()