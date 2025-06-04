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

@hydra.main(config_path="config", config_name="config")
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
    yaml_path = prepare_dataset("/mnt/c/Users/tkdgk/BYU---Locating-Bacterial-Flagellar-Motors/parse_data/yolo_dataset", os.path.join(orig_cwd, "output"))
    print(f"Using YAML file: {yaml_path}")

    model, results = train_yolo_model(
            yaml_path,
            pretrained_weights_path=cfg.yolo.pretrained_weights,
            project_name=str(cfg.yolo.seed),
            run_name=run_name,
            backbone=cfg.backbone.backbone,
            out_indices=cfg.backbone.out_indices,
            pre_trained=cfg.backbone.pre_trained,
            epochs=cfg.yolo.epochs,
            batch_size=cfg.yolo.batch_size,
            img_size=cfg.yolo.img_size,
            patience=cfg.yolo.patience,

            optimizer=cfg.yolo.optimizer,
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
    print("Training complete!")


    print("\nRunning predictions on sample images...")
    predict_on_samples(model, "/mnt/c/Users/tkdgk/BYU---Locating-Bacterial-Flagellar-Motors/parse_data/yolo_dataset", num_samples=4)

if __name__ == "__main__":
    main()