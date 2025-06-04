import os
import torch
import numpy as np
import random
from sklearn.model_selection import KFold
from pathlib import Path
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

@hydra.main(config_path="config", config_name="config11l")
def main(cfg: DictConfig):
    # reproducibility
    np.random.seed(cfg.yolo.seed)
    random.seed(cfg.yolo.seed)
    torch.manual_seed(cfg.yolo.seed)

    orig_cwd = hydra.utils.get_original_cwd()
    dataset_dir = os.path.join(orig_cwd, "..", cfg.yolo.paths.yolo_dataset_dir)

    # collect image files
    img_paths = sorted(Path(dataset_dir, 'images').glob('*.*'))
    assert img_paths, f"No images found in {dataset_dir}/images"
    kf = KFold(n_splits=cfg.yolo.k_folds, shuffle=True, random_state=cfg.yolo.seed)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_paths), start=1):
        print(f"\n=== Fold {fold}/{cfg.yolo.k_folds} ===")
        # prepare subset YAML
        train_list = [str(img_paths[i]) for i in train_idx]
        val_list   = [str(img_paths[i]) for i in val_idx]
        data_dict = {
            'train': train_list,
            'val':   val_list,
            'nc':    1,
            'names': {0: 'motor'}
        }
        # save temporary data yaml
        fold_yaml = Path(orig_cwd) / f"fold_{fold}.yaml"
        with open(fold_yaml, 'w') as f:
            import yaml
            yaml.safe_dump(data_dict, f)
        print(f"Data YAML for fold {fold}: {fold_yaml}")

        # initialize model
        model = YOLO(cfg.yolo.pretrained_weights)

        # train
        _ = model.train(
            data=str(fold_yaml),
            epochs=cfg.yolo.epochs,
            freeze=cfg.yolo.freeze_backbone_epochs,
            batch=cfg.yolo.batch_size,
            imgsz=cfg.yolo.img_size,
            project=str(cfg.yolo.seed),
            name=f"{cfg.yolo.run_name}_fold{fold}",
            exist_ok=True,
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
        # validation
        val_metrics = model.val(
            data=str(fold_yaml),
            batch=cfg.yolo.batch_size,
            imgsz=cfg.yolo.img_size
        )
        # extract metrics
        precision = val_metrics.metrics.precision
        recall = val_metrics.metrics.recall
        # compute F2 score: (1+2^2)*P*R / (2^2*P + R)
        beta2 = 2.0
        f2_score = (1 + beta2**2) * precision * recall / (beta2**2 * precision + recall + 1e-8)

        # collect metrics
        fold_results.append({
            'fold': fold,
            'map50':     val_metrics.metrics.map50,
            'map5095':   val_metrics.metrics.map,
            'precision': precision,
            'recall':    recall,
            'f2_score':  f2_score
        })
        print(f"Fold {fold} results: {fold_results[-1]}")

    # aggregate
    import pandas as pd
    df = pd.DataFrame(fold_results)
    print("\n=== Cross-Validation Summary ===")
    summary = df.agg(['mean', 'std'])
    print(summary)

if __name__ == "__main__":
    main()
