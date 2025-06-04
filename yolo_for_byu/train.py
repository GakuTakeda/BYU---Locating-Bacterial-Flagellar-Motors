# train.py
import hydra, torch, os, pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 既存クラス群
from utils import DataModule, CenterDetectModule

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg, resolve=True))  # 完全展開された config の確認用

    # ---------------- Data / Model ----------------
    dm     : DataModule           = hydra.utils.instantiate(cfg.datamodule)
    model  : CenterDetectModule   = hydra.utils.instantiate(cfg.model)

    # ---------------- Logger ----------------------
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name="center_detect")

    # ---------------- Callbacks -------------------
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        monitor="val/F2",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )

    early_cb = EarlyStopping(
        monitor="val/F2",
        mode="max",
        patience=10,
        verbose=True,
    )

    # ---------------- Trainer ---------------------
    trainer = pl.Trainer(
        **cfg.trainer,                    # ← YAML で書いた Trainer パラメータ
        logger=tb_logger,
        callbacks=[ckpt_cb, early_cb],
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
