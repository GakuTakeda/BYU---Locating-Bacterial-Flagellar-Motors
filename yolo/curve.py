from utils import plot_dfl_loss_curve
import os 
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
        description="このスクリプトの概要説明",
        epilog="追加の説明文（任意）"
    )

parser.add_argument(
        "--epoch",
        default=None,
        type=int     
    )

parser.add_argument(
    "--result",
    type=str
)

args = parser.parse_args()

plot_dfl_loss_curve(args.result)
csv_path = os.path.join(args.result, "results.csv")
df = pd.read_csv(csv_path)

p = df["metrics/precision(B)"]
r = df["metrics/recall(B)"]
df["F1"] = 2 * p * r / (p + r)
df["Fb"] = 2 * p * r / (4*p + r)

# mAP@0.5 が最大の行のインデックスを取得
best50_idx = df["F1"].idxmax()
bestF1_epoch = int(df.loc[best50_idx, "epoch"])
bestF1_value = df.loc[best50_idx, "F1"]

best50_idx = df["Fb"].idxmax()
bestFb_epoch = int(df.loc[best50_idx, "epoch"])
bestFb_value = df.loc[best50_idx, "Fb"]

# mAP@0.5 が最大の行のインデックスを取得
best50_idx = df["metrics/mAP50(B)"].idxmax()
best50_epoch = int(df.loc[best50_idx, "epoch"])
best50_value = df.loc[best50_idx, "metrics/mAP50(B)"]

# mAP@0.5:0.95 が最大の行のインデックスを取得
best50_95_idx = df["metrics/mAP50-95(B)"].idxmax()
best50_95_epoch = int(df.loc[best50_95_idx, "epoch"])
best50_95_value = df.loc[best50_95_idx, "metrics/mAP50-95(B)"]

# mAP@0.5:0.95 が最大の行のインデックスを取得
best50_95_idx = df["val/dfl_loss"].idxmin()
bestdfl_epoch = int(df.loc[best50_95_idx, "epoch"])
bestdfl_value = df.loc[best50_95_idx, "val/dfl_loss"]



print(f"▶ Best F1   = {bestF1_value:.4f}  エポック {bestF1_epoch}")
print(f"▶ Best Fb   = {bestFb_value:.4f}  エポック {bestFb_epoch}")
print(f"▶ Best mAP@0.5   = {best50_value:.4f}  エポック {best50_epoch}")
print(f"▶ Best mAP@0.5:0.95 = {best50_95_value:.4f}  エポック {best50_95_epoch}")
print(f"▶ Best dfl   = {bestdfl_value:.4f}  エポック {bestdfl_epoch}")

if args.epoch is not None:
    print(f"▶ エポック {args.epoch} のvalue {df.loc[df["epoch"] == args.epoch]}")