import os 
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
        description="このスクリプトの概要説明",
        epilog="追加の説明文（任意）"
    )

parser.add_argument(
    "--result",
    type=str,
    help="outputs/2025-05-07/19-51-56/42/yolo11l_train"
)

args = parser.parse_args()

csv_path = os.path.join(args.result, "results.csv")
df = pd.read_csv(csv_path)

p = df["metrics/precision(B)"]
r = df["metrics/recall(B)"]
df["F1"] = 2 * p * r / (p + r)
df["F2"] = 5 * p * r / (4*p + r)

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
plot_path = os.path.join(args.result, 'Plot_F2.png')
plt.savefig(plot_path)
plt.close()