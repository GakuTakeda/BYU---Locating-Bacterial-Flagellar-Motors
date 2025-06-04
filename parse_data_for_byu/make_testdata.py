import shutil, os, pandas as pd
from pathlib import Path

TOMO_LIST      = Path("dataset/tomo_list")   # 名前だけが並ぶテキスト
SOURCE_IMG_DIR = Path("../input/byu-locating-bacterial-flagellar-motors-2025/train")      # 既存 jpeg フォルダ群
DEST_IMG_DIR   = Path("test_data/images")                      # コピー先
LABEL_CSV      = Path("../input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")                      # 元のラベル CSV
OUT_CSV        = Path("test_data/labels_subset.csv")  

DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

with TOMO_LIST.open() as f:
    tomo_ids = [line.strip() for line in f if line.strip()]

for tid in tomo_ids:
    src = SOURCE_IMG_DIR / tid           # 例: images/tomo_172f08
    dst = DEST_IMG_DIR   / tid
    if not src.exists():
        print(f"[skip] {src} が見つかりません")
        continue
    if dst.exists():
        shutil.rmtree(dst)               # 既にあれば上書きのため削除
    shutil.copytree(src, dst)
    print(f"[copy] {src}  →  {dst}")

# ------------- ② CSV をフィルタして保存 ---------------------------------------------
df      = pd.read_csv(LABEL_CSV)
subset  = df[df["tomo_id"].isin(tomo_ids)].reset_index(drop=True)
subset.to_csv(OUT_CSV, index=False)
print(f"\n{len(subset)} 行を書き出しました → {OUT_CSV}")