# inference.py  ── 推論だけに最小化したスクリプト ──────────────────────────────
import os, glob, cv2, copy, numpy as np, pandas as pd, torch
from pathlib import Path
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
from utils import CenterDetectModule                         # ★ LightningModule 定義
from metric import score

# --------------------------------------------------------------------------
# 1.  設定値（最低限だけ残しています）
# --------------------------------------------------------------------------
CKPT_PATH    = "checkpoints/last-v18.ckpt"                   # ★ モデル ckpt
IMG_DIR      = "../parse_data_for_byu/test_data/images/"    # ★ JPG スライス dir
SAVE_SUBMIT  = "submission.csv"

IMG_SIZE     = 640
BATCH_SIZE   = 32
SLICE_STEP   = 3
TOPK_PER_IMG = 50
PIX_THRESH   = int(0.20 * IMG_SIZE)                          # (= box_size * iou_thr)
CONF_THRESH  = 0.5

# --------------------------------------------------------------------------
# 2.  モデル複製 & デバイス検出
# --------------------------------------------------------------------------
base = CenterDetectModule.load_from_checkpoint(CKPT_PATH)
base.model.half().eval()          # fp16 で推論
base.freeze()

num_gpu = torch.cuda.device_count()
device_ids = list(range(num_gpu)) or [None]                   # GPU 無ければ [None]

print(f"detected GPU : {len(device_ids)}")

center_model = []
for dev in device_ids:
    replica = copy.deepcopy(base)
    replica.to(f"cuda:{dev}" if dev is not None else "cpu")
    center_model.append(replica)

print(f"prepared replicas : {len(center_model)} ✔")

# 画像前処理
prep = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),               # 0-1 Float32
    T.ConvertImageDtype(torch.float16),
])

# --------------------------------------------------------------------------
# 3.  1 GPU(または CPU) あたり 1 つ走らせる関数
# --------------------------------------------------------------------------
class SliceDS(Dataset):
    """単一トモグラフ用の DataSet（スライス z と画像 Tensor を返す）"""
    def __init__(self, file_list, slice_idx):
        self.file_list = file_list
        self.slice_idx = slice_idx
    def __len__(self): return len(self.file_list)
    def __getitem__(self, i):
        img = cv2.imread(self.file_list[i], cv2.IMREAD_GRAYSCALE)
        img = prep(img)
        return self.slice_idx[i], img

@torch.inference_mode()
def run_replica(model: pl.LightningModule, tomo_ids, rank: int):
    out = []
    for ti, tomo_id in enumerate(tomo_ids, 1):
        # --- スライス一覧 --------------------------------------------------
        files      = sorted(glob.glob(f"{IMG_DIR}/{tomo_id}/*.jpg"))
        slice_idx  = np.arange(len(files))[::SLICE_STEP]
        files      = [files[i] for i in slice_idx]

        loader = DataLoader(
            SliceDS(files, slice_idx),
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            pin_memory  = True,
            num_workers = 2,
        )

        det = []                              # ← ここに各 slice の検出結果を集める
        for z_batch, img_batch in loader:
            img_batch = img_batch.to(model.device)          # fp16
            pred = model(img_batch).float()                 # ① fp32 へ

            # ② NaN / Inf をゼロに
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            # ③ sigmoid : obj は確率、xy は 0–1 範囲の相対座標になる
            obj = torch.sigmoid(pred[:, 2])                 # [B,H,W]
            xy  = torch.sigmoid(pred[:, :2])                # [B,2,H,W]

            B, H, W = obj.shape
            conf, idx = torch.topk(obj.flatten(1), k=min(TOPK_PER_IMG, H * W))
            xy = xy.permute(0, 2, 3, 1).reshape(B, -1, 2)   # [B,H*W,2]

            for b in range(B):
                # z_batch は DataLoader がまとめた tensor
                z = int(z_batch[b].item())     # ← ここを修正
                for c, id_ in zip(conf[b], idx[b]):
                    if c < CONF_THRESH:
                        break
                    y_cell, x_cell = divmod(id_.item(), W)
                    x_rel, y_rel   = xy[b, id_].cpu().tolist()

                    det.append({
                        "z":   int(z),                      # ← すでに int
                        "y":   y_rel * IMG_SIZE,
                        "x":   x_rel * IMG_SIZE,
                        "confidence": float(c),
                    })

        # ---------- 3-D NMS -----------------------------------------------
        det.sort(key=lambda d: d["confidence"], reverse=True)
        keep = []
        while det:
            best = det.pop(0)
            keep.append(best)
            det = [
                d for d in det
                if np.linalg.norm([
                    d["z"] - best["z"],
                    d["y"] - best["y"],
                    d["x"] - best["x"],
                ]) > PIX_THRESH
            ]

        # ---------- 1 tomo の結果を整形 -----------------------------------
        if keep:
            best = keep[0]
            # round してから int なら fp 誤差にも安全
            coord = {f"Motor axis {i}": int(round(best[k]))
                     for i, k in enumerate(("z", "y", "x"))}
            conf  = best["confidence"]
        else:
            coord = {f"Motor axis {i}": -1 for i in range(3)}
            conf  = CONF_THRESH

        out.append(dict(tomo_id=tomo_id, **coord, confidence=conf))

        print(f"\rGPU{rank}  {ti}/{len(tomo_ids)}  {tomo_id}  conf={conf:.2f}",
              end="", flush=True)

    print("")
    return out

# --------------------------------------------------------------------------
# 4.  Tomograph list をデバイス数で round-robin に割り当てて並列実行
# --------------------------------------------------------------------------
valid_id = sorted(os.listdir(IMG_DIR))    # トモグラフ ID の list

start = timer()
with ThreadPoolExecutor(max_workers=len(device_ids)) as ex:
    futures = [
        ex.submit(run_replica,
                  center_model[r],
                  valid_id[r::len(device_ids)],
                  r)
        for r in range(len(device_ids))
    ]
result = [f.result() for f in futures]

elapsed = timer() - start
print(f"\nTotal elapsed: {elapsed/60:.1f} min")

# --------------------------------------------------------------------------
# 5.  CSV 保存
# --------------------------------------------------------------------------
result_df = pd.concat([pd.DataFrame(r) for r in result], ignore_index=True)
result_df.to_csv("result.csv", index=False)

submit_df = result_df[["tomo_id",
                       "Motor axis 0",
                       "Motor axis 1",
                       "Motor axis 2"]]
submit_df.to_csv(SAVE_SUBMIT, index=False)
print("Saved:", SAVE_SUBMIT)

solution = pd.read_csv("../parse_data_for_byu/test_data/labels_subset.csv")

print(score(solution, submit_df, 1000, 2))