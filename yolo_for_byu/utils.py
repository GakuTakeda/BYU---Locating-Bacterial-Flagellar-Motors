import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import albumentations as A
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric
import torchvision.transforms.v2 as T
from pathlib import Path
import random
import cv2

# ---------------------------
# Model Components
# ---------------------------
class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1, bias=False, act=True):
        layers = [nn.Conv2d(in_ch, out_ch, k, s, k//2, groups=g, bias=bias), nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(5,9,13)):
        super().__init__()
        mid = in_ch//2
        self.conv1 = ConvNormAct(in_ch, mid, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(k,1,k//2) for k in kernels])
        self.conv2 = ConvNormAct(mid*(len(kernels)+1), out_ch, 1)
    def forward(self,x):
        y = self.conv1(x)
        ys = [y] + [p(y) for p in self.pools]
        return self.conv2(torch.cat(ys,1))

class Backbone(nn.Module):
    def __init__(self, model_name="fastvit_sa12.apple_dist_in1k", in_chans=1):
        super().__init__()
        self.base = timm.create_model(model_name, pretrained=True, in_chans=in_chans,
                                      features_only=True, out_indices=(1,2,3))
        chs = self.base.feature_info.channels()
        self.spp = SpatialPyramidPooling(chs[-1], chs[-1])
    def forward(self,x):
        p0,p1,p2 = self.base(x)
        p2 = self.spp(p2)
        return p0,p1,p2
    def reparameterize(self):
        for m in self.modules():
            if hasattr(m,'reparameterize'):
                m.reparameterize()

class Neck(nn.Module):
    def __init__(self, channels=(128, 256, 512)):
        super().__init__()
        C0, C1, C2 = channels
        self.merge2 = timm.models.fastvit.MobileOneBlock(
            C2 + C1, C1, kernel_size=3, use_se=True, num_conv_branches=4
        )
        self.merge1 = timm.models.fastvit.MobileOneBlock(
            C1 + C0, C0, kernel_size=3, use_se=True, num_conv_branches=4
        )
        self.merge3 = timm.models.fastvit.MobileOneBlock(
            C0 + C1, C1, kernel_size=3, use_se=True, num_conv_branches=4
        )
        self.merge4 = timm.models.fastvit.MobileOneBlock(
            C1 + C2, C2, kernel_size=3, use_se=True, num_conv_branches=4
        )
    def forward(self,p0,p1,p2):
        up21 = F.interpolate(p2,size=p1.shape[-2:],mode='bilinear',align_corners=False)
        h1 = self.merge2(torch.cat([up21,p1],1))
        up10 = F.interpolate(h1,size=p0.shape[-2:],mode='bilinear',align_corners=False)
        o0 = self.merge1(torch.cat([up10,p0],1))
        o0_h1 = F.interpolate(o0,size=h1.shape[-2:],mode='bilinear',align_corners=False)
        o1 = self.merge3(torch.cat([o0_h1,h1],1))
        o1_p2 = F.interpolate(o1,size=p2.shape[-2:],mode='bilinear',align_corners=False)
        o2 = self.merge4(torch.cat([o1_p2,p2],1))
        return [o0,o1,o2]
    def reparameterize(self):
        for m in self.modules():
            if hasattr(m,'reparameterize'):
                m.reparameterize()

class Head(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        mid = max(in_channels, num_classes)

        self.point = nn.Sequential(
            timm.models.fastvit.MobileOneBlock(
                in_channels,
                16,
                kernel_size=3,
                stride=1,
                use_se=True,
                num_conv_branches=4
            ),
            nn.Conv2d(16, 2, kernel_size=1)
        )

        self.cls = nn.Sequential(
            timm.models.fastvit.MobileOneBlock(
                in_channels,
                mid,
                kernel_size=3,
                stride=1,
                use_se=True,
                num_conv_branches=4
            ),
            nn.Conv2d(mid, num_classes, kernel_size=1)
        )
    def forward(self, x):
        # ======= ① logits を計算する =======
        pt_logits = self.point(x)        # [B, 2, H, W]
        cl_logits = self.cls(x)          # [B, C, H, W]
        obj_logits = cl_logits.max(dim=1, keepdim=True)[0]   # [B,1,H,W]

        # ======= ② 訓練中は logits をそのまま返す =======
        if self.training:
            return torch.cat([pt_logits, obj_logits, cl_logits], 1)

        # ======= ③ 推論時だけ sigmoid を掛けて返す =======
        pt = torch.sigmoid(pt_logits)
        cl = torch.sigmoid(cl_logits)
        obj = torch.sigmoid(obj_logits)
        return torch.cat([pt, obj, cl], 1)
    def reparameterize(self):
        for m in self.modules():
            if hasattr(m,'reparameterize'):
                m.reparameterize()

class Net(nn.Module):
    def __init__(self, chs=(128,256,512), num_cls=1):
        super().__init__()
        self.back = Backbone(in_chans=1)
        self.neck = Neck(chs)
        self.heads = nn.ModuleList([Head(c,num_cls) for c in chs])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels==1:  # obj conv
                nn.init.constant_(m.bias, -4.6)      # σ(−4.6)≈0.01

    def forward(self,x):
        p0,p1,p2 = self.back(x)
        outs = self.neck(p0,p1,p2)
        preds = [h(o) for h,o in zip(self.heads,outs)]
        return preds if self.training else preds[torch.argmax(torch.tensor([p[:,2].max() for p in preds]))]
    def reparameterize(self):
        self.back.reparameterize(); self.neck.reparameterize(); [h.reparameterize() for h in self.heads]

# ---------------------------
# Losses
# ---------------------------
class PointLoss(nn.Module):
    """正例セルだけ Smooth-L1 (δ=1)"""
    def __init__(self, delta: float = 10.):
        super().__init__()
        self.delta = delta

    def forward(self, pred_xy, gt_xy, mask):
        # mask : (N,) 1 正例, 0 負例
        if mask.sum() == 0:
            return pred_xy.new_tensor(0.)
        diff = torch.abs(pred_xy - gt_xy)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff.pow(2) / self.delta,
            diff - 0.5 * self.delta,
        )
        return loss[mask.bool()].mean()
    
class ObjLoss(nn.Module):
    def __init__(self, pos_w=1.0, neg_w=0.2):
        super().__init__()
        self.pos_w, self.neg_w = pos_w, neg_w

    def forward(self, pred, target):
        pos_mask = target == 1
        neg_mask = target == 0
        loss = 0
        if pos_mask.any():
            loss += self.pos_w * F.binary_cross_entropy_with_logits(
                pred[pos_mask], target[pos_mask], reduction='mean')
        if neg_mask.any():
            loss += self.neg_w * F.binary_cross_entropy_with_logits(
                pred[neg_mask], target[neg_mask], reduction='mean')
        return loss

    
class DetectionLoss(nn.Module):
    def __init__(self,
                 λ_point=1.0,
                 λ_obj=1.0,
                 λ_cls=1.0):
        super().__init__()
        self.pt_loss = PointLoss()
        self.obj_loss = ObjLoss(pos_w=2.0, neg_w=0.05)
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.λp, self.λo, self.λc = λ_point, λ_obj, λ_cls

    def forward(self, preds, targets):
        device = preds[0].device               # 追加
        Lp = torch.tensor(0., device=device)   # ← tensor で初期化
        Lo = torch.tensor(0., device=device)
        Lc = torch.tensor(0., device=device)   # cls を使わないなら 0 のまま

        for p, t in zip(preds, targets):
            """
            p : [B, 3+cls, H, W]  (logits)
            t : [B, 3+cls, H, W]  (教師)
                   └0,1: xy (rel) / 2: obj
            """
            B, C, H, W = p.shape

            # --- Point loss --------------------------------------
            pred_xy = p[:, :2].permute(0, 2, 3, 1).reshape(-1, 2)
            gt_xy   = t[:, :2].permute(0, 2, 3, 1).reshape(-1, 2)
            obj_tgt = t[:, 2]                       # [B,H,W]
            mask    = obj_tgt.reshape(-1) == 1      # 正例だけ
            Lp     += self.pt_loss(pred_xy, gt_xy, mask)

            # --- Objectness loss ---------------------------------
            pred_obj = p[:, 2].reshape(-1)
            gt_obj   = obj_tgt.reshape(-1)
            Lo      += self.obj_loss(pred_obj, gt_obj)

        total = self.λp * Lp + self.λo * Lo + self.λc * Lc
        return total, Lp, Lo, Lc 

# ---------------------------
# Dataset
# ---------------------------
class CenterDataset(Dataset):
    """中心 (x,y) だけを学習するデータセット : Mosaic / MixUp 対応版"""
    def __init__(
        self,
        img_dir:str,
        label_dir:str,
        img_size:int = 640,
        transforms: A.BasicTransform | None = None,
        # augmentation hyper-params ↓↓↓
        mosaic: float = 1.0,
        close_mosaic: float = 10.0,   # (%) Mosaic 中心のランダムずらし
        mixup : float = 0.4,
        degrees: float = 45,
        shear: float  = 0.,
        perspective: float = 0.,
        translate: float = 0.1,
        scale: float = 0.25,
        flipud: float = .5,
        fliplr: float = .5,
        mode: str = "train",          # "train" or "val"
    ):
        super().__init__()
        self.imgs   = list(Path(img_dir ).glob('*.jpg'))
        self.lbls   = Path(label_dir)
        self.size   = img_size
        self.mode   = mode

        # torchvision → Tensor
        self.to_t = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

        # albumentations (単一画像にかける transform)
        self.tf_single = transforms or A.Compose([
            A.ShiftScaleRotate(translate, scale, degrees, shear, p=1),
            A.Perspective(scale=(0, perspective), p=1) if perspective else A.NoOp(),
            A.VerticalFlip(p=flipud),
            A.HorizontalFlip(p=fliplr),
        ])

        # YOLO-style augm. ハイパー
        self.mosaic_p  = mosaic
        self.close_m   = close_mosaic
        self.mixup_p   = mixup

    # ----------- util -----------
    def _read_one(self, idx:int):
        """画像１枚とアノテーション１行 (無ければ None) を返す"""
        ip = self.imgs[idx]
        img = np.array(Image.open(ip).convert("L"))          # (H,W)

        # label
        lp = self.lbls / ip.name.replace(".jpg", ".txt")
        target = None
        if lp.exists():
            lines = [l for l in lp.read_text().splitlines()
                     if l and l[0].isdigit() or l[0] == '-']   # ヘッダー除去
            if lines:
                vals = list(map(float, " ".join(lines).split()))
                if len(vals) >= 4 and vals[0] >= 0 and vals[1] >= 0:
                    target = tuple(vals[:2])   # (x_rel, y_rel)

        return img, target

    # ----------- core -----------
    def __getitem__(self, idx):
        if self.mode == "train":                # ───────── train only
            r = random.random()
            if r < self.mosaic_p:               # ----- Mosaic -----
                return self._mosaic(idx)
            if r < self.mosaic_p + self.mixup_p:# ----- MixUp  -----
                return self._mixup(idx)

        # ----- 単独読み込み -----
        img, tgt = self._read_one(idx)
        img = self.tf_single(image=img)["image"]
        img = Image.fromarray(img)
        img_t = self.to_t(img)                 # [1,H,W]

        H,W   = img_t.shape[-2:]
        if tgt:
            x_px, y_px = tgt[0]*W, tgt[1]*H
            target = torch.tensor([x_px, y_px, 1.], dtype=torch.float32)
        else:
            target = torch.tensor([-1., -1., 0.], dtype=torch.float32)
        return img_t, target

    # --------- Mosaic ----------
    def _mosaic(self, idx:int):
        indices = [idx] + random.sample(range(len(self)), 3)
        s   = self.size
        xc  = int(random.uniform(0.5-self.close_m/100,
                                 0.5+self.close_m/100)*s)
        yc  = int(random.uniform(0.5-self.close_m/100,
                                 0.5+self.close_m/100)*s)
        canvas = np.zeros((s*2, s*2), dtype=np.uint8)     # (2s,2s)

        labels = []
        for i, j in enumerate(indices):
            img, tgt = self._read_one(j)
            h, w = img.shape
            # 4象限に貼り付け
            if i == 0:  x1,y1,x2,y2 = max(xc-w,0), max(yc-h,0), xc, yc
            elif i==1:  x1,y1,x2,y2 = xc, max(yc-h,0), min(xc+w,2*s), yc
            elif i==2:  x1,y1,x2,y2 = max(xc-w,0), yc, xc, min(yc+h,2*s)
            else:       x1,y1,x2,y2 = xc, yc, min(xc+w,2*s), min(yc+h,2*s)

            # resize → paste
            img_rs = cv2.resize(img, (x2-x1, y2-y1))
            canvas[y1:y2, x1:x2] = img_rs

            # label 座標をキャンバス座標へ
            if tgt:
                x_rel, y_rel = tgt
                new_x = x1 + x_rel*(x2-x1)
                new_y = y1 + y_rel*(y2-y1)
                labels.append((new_x/(2*s), new_y/(2*s)))

        # 中心 s/2,s/2 で crop
        crop = canvas[s//2:s//2+s, s//2:s//2+s]
        img = self.tf_single(image=crop)["image"]
        img = Image.fromarray(img)
        img_t = self.to_t(img)

        # 1枚に対しラベルが複数できるので「代表値」として平均
        if labels:
            xs,ys = zip(*labels)
            x_px = sum(xs)/len(xs) * self.size
            y_px = sum(ys)/len(ys) * self.size
            target = torch.tensor([x_px, y_px, 1.], dtype=torch.float32)
        else:
            target = torch.tensor([-1., -1., 0.], dtype=torch.float32)
        return img_t, target

    # --------- MixUp ----------
    def _mixup(self, idx:int):
        img1,tgt1 = self._read_one(idx)
        img2,tgt2 = self._read_one(random.randrange(len(self)))
        lam = np.random.beta(32.,32.)

        img = (lam*img1 + (1-lam)*img2).astype(np.uint8)
        img = self.tf_single(image=img)["image"]
        img = Image.fromarray(img)
        img_t = self.to_t(img)

        # 座標は線形補間（どちらか片方 None ならそちらを採用）
        if tgt1 and tgt2:
            x_rel = lam*tgt1[0] + (1-lam)*tgt2[0]
            y_rel = lam*tgt1[1] + (1-lam)*tgt2[1]
            target = torch.tensor([x_rel*self.size, y_rel*self.size, 1.], dtype=torch.float32)
        else:
            if tgt1:
                x_rel,y_rel = tgt1; obj=1.
            elif tgt2:
                x_rel,y_rel = tgt2; obj=1.
            else:
                x_rel=y_rel=-1.; obj=0.
            target = torch.tensor([x_rel*self.size, y_rel*self.size, obj], dtype=torch.float32)
        return img_t, target
    
    def __len__(self):      
        return len(self.imgs)

# ---------------------------
# DataModule
# ---------------------------
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_img, train_lbl,
        val_img, val_lbl,
        img_size=640,
        batch_size=8,
        num_workers=4,
        **aug_kwargs          # ← 上記 Mosaic/MixUp パラメータをそのまま渡せる
    ):
        super().__init__()
        self.train_ds = CenterDataset(train_img, train_lbl, img_size,
                                      mode="train", **aug_kwargs)
        self.val_ds = CenterDataset(
            val_img, val_lbl, img_size,

            transforms=A.Compose([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ]),
            # Mosaic / MixUp を 0 にして無効化
            mosaic=0.0, mixup=0.0,
            mode="val",
        )
        self.bs, self.nw = batch_size, num_workers

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.bs, True , num_workers=self.nw)
    def val_dataloader(self):
        return DataLoader(self.val_ds  , self.bs, False, num_workers=self.nw)

# ---------------------------
# LightningModule
# ---------------------------
class CenterDetectModule(pl.LightningModule):
    def __init__(self,lr=1e-3,wdist=1.,wobj=1.,img_size=640, λ_point=1.0, λ_obj=1.0, λ_cls=0.0):
        super().__init__() 
        self.save_hyperparameters()
        self.model=Net() 
        self.loss = DetectionLoss(λ_point, λ_obj, λ_cls)
        self.trl=MeanMetric() 
        self.vll=MeanMetric()

    def forward(self,x): 
        return self.model(x)
    
    def _step(self, batch):
        imgs, targets = batch
        preds  = self(imgs)
        if not isinstance(preds,(list,tuple)):
            preds=[preds]
        t_list = [self._make_target(p, targets) for p in preds]

        total, dl, ol, cl = self.loss(preds, t_list)   # ← 4 変数で受ける
        return total, dl, ol, cl                       # ← 必要に応じて返す

    
    def training_step(self,b,bi): 
        total, dl, ol, cl = self._step(b)
        self.trl.update(total+ol)
        bs = b[0].size(0)
        self.log('train/loss',total, batch_size=bs) 
        self.log('train/d',dl, batch_size=bs) 
        self.log('train/o',ol, batch_size=bs) 
        return total
    def on_validation_epoch_start(self):
        self._pred_buffer = []     # [(conf, x, y, img_idx), ...]
        self._gt_buffer   = []     # [(x, y, img_idx), ...]

    # ────────── validation_step で予測を保存 ──────────
    def validation_step(self, batch, batch_idx):
        total, dl, ol, cl = self._step(batch)
        bs = batch[0].size(0)
        self.vll.update(total)
        self.log("val/loss", total, batch_size=bs)
        self.log('val/d',dl, prog_bar=True, batch_size=bs) 
        self.log('val/o',ol, prog_bar=True, batch_size=bs) 

        imgs, targets = batch
        preds = self(imgs)
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        # ここでは一番高解像度 head だけ使う例
        obj_map = preds[0][:, 2]         # [B, H, W] objectness
        xy_map  = preds[0][:, :2]        # [B, 2, H, W]

        B, H, W = obj_map.shape
        for b in range(B):
            # --- GT 座標をバッファに ---
            x_gt, y_gt, obj = targets[b]
            if obj > 0:
                self._gt_buffer.append((x_gt.item(), y_gt.item(), b + batch_idx*self.trainer.val_dataloaders.batch_size))

            # --- 予測をバッファに (top-k だけ残す手抜き) ---
            conf = obj_map[b].flatten()
            xy   = xy_map[b].permute(1,2,0).reshape(-1,2)  # [H*W,2]
            p = 0.05
            flat_conf = obj_map[b].flatten()
            k = max(1, int(flat_conf.numel() * p))
            topk = torch.topk(flat_conf, k=k)                # 例: 各画像50点
            for score, idx in zip(topk.values, topk.indices):
                y_cell, x_cell = divmod(idx.item(), W)
                x_rel = xy[idx, 0].item()
                y_rel = xy[idx, 1].item()
                x_px  = x_rel * self.hparams.img_size
                y_px  = y_rel * self.hparams.img_size
                self._pred_buffer.append((score.item(), x_px, y_px, b + batch_idx*self.trainer.val_dataloaders.batch_size))

    # ────────── epoch 終了時に F2 を計算 ──────────
    def on_validation_epoch_end(self):
        if not self._gt_buffer:          # 全バッチ skip の場合
            return

        preds = torch.tensor(self._pred_buffer)  # [N,4] (conf,x,y,img_idx)
        gts   = torch.tensor(self._gt_buffer)    # [M,3] (x,y,img_idx)

        best_F2 = 0.0
        thr_list = torch.linspace(0.1, 0.9, 9)   # 0.1〜0.9 を試す
        for t in thr_list:
            keep = preds[:,0] >= t
            if keep.sum()==0:
                continue
            p_sel = preds[keep]

            # ─ マッチング (簡易版：同一 img_idx で距離 < 5px なら TP) ─
            TP, FP, FN = 0, 0, 0
            used_gt = torch.zeros(len(gts), dtype=torch.bool)
            for conf,xp,yp,idx in p_sel:
                d = torch.sqrt((gts[:,0]-xp)**2 + (gts[:,1]-yp)**2)
                same_img = gts[:,2]==idx
                cand = torch.nonzero((d < 5000) & same_img & (~used_gt))
                if len(cand):
                    TP += 1
                    used_gt[cand[0]] = True
                else:
                    FP += 1
            FN = (~used_gt).sum().item()

            precision = TP / (TP+FP+1e-9)
            recall    = TP / (TP+FN+1e-9)
            beta2 = 2**2
            if precision+recall==0:
                F2 = 0.0
            else:
                F2 = (1+beta2)*precision*recall/(beta2*precision+recall)
            best_F2 = max(best_F2, F2)

        # --- ログ & モニタリング用 ---
        self.log("val/F2", best_F2, prog_bar=True, sync_dist=True)
        #self.log("val/TP", TP, prog_bar=True, sync_dist=True)

    def configure_optimizers(self): 
        return torch.optim.AdamW(self.parameters(),lr=self.hparams.lr)
    
    def _make_target(self,preds,targets):
        B,C,H,W=preds.shape 
        t=torch.zeros(B,C,H,W,device=preds.device)
        for i,(x_px,y_px,obj) in enumerate(targets):
            if obj<=0: 
                continue
            rel_x=x_px/self.hparams.img_size 
            rel_y=y_px/self.hparams.img_size
            cx=int(rel_x*W) 
            cy=int(rel_y*H)
            cx=max(0,min(cx,W-1)) 
            cy=max(0,min(cy,H-1))
            t[i,0,cy,cx]=rel_x; t[i,1,cy,cx]=rel_y; t[i,2,cy,cx]=1
        return t
    



