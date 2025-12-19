import os
import json
import time
import math
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# =========================
# 0) 경로 설정
# =========================
IMG_ROOT  = "/home/ohs3201/work/step2_unified_coco/images"
TRAIN_JSON = "/home/ohs3201/work/step4_runs_remap/train_remap.json"
VAL_JSON   = "/home/ohs3201/work/step4_runs_remap/val_remap.json"

OUT_DIR = "/home/ohs3201/6team_beginner_project/project_ver1/checkpoints_earlystop"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
MAX_EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 4

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# EarlyStop
EARLYSTOP_PATIENCE = 10
EARLYSTOP_MIN_DELTA = 1e-4

# Eval
SCORE_THRESH = 0.001
MAX_DETS_PER_IMAGE = 300

# ROI-safe 필터
MIN_BOX_SIZE = 1.0       # w,h 최소(px)
MIN_BOX_AREA = 1.0       # area 최소(px^2)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1) Dataset (ROI-safe)
# =========================
class CocoDetSafe(CocoDetection):
    """
    - COCO JSON의 bbox(x,y,w,h)를 받아 FasterRCNN target(boxes[x1,y1,x2,y2], labels)로 변환
    - 이미지 크기 기준 clamp + invalid 제거
    - 최종적으로 boxes가 0개면 None 반환 -> collate에서 drop
    """
    def __init__(self, img_root: str, ann_file: str, K: int):
        super().__init__(img_root, ann_file)
        self.K = int(K)  # foreground class count (1..K)

    def __getitem__(self, idx: int):
        img, anns = super().__getitem__(idx)

        # coco image id
        img_id = self.ids[idx]
        w, h = img.size

        boxes = []
        labels = []

        for a in anns:
            # category_id는 remap으로 1..K 라고 가정
            cid = int(a.get("category_id", -1))
            bbox = a.get("bbox", None)

            if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x, y, bw, bh = bbox
            # 숫자 형태 확인
            try:
                x = float(x); y = float(y); bw = float(bw); bh = float(bh)
            except Exception:
                continue

            if not np.isfinite([x, y, bw, bh]).all():
                continue
            if bw <= 0 or bh <= 0:
                continue

            # xywh -> xyxy
            x1 = x
            y1 = y
            x2 = x + bw
            y2 = y + bh

            # clamp (여기서 GT가 이미지 밖으로 나가는 문제를 강제로 막음)
            x1 = max(0.0, min(x1, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            x2 = max(0.0, min(x2, w - 1.0))
            y2 = max(0.0, min(y2, h - 1.0))

            bw2 = x2 - x1
            bh2 = y2 - y1
            if bw2 < MIN_BOX_SIZE or bh2 < MIN_BOX_SIZE:
                continue
            if (bw2 * bh2) < MIN_BOX_AREA:
                continue

            # label range 강제 (CUDA assert 방지의 핵심)
            if cid < 1 or cid > self.K:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cid)

        # boxes가 0개면 학습/평가에서 아예 제외
        if len(boxes) == 0:
            return None

        img_t = F.to_tensor(img)  # float32 0~1

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }

        # 마지막 안전장치 (NaN/inf/범위)
        if not torch.isfinite(target["boxes"]).all():
            return None
        if target["labels"].min() < 1 or target["labels"].max() > self.K:
            return None

        return img_t, target


def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images, targets = zip(*batch)
    return list(images), list(targets)


# =========================
# 2) Model
# =========================
def build_model(num_classes: int):
    """
    torchvision 기본 fasterrcnn_resnet50_fpn 사용
    num_classes = K+1 (background 포함)
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# =========================
# 3) COCOeval helper
# =========================
def ensure_coco_meta(coco_gt: COCO):
    """
    pycocotools COCO.loadRes() 가 self.dataset['info'] 등을 deepcopy 하려다 KeyError 나는 케이스 방지
    """
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []


@torch.no_grad()
def evaluate_map(model, val_loader, coco_gt: COCO, device, K: int):
    model.eval()

    ensure_coco_meta(coco_gt)

    results = []
    pbar = tqdm(val_loader, desc="[VAL] inference", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        images, targets = batch
        images = [img.to(device) for img in images]

        outputs = model(images)  # list of dict

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            if boxes.numel() == 0:
                continue

            # score thresh
            keep = scores >= SCORE_THRESH
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # label range 방어 (혹시라도 이상치 나오면 제거)
            if labels.numel() > 0:
                keep2 = (labels >= 1) & (labels <= K)
                boxes = boxes[keep2]
                scores = scores[keep2]
                labels = labels[keep2]

            # top-k
            if boxes.shape[0] > MAX_DETS_PER_IMAGE:
                idx = torch.argsort(scores, descending=True)[:MAX_DETS_PER_IMAGE]
                boxes = boxes[idx]
                scores = scores[idx]
                labels = labels[idx]

            # xyxy -> xywh
            for b, s, c in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                results.append({
                    "image_id": img_id,
                    "category_id": int(c.item()),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(s.item())
                })

    # 결과가 0개면 mAP 계산이 의미 없으니 0으로 리턴
    if len(results) == 0:
        return {
            "mAP@0.5:0.95": 0.0,
            "mAP@0.5": 0.0,
            "mAP@0.75": 0.0,
            "AR@1": 0.0,
            "AR@10": 0.0,
            "AR@100": 0.0,
        }

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # 12개
    # 0: AP@[.5:.95], 1: AP@.5, 2: AP@.75, 6: AR@1, 7: AR@10, 8: AR@100
    return {
        "mAP@0.5:0.95": float(stats[0]),
        "mAP@0.5": float(stats[1]),
        "mAP@0.75": float(stats[2]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
    }


# =========================
# 4) Train loop + NaN guard + EarlyStop
# =========================
def train():
    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] IMG_ROOT: {IMG_ROOT}")
    print(f"[INFO] TRAIN_JSON: {TRAIN_JSON}")
    print(f"[INFO] VAL_JSON: {VAL_JSON}")

    # K 파악 (foreground)
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        tr = json.load(f)
    if "categories" not in tr:
        raise RuntimeError("train json에 categories가 없습니다.")
    K = len(tr["categories"])
    num_classes = K + 1
    print(f"[INFO] K (foreground): {K}")
    print(f"[INFO] num_classes: {num_classes}")

    # COCO GT for eval
    coco_val = COCO(VAL_JSON)
    ensure_coco_meta(coco_val)

    # datasets
    train_ds = CocoDetSafe(IMG_ROOT, TRAIN_JSON, K=K)
    val_ds   = CocoDetSafe(IMG_ROOT, VAL_JSON, K=K)

    # valid count (대략적인 확인)
    def count_valid(ds, name):
        n = 0
        for i in range(len(ds)):
            x = ds[i]
            if x is not None:
                n += 1
        print(f"[Dataset] valid images: {n} ({name})")
        return n

    # 빠르게 진행하려면 샘플링 체크를 꺼도 됨 (지금은 안정성 위해 유지)
    count_valid(train_ds, "train")
    count_valid(val_ds, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_drop_none,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_drop_none,
        drop_last=False,
    )

    model = build_model(num_classes=num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # AMP (최신 API)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_map = -1.0
    best_epoch = -1
    no_improve = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")
        for batch in pbar:
            if batch is None:
                continue
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            try:
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values())
                else:
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                # NaN/Inf 방지
                if not torch.isfinite(loss):
                    pbar.set_postfix({"loss": "nan/inf (skip)"})
                    continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            except RuntimeError as e:
                # CUDA assert 등 치명 에러 잡아서 어떤 배치였는지 알 수 있게만 남김
                print("\n[ERROR] RuntimeError during train step:", str(e))
                raise

        lr_scheduler.step()

        mean_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float("inf")
        print(f"[Epoch {epoch}] mean loss: {mean_loss:.4f}")

        # ===== eval mAP =====
        metrics = evaluate_map(model, val_loader, coco_val, device, K=K)
        cur_map = metrics["mAP@0.5:0.95"]
        print(f"[Epoch {epoch}] mAP@0.5:0.95 = {cur_map:.6f}")

        # 기록
        row = {"epoch": epoch, "train_loss": mean_loss, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "train_log.csv"), index=False)

        # checkpoint 저장 (매 epoch)
        ckpt_path = os.path.join(OUT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "K": K,
            "num_classes": num_classes,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "train_loss": mean_loss,
        }, ckpt_path)

        # best 저장 + early stop
        if cur_map > (best_map + EARLYSTOP_MIN_DELTA):
            best_map = cur_map
            best_epoch = epoch
            no_improve = 0
            best_path = os.path.join(OUT_DIR, "best_map.pth")
            torch.save({
                "epoch": epoch,
                "K": K,
                "num_classes": num_classes,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "train_loss": mean_loss,
            }, best_path)
            print(f"✅ [BEST] epoch={epoch} mAP={best_map:.6f} saved -> {best_path}")
        else:
            no_improve += 1
            print(f"[EarlyStop] no improve: {no_improve}/{EARLYSTOP_PATIENCE} (best={best_map:.6f} @ epoch {best_epoch})")
            if no_improve >= EARLYSTOP_PATIENCE:
                print("🛑 Early stopping triggered.")
                break

    print("\n[TRAIN DONE]")
    print(f"Best mAP@0.5:0.95 = {best_map:.6f} at epoch {best_epoch}")
    print(f"Logs/ckpt saved in: {OUT_DIR}")


if __name__ == "__main__":
    train()