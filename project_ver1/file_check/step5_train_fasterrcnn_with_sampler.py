import os
import json
import math
import time
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, WeightedRandomSampler

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# =========================
# CONFIG
# =========================
IMG_ROOT = "/home/ohs3201/work/step2_unified_coco/images"
TRAIN_JSON = "/home/ohs3201/work/step4_runs_remap/train_remap.json"
VAL_JSON   = "/home/ohs3201/work/step4_runs_remap/val_remap.json"

WEIGHT_JSON = "/home/ohs3201/work/step3_stats/train_image_weights.json"

SAVE_DIR = "./checkpoints_step5_sampler"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LR = 1e-4


# =========================
# Dataset
# =========================
class CocoDetectionSafe(data.Dataset):
    def __init__(self, img_root, ann_file):
        self.img_root = img_root
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        path = os.path.join(self.img_root, img_info["file_name"])

        if not os.path.exists(path):
            return None

        image = torchvision.io.read_image(path).float() / 255.0

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for a in anns:
            if "bbox" not in a or len(a["bbox"]) != 4:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        return image, target


# =========================
# Collate (빈 batch 차단)
# =========================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


# =========================
# mAP 평가
# =========================
@torch.no_grad()
def evaluate_map(model, dataset, coco_gt):
    model.eval()
    results = []

    for img, target in tqdm(dataset, desc="[VAL] inference"):
        img = img.to(DEVICE)
        outputs = model([img])[0]

        image_id = int(target["image_id"].item())

        for box, score, label in zip(
            outputs["boxes"], outputs["scores"], outputs["labels"]
        ):
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    if len(results) == 0:
        return 0.0

    coco_gt.dataset.setdefault("info", {})
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]


# =========================
# Train
# =========================
def train():
    print("[INFO] Device:", DEVICE)

    train_ds = CocoDetectionSafe(IMG_ROOT, TRAIN_JSON)
    val_ds   = CocoDetectionSafe(IMG_ROOT, VAL_JSON)

    print(f"[Dataset] Train images: {len(train_ds)}")
    print(f"[Dataset] Val images  : {len(val_ds)}")

    # ---------- Sampler ----------
    weights_json = json.load(open(WEIGHT_JSON))
    weights = []

    for img_id in train_ds.ids:
        weights.append(weights_json.get(str(img_id), 1.0))

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # ---------- Model ----------
    coco_train = COCO(TRAIN_JSON)
    K = len(coco_train.getCatIds())
    print("[INFO] K (foreground):", K)

    model = fasterrcnn_resnet50_fpn(num_classes=K + 1)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_map = -1
    no_improve = 0

    # ---------- Epoch Loop ----------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")
        for batch in pbar:
            if batch is None:
                continue

            images, targets = batch
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"[Epoch {epoch}] mean loss: {mean_loss:.4f}")

        # ---------- Validation ----------
        coco_val = COCO(VAL_JSON)
        map50_95 = evaluate_map(model, val_ds, coco_val)

        print(f"[Epoch {epoch}] mAP@0.5:0.95 = {map50_95:.6f}")

        # ---------- Early Stop ----------
        if map50_95 > best_map:
            best_map = map50_95
            no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "best_map.pth")
            )
            print(f"✅ [BEST] epoch={epoch}, mAP={best_map:.6f}")
        else:
            no_improve += 1
            print(f"[EarlyStop] no improve: {no_improve}/{EARLY_STOP_PATIENCE}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping triggered.")
            break


if __name__ == "__main__":
    train()