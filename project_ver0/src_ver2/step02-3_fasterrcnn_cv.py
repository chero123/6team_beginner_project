import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.ops import box_iou
from torchvision.transforms import functional as F

from PIL import Image
from tqdm import tqdm

# 기본 경로 설정
BASE = "/home/ohs3201/6team_beginner_project"
TRAIN_IMG_DIR = os.path.join(BASE, "data/train_images")
TRAIN_ANN_DIR = os.path.join(BASE, "data/train_annotations")
FOLDS_PATH = os.path.join(BASE, "folds_5.json")
MAPPING_PATH = os.path.join(BASE, "category_mapping.json")
RESULTS_ROOT = os.path.join(BASE, "results/cv/fasterrcnn")

os.makedirs(RESULTS_ROOT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# category mapping 로드
def load_category_mapping():
    with open(MAPPING_PATH, "r") as f:
        mp = json.load(f)

    cat2yolo = mp["cat2yolo"]       # {원본 category_id: 0~N-1}
    sorted_cat = sorted(cat2yolo.keys(), key=lambda x: int(x))
    catid2idx = {int(k): int(v) for k, v in cat2yolo.items()}

    return sorted_cat, catid2idx


# JSON 파일들 스캔
def build_image_json_map():
    json_files = []
    for root, _, files in os.walk(TRAIN_ANN_DIR):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    img2anns = {}

    for jp in tqdm(json_files, desc="📄 JSON 로딩"):
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except:
            continue

        if "images" not in data or "annotations" not in data:
            continue

        img_name = data["images"][0]["file_name"]
        ann_list = data["annotations"]

        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        if img_name not in img2anns:
            img2anns[img_name] = []

        img2anns[img_name].extend(ann_list)

    return img2anns


# Dataset 클래스
class DetectionDataset(Dataset):
    def __init__(self, img_names, img2anns, catid2idx, transforms=None):
        self.img_names = img_names
        self.img2anns = img2anns
        self.catid2idx = catid2idx
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)

        img = Image.open(img_path).convert("RGB")
        anns = self.img2anns[img_name]

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            original_cid = int(ann["category_id"])
            labels.append(self.catid2idx[original_cid] + 1)  # FasterRCNN: 0=background

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


# Augmentation
class WeakAug:
    def __init__(self, train=True):
        self.train = train

    def __call__(self, img, target):
        if self.train:
            if torch.rand(1) < 0.5:
                w, h = img.size
                img = F.hflip(img)
                boxes = target["boxes"]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

            if torch.rand(1) < 0.5:
                img = F.adjust_brightness(img, 0.9 + 0.2 * torch.rand(1))
                img = F.adjust_contrast(img, 0.9 + 0.2 * torch.rand(1))

        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    return list(zip(*batch))


# FasterRCNN 모델 구성
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes + 1  # + background
    )
    return model


# mAP50 유사 점수 계산
def evaluate_mAP50(model, loader):
    model.eval()
    total_gt, total_match = 0, 0

    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            preds = model(imgs)

            for pred, tgt in zip(preds, tgts):
                gt_boxes = tgt["boxes"].to(DEVICE)
                pred_boxes = pred["boxes"]
                scores = pred["scores"]

                keep = scores > 0.1
                pred_boxes = pred_boxes[keep]

                if gt_boxes.numel() == 0:
                    continue

                total_gt += gt_boxes.size(0)

                if pred_boxes.numel() == 0:
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                max_iou, _ = ious.max(dim=1)

                total_match += (max_iou > 0.5).sum().item()

    if total_gt == 0:
        return 0.0
    return total_match / total_gt


# Fold 학습
def train_one_fold(fold_idx, folds, img2anns, catid2idx, num_classes):
    fold_dir = os.path.join(RESULTS_ROOT, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_imgs = folds[str(fold_idx)]["train"]
    val_imgs = folds[str(fold_idx)]["val"]

    # annotation 없는 이미지 제외
    train_imgs = [img for img in train_imgs if img in img2anns]
    val_imgs = [img for img in val_imgs if img in img2anns]

    train_ds = DetectionDataset(train_imgs, img2anns, catid2idx, transforms=WeakAug(train=True))
    val_ds   = DetectionDataset(val_imgs, img2anns, catid2idx, transforms=WeakAug(train=False))

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    model = build_model(num_classes).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    best_score = 0.0
    best_path = os.path.join(fold_dir, "best.pth")

    EPOCHS = 7

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, tgts in tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS}"):
            imgs = [img.to(DEVICE) for img in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]

            losses = model(imgs, tgts)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        val_score = evaluate_mAP50(model, val_loader)

        print(f"Fold {fold_idx} | Epoch {epoch+1} | Loss={total_loss:.4f} | mAP50={val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)

    with open(os.path.join(fold_dir, "cv_result.json"), "w") as f:
        json.dump({"mAP50_like": best_score}, f, indent=2)

    print(f"✅ Fold {fold_idx} 완료 — best mAP50={best_score:.4f}")
    return best_score


# MAIN
def main():
    print("\n📌 Step02-3: FasterRCNN 5-Fold CV 시작")

    with open(FOLDS_PATH, "r") as f:
        folds = json.load(f)

    sorted_cat, catid2idx = load_category_mapping()
    num_classes = len(sorted_cat)

    img2anns = build_image_json_map()

    all_scores = []
    for fold_idx in range(5):
        score = train_one_fold(fold_idx, folds, img2anns, catid2idx, num_classes)
        all_scores.append(score)

    print("\n🎉 FasterRCNN CV 완료")
    print("평균 mAP50-like =", np.mean(all_scores))


if __name__ == "__main__":
    main()