import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# 기본 경로 설정
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

DATA_DIR = os.path.join(BASE_PROJECT, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_ANN_DIR = os.path.join(DATA_DIR, "train_annotations")

YOLO_BASE = os.path.join(BASE_PROJECT, "yolo_dataset")
RESULT_CV_PATH = os.path.join(BASE_PROJECT, "results", "cv")
os.makedirs(RESULT_CV_PATH, exist_ok=True)

with open(os.path.join(BASE_PROJECT, "category_mapping.json")) as f:
    mapping = json.load(f)
sorted_cat_ids = mapping["sorted_cat_ids"]
catid2idx = mapping["catid2idx"]
idx2catid = {v: int(k) for k, v in catid2idx.items()}

# fold 파일 로드
with open(os.path.join(BASE_PROJECT, "folds_5.json")) as f:
    folds = json.load(f)

# annotation json 캐싱
json_paths = {}
for root, dirs, files in os.walk(TRAIN_ANN_DIR):
    for f in files:
        if f.endswith(".json"):
            json_paths[f] = os.path.join(root, f)


# Dataset 정의
class CustomDataset(Dataset):
    def __init__(self, img_list, img_dir):
        self.img_list = img_list
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.

        ann_name = img_name.replace(".png", ".json")
        ann_path = json_paths[ann_name]

        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        boxes = []
        labels = []

        for ann in ann_data["annotations"]:
            cid = ann["category_id"]
            if str(cid) not in catid2idx:
                continue

            bbox = ann["bbox"]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(catid2idx[str(cid)])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_name.replace(".png", "")  # 문자열 ID 지원
        }

        return img_tensor, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# COCO mAP 계산 함수
def compute_coco_map(gt_ann_list, pred_list):
    """
    gt_ann_list: [{image_id, category_id, bbox, area, id, iscrowd}]
    pred_list:   [{image_id, category_id, bbox, score}]
    """

    # ---- FIX: 'info'와 'licenses' 추가 ----
    coco_gt = {
        "info": {"description": "pill dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    img_ids = set()

    for g in gt_ann_list:
        img_ids.add(g["image_id"])
        coco_gt["annotations"].append(g)

    for img_id in img_ids:
        coco_gt["images"].append({"id": img_id})

    for cid in sorted_cat_ids:
        coco_gt["categories"].append({"id": cid, "name": str(cid)})

    temp_gt = os.path.join(BASE_PROJECT, "temp_gt.json")
    temp_pred = os.path.join(BASE_PROJECT, "temp_pred.json")

    with open(temp_gt, "w") as f:
        json.dump(coco_gt, f)

    with open(temp_pred, "w") as f:
        json.dump(pred_list, f)

    coco_gt = COCO(temp_gt)
    coco_pred = coco_gt.loadRes(temp_pred)

    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP50 = coco_eval.stats[1]
    mAP5095 = coco_eval.stats[0]

    return mAP50, mAP5095

# Faster R-CNN Cross-Validation
def run_frcnn_cv(epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_scores = []

    for fold_idx in range(5):
        print("\n----------- Fold", fold_idx, "-----------")

        train_imgs = folds[fold_idx]["train"]
        val_imgs = folds[fold_idx]["val"]

        train_dataset = CustomDataset(train_imgs, TRAIN_IMG_DIR)
        val_dataset = CustomDataset(val_imgs, TRAIN_IMG_DIR)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                                collate_fn=collate_fn)

        model = fasterrcnn_resnet50_fpn(num_classes=len(sorted_cat_ids)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Training
        model.train()
        for epoch in range(epochs):
            print(f"[Fold {fold_idx}] Epoch {epoch}")
            for imgs, targets in tqdm(train_loader):
                imgs = [img.to(device) for img in imgs]

                new_targets = []
                for t in targets:
                    new_targets.append({
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device)
                    })

                loss_dict = model(imgs, new_targets)
                loss = sum(v for v in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # GT / Pred 생성 → COCO 평가
        print(f"[Fold {fold_idx}] Running Validation...")

        model.eval()
        gt_list = []
        pred_list = []

        ann_id = 1  # annotation_id 증가용

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)

                for i, out in enumerate(outputs):
                    img_id = targets[i]["image_id"]

                    # GT
                    for b, lab in zip(targets[i]["boxes"], targets[i]["labels"]):
                        x1, y1, x2, y2 = b.tolist()
                        w = x2 - x1
                        h = y2 - y1

                        gt_list.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": idx2catid[int(lab)],
                            "bbox": [x1, y1, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1

                    # Pred
                    for b, s, lab in zip(out["boxes"], out["scores"], out["labels"]):
                        x1, y1, x2, y2 = b.tolist()
                        w = x2 - x1
                        h = y2 - y1

                        pred_list.append({
                            "image_id": img_id,
                            "category_id": idx2catid[int(lab)],
                            "bbox": [x1, y1, w, h],
                            "score": float(s)
                        })

        mAP50, mAP5095 = compute_coco_map(gt_list, pred_list)

        print(f"Fold {fold_idx} mAP50 = {mAP50:.4f}, mAP5095 = {mAP5095:.4f}")

        fold_scores.append(mAP50)

    # 결과 저장
    avg_score = float(np.mean(fold_scores))
    out_path = os.path.join(RESULT_CV_PATH, "fasterrcnn_map.json")

    with open(out_path, "w") as f:
        json.dump({
            "model": "fasterrcnn_resnet50_fpn",
            "fold_scores": fold_scores,
            "avg_score": avg_score
        }, f, indent=2)

    print("\n🎉 Faster R-CNN (mAP Version) 5-Fold CV 완료!")
    print("📌 평균 mAP50:", avg_score)
    print("📁 저장 위치:", out_path)


if __name__ == "__main__":
    run_frcnn_cv(epochs=1)
