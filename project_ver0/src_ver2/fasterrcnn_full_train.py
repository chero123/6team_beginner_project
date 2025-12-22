import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"
COCO_DIR = f"{BASE}/yolo_dataset/coco"

TRAIN_JSON = f"{COCO_DIR}/train.json"
VAL_JSON = f"{COCO_DIR}/val.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# COCO DATASET
class CocoDataset(Dataset):
    def __init__(self, json_path):
        from pycocotools.coco import COCO
        self.coco = COCO(json_path)
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs([img_id])[0]

        img_path = os.path.join(BASE, "data/train_images", img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])  # no +1 (we use 0..C-1)

        # 최소 1개 이상 box가 있어야 함
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    return list(zip(*batch))


# MODEL BUILDER (num_classes = number of real classes)
def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features

    # num_classes 그대로 → background 자동 포함됨
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes
    )
    return model


# FULL TRAINING FUNCTION
def train_fasterrcnn_full(pretrained_weight, output_dir, epochs=70, batch_size=4):

    print("🚀 FasterRCNN FULL TRAINING 시작")

    if not os.path.exists(TRAIN_JSON):
        raise FileNotFoundError("❌ train.json 없음! step01-2_make_coco_dataset.py 먼저 실행!")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset
    train_ds = CocoDataset(TRAIN_JSON)
    val_ds = CocoDataset(VAL_JSON)

    num_classes = len(train_ds.coco.getCatIds())   # 숫자 56개
    print(f"📌 num_classes = {num_classes}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    # 2. Model
    model = build_model(num_classes)
    model = model.to(DEVICE)

    # 로드할 때 불일치 나는 head 부분은 skip해서 load
    if pretrained_weight is not None and os.path.exists(pretrained_weight):
        print(f"🔄 Pretrained Weight 로드: {pretrained_weight}")
        ckpt = torch.load(pretrained_weight, map_location=DEVICE)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in ckpt.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    else:
        print("⚠ Pretrained Weight 없음 → 랜덤 초기화 진행")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


    # TRAIN LOOP + EARLY STOPPING
    best_loss = float("inf")
    patience = 10
    patience_count = 0

    best_path = os.path.join(output_dir, "best.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, targets in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"📉 Epoch {epoch+1} | Train Loss = {avg_loss:.4f}")

        # BEST?
        if avg_loss < best_loss:
            print(f"💾 Loss 개선! {best_loss:.4f} → {avg_loss:.4f} 저장")
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            patience_count = 0
        else:
            patience_count += 1
            print(f"⚠ 개선 없음 ({patience_count}/{patience})")

            if patience_count >= patience:
                print("🛑 Early Stopping 발동!")
                break

    print(f"🎉 Full Training 완료 → best.pth: {best_path}")
    return best_path