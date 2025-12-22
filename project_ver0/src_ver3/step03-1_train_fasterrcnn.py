import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"

FRCNN_COCO_DIR = f"{BASE}/frcnn_dataset_v3"
TRAIN_JSON = f"{FRCNN_COCO_DIR}/train.json"
VAL_JSON   = f"{FRCNN_COCO_DIR}/val.json"

RESULT_DIR = f"{BASE}/results_v3/fasterrcnn"
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False

    def check(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True


# category mapping
with open(os.path.join(BASE, "category_mapping.json"), "r") as f:
    mp = json.load(f)

yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}
NUM_CLASSES = len(yolo2cat)  # 28


class CocoFRCNNDataset(Dataset):
    def __init__(self, json_path):
        self.coco = COCO(json_path)
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs([img_id])[0]

        file_name = img_info["file_name"]

        # YOLO dataset에서 train / val 둘 다 찾도록
        img_train = os.path.join(BASE, "yolo_dataset_v3/images/train", file_name)
        img_val   = os.path.join(BASE, "yolo_dataset_v3/images/val", file_name)
        if os.path.exists(img_train):
            img_path = img_train
        else:
            img_path = img_val

        from PIL import Image
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            cid = ann["category_id"]  # 0~27 (STEP02에서 이렇게 만들어줌)
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(int(cid) + 1)  # background(0) + 1~28

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        img = F.to_tensor(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        return img, target


def collate_fn(batch):
    return list(zip(*batch))


def build_model(num_classes: int):
    # num_classes: 실제 클래스 수(28)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feat, num_classes + 1  # background 포함해서 29
    )
    return model


def train_frcnn(epochs=150, lr=1e-4, batch_size=4):
    print("\n========== STEP03-1: FasterRCNN Training (28 classes) ==========")
    print(f"- Epochs  : {epochs}")
    print(f"- Classes : {NUM_CLASSES}")

    train_ds = CocoFRCNNDataset(TRAIN_JSON)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model = build_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    early = EarlyStopping(patience=15, min_delta=0.0005)

    best_loss = float("inf")
    best_path = os.path.join(RESULT_DIR, "best.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, targets in tqdm(train_dl, desc=f"[FRCNN] Epoch {epoch+1}/{epochs}"):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss = {avg_loss:.4f}")

        # EarlyStopping
        early.check(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            print(f"💾 Best updated → {best_path}")

        if early.stop:
            print("⛔ Early Stopping triggered!")
            break

    print("\n🎉 FasterRCNN Training 완료")
    print(f"📁 Best model path: {best_path}")


if __name__ == "__main__":
    train_frcnn(epochs=150, batch_size=4, lr=1e-4)