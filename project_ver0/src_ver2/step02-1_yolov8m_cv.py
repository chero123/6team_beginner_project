import os
import json
from ultralytics import YOLO

BASE = "/home/ohs3201/6team_beginner_project"
YOLO_DIR = f"{BASE}/yolo_dataset"
FOLDS_PATH = f"{BASE}/folds_5.json"
MAP_PATH = f"{BASE}/category_mapping.json"
RESULTS_ROOT = f"{BASE}/results/cv/yolov8m"

os.makedirs(RESULTS_ROOT, exist_ok=True)

print("📌 Step02-1: YOLOv8m 5-Fold CV 시작")

# 1) folds & mapping 로드
with open(FOLDS_PATH, "r") as f:
    folds = json.load(f)  

with open(MAP_PATH, "r") as f:
    mapping = json.load(f)
cat2yolo = mapping["cat2yolo"]
num_classes = len(cat2yolo)

IM_TRAIN_DIR = os.path.join(YOLO_DIR, "images/train")
IM_VAL_DIR   = os.path.join(YOLO_DIR, "images/val")

def resolve_img_path(img_name):
    p1 = os.path.join(IM_TRAIN_DIR, img_name)
    p2 = os.path.join(IM_VAL_DIR, img_name)
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    return None

# 2) fold 단위 수행
for fold_idx in range(5):
    fold = folds[str(fold_idx)]
    train_imgs = fold["train"]
    val_imgs   = fold["val"]

    fold_dir = os.path.join(RESULTS_ROOT, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train.txt")
    val_txt   = os.path.join(fold_dir, "val.txt")
    yaml_path = os.path.join(fold_dir, "dataset.yaml")

    # train.txt / val.txt 생성
    with open(train_txt, "w") as ft:
        for img in train_imgs:
            p = resolve_img_path(img)
            if p: ft.write(p + "\n")

    with open(val_txt, "w") as fv:
        for img in val_imgs:
            p = resolve_img_path(img)
            if p: fv.write(p + "\n")

    # dataset.yaml 생성
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write(f"train: {train_txt}\n")
        f.write(f"val: {val_txt}\n")
        f.write("names:\n")
        for idx in range(num_classes):
            f.write(f"  {idx}: cls_{idx}\n")

    # YOLOv8m 학습
    print(f"\n🚀 Fold {fold_idx} - YOLOv8m 학습 시작")
    model = YOLO("yolov8m.pt")

    model.train(
        data=yaml_path,
        epochs=10,       # 안정적 CV epoch
        imgsz=640,
        batch=16,
        device=0,
        project=RESULTS_ROOT,
        name=f"fold{fold_idx}",
        patience=3,
        workers=4,
        seed=0,
        exist_ok=True
    )

print("\n🎉 YOLOv8m 5-Fold CV 완료!")