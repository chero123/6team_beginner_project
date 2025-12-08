import os
from ultralytics import YOLO
import yaml
import shutil
from PIL import Image
import json

HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

DATA_DIR = os.path.join(BASE_PROJECT, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_ANN_DIR = os.path.join(DATA_DIR, "train_annotations")

YOLO_BASE = os.path.join(BASE_PROJECT, "rtdetr_full")
os.makedirs(YOLO_BASE, exist_ok=True)

with open(os.path.join(BASE_PROJECT, "category_mapping.json")) as f:
    mapping = yaml.safe_load(f)

sorted_cat_ids = mapping["sorted_cat_ids"]
cat_names = {i: str(cid) for i, cid in enumerate(sorted_cat_ids)}
num_classes = len(cat_names)

json_paths = {}
for root, dirs, files in os.walk(TRAIN_ANN_DIR):
    for f in files:
        if f.endswith(".json"):
            json_paths[f] = os.path.join(root, f)

def write_label(img_name, dst_path):
    json_name = img_name.replace(".png", ".json")
    if json_name not in json_paths:
        return

    with open(json_paths[json_name]) as f:
        data = json.load(f)

    anns = data["annotations"]
    img_w, img_h = Image.open(os.path.join(TRAIN_IMG_DIR, img_name)).size

    lines = []
    for ann in anns:
        cid = ann["category_id"]
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        bw = w / img_w
        bh = h / img_h
        cls_idx = mapping["catid2idx"][str(cid)]
        lines.append(f"{cls_idx} {cx} {cy} {bw} {bh}")

    with open(dst_path, "w") as f:
        f.write("\n".join(lines))

def prepare_full_dataset():
    for sub in ["images", "labels"]:
        shutil.rmtree(os.path.join(YOLO_BASE, sub), ignore_errors=True)
        os.makedirs(os.path.join(YOLO_BASE, sub), exist_ok=True)

    for img_name in os.listdir(TRAIN_IMG_DIR):
        if not img_name.endswith(".png"):
            continue
        shutil.copy2(
            os.path.join(TRAIN_IMG_DIR, img_name),
            os.path.join(YOLO_BASE, "images", img_name)
        )
        write_label(
            img_name,
            os.path.join(YOLO_BASE, "labels", img_name.replace(".png", ".txt"))
        )

def build_yaml():
    data = {
        "path": YOLO_BASE,
        "train": os.path.join(YOLO_BASE, "images"),
        "val": os.path.join(YOLO_BASE, "images"),
        "names": cat_names,
        "nc": num_classes
    }
    yaml_path = os.path.join(YOLO_BASE, "dataset_full.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return yaml_path

def run_full_train_rtdetr(epochs=200):
    yaml_path = build_yaml()
    prepare_full_dataset()

    model = YOLO(os.path.join(BASE_PROJECT, "rtdetr-l.pt"))

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=12,
        device=0,
        patience=25,
        project=os.path.join(BASE_PROJECT, "runs_full"),
        name="rtdetr_full",
        exist_ok=True
    )

    print("\n🎉 RT-DETR-L Full Training 완료!")
    print("📁 Best weight:", results.save_dir)

if __name__ == "__main__":
    run_full_train_rtdetr(epochs=200)