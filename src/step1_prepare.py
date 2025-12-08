import os
import json
import glob
import random
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold

# 기본 경로
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

DATA_DIR = os.path.join(BASE_PROJECT, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_ANN_DIR = os.path.join(DATA_DIR, "train_annotations")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

YOLO_BASE = os.path.join(BASE_PROJECT, "yolo_dataset")
os.makedirs(YOLO_BASE, exist_ok=True)

print("📌 TRAIN_IMG_DIR :", TRAIN_IMG_DIR)
print("📌 TRAIN_ANN_DIR :", TRAIN_ANN_DIR)
print("📌 TEST_IMG_DIR  :", TEST_IMG_DIR)
print("📌 YOLO_BASE     :", YOLO_BASE)


# JSON 스캔
print("\n[1] JSON 스캔 및 category 매핑 생성")

image_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".png")])
json_paths = glob.glob(os.path.join(TRAIN_ANN_DIR, "**", "*.json"), recursive=True)

print("총 이미지 수:", len(image_files))
print("총 JSON 파일 수:", len(json_paths))

json_map = {}
all_categories = set()

coco_images = []
coco_annotations = []
coco_categories_map = {}
global_ann_id = 1
global_img_id_map = {}
global_img_id_counter = 1

# JSON ↔ 이미지 매핑
for jp in tqdm(json_paths, desc="JSON 파싱"):
    try:
        with open(jp, "r") as f:
            data = json.load(f)
    except:
        print("⚠ JSON 파싱 오류:", jp)
        continue

    if "images" not in data or "annotations" not in data:
        continue

    img_info = data["images"][0]
    img_name = img_info["file_name"]

    if img_name not in image_files:
        continue

    json_map[img_name] = jp

    # 이미지 고유 ID 부여
    if img_name not in global_img_id_map:
        global_img_id_map[img_name] = global_img_id_counter
        coco_images.append({
            "id": global_img_id_counter,
            "file_name": img_name,
            "width": img_info.get("width", 0),
            "height": img_info.get("height", 0)
        })
        global_img_id_counter += 1

    img_id = global_img_id_map[img_name]

    # annotation 처리
    for ann in data["annotations"]:
        cid = ann["category_id"]
        all_categories.add(cid)

        coco_annotations.append({
            "id": global_ann_id,
            "image_id": img_id,
            "bbox": ann["bbox"],
            "category_id": cid
        })
        global_ann_id += 1

    # categories 처리
    for cat in data.get("categories", []):
        coco_categories_map[cat["id"]] = cat["name"]

print("JSON 있는 이미지:", len(json_map))
print("고유 category:", len(all_categories))

sorted_cat_ids = sorted(list(all_categories))
catid2idx = {cid: i for i, cid in enumerate(sorted_cat_ids)}
idx2catid = {i: cid for cid, i in catid2idx.items()}

# category_mapping.json 저장
mapping_path = os.path.join(BASE_PROJECT, "category_mapping.json")
with open(mapping_path, "w") as f:
    json.dump({
        "sorted_cat_ids": sorted_cat_ids,
        "catid2idx": catid2idx,
        "idx2catid": idx2catid
    }, f, indent=2, ensure_ascii=False)

print("✅ category_mapping.json 저장 완료:", mapping_path)


# 5-Fold split 생성
print("\n[2] 5-Fold 생성")

images_with_json = [img for img in image_files if img in json_map]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_splits = []
imgs_array = np.array(images_with_json)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(imgs_array)):
    fold_splits.append({
        "train": imgs_array[train_idx].tolist(),
        "val": imgs_array[val_idx].tolist()
    })

folds_path = os.path.join(BASE_PROJECT, "folds_5.json")
with open(folds_path, "w") as f:
    json.dump(fold_splits, f, indent=2, ensure_ascii=False)

print("✅ folds_5.json 저장 완료:", folds_path)


# YOLO용 기본 디렉토리 준비
print("\n[3] YOLO Dataset 초기화")

for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(YOLO_BASE, sub), exist_ok=True)

print("🎉 Step1 완료! 모든 사전 데이터 준비 완료!")