import os
import json
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))

# 원본 경로
data_root = os.path.join(BASE, "data_ai06")
img_root = os.path.join(data_root, "train_images")
ann_root = os.path.join(data_root, "train_annotations")

# YOLO 저장 폴더
save_root = os.path.join(BASE, "yolo_dataset")
os.makedirs(save_root, exist_ok=True)

for split in ["train", "val"]:
    os.makedirs(os.path.join(save_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(save_root, "labels", split), exist_ok=True)

# ✅ 분할 정보 읽기 (프로젝트 루트에 split.json 있음)
with open(os.path.join(BASE, "split.json"), "r", encoding="utf-8") as f:
    split_info = json.load(f)

# ✅ category_id mapping 읽기 (역시 루트에 존재)
with open(os.path.join(BASE, "category_id_mapping.json"), "r", encoding="utf-8") as f:
    cat_map = json.load(f)
old_ids = sorted([int(k) for k in cat_map.keys()])
id_map = {old: idx for idx, old in enumerate(old_ids)}

def convert(split):
    json_files = split_info[split]

    for json_file in json_files:
        ann_path = os.path.join(ann_root, json_file)
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_info = data["images"][0]
        img_name = img_info["file_name"]
        img_file = os.path.basename(img_name)
        img_path = os.path.join(img_root, img_file)

        # YOLO 이미지 저장 위치
        dst_img_path = os.path.join(save_root, "images", split, img_file)
        label_path = os.path.join(save_root, "labels", split, img_file.replace(".png", ".txt"))

        # 이미지 복사
        img = cv2.imread(img_path)
        if img is None:
            print("이미지 없음:", img_path)
            continue

        cv2.imwrite(dst_img_path, img)

        h, w = img.shape[:2]
        lines = []

        for ann in data["annotations"]:
            x, y, bw, bh = ann["bbox"]
            old_cid = int(ann["category_id"])
            cid = id_map[old_cid]

            # YOLO 형식: class cx cy w h (normalized)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            w_norm = bw / w
            h_norm = bh / h

            lines.append(f"{cid} {cx} {cy} {w_norm} {h_norm}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))


print("👉 Train 변환중…")
convert("train")

print("👉 Val 변환중…")
convert("val")

print("🎉 YOLO 변환 완료!")


##########################

import os

BASE = os.path.dirname(os.path.abspath(__file__))
yolo_root = os.path.join(BASE, "yolo_dataset")

def count_files(path):
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

paths = {
    "train_images": os.path.join(yolo_root, "images", "train"),
    "val_images": os.path.join(yolo_root, "images", "val"),
    "train_labels": os.path.join(yolo_root, "labels", "train"),
    "val_labels": os.path.join(yolo_root, "labels", "val"),
}

print("\n📌 YOLO 변환 결과 확인\n")

for name, path in paths.items():
    print(f"{name}: {count_files(path)} files (path: {path})")

print("\n🔍 yolo_dataset 폴더 내부 구조 체크 완료!")
