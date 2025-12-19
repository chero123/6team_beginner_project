import os
import json
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# =========================
# PATH
# =========================
IN_JSON = "/home/ohs3201/work/step2_clean_coco/clean.json"
MAPPING_JSON = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"

OUT_ROOT = "/home/ohs3201/work/step4_yolov8"
IMG_OUT = os.path.join(OUT_ROOT, "images")
LBL_OUT = os.path.join(OUT_ROOT, "labels")

for d in [IMG_OUT, LBL_OUT]:
    os.makedirs(os.path.join(d, "train"), exist_ok=True)
    os.makedirs(os.path.join(d, "val"), exist_ok=True)

# =========================
# CONFIG
# =========================
VAL_RATIO = 0.2
SEED = 42
random.seed(SEED)

# =========================
# LOAD COCO
# =========================
with open(IN_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

img_by_id = {im["id"]: im for im in images}

anns_by_img = defaultdict(list)
for an in annotations:
    anns_by_img[an["image_id"]].append(an)

# =========================
# LOAD MAPPING (dl_idx -> cls)
# =========================
raw_map = json.load(open(MAPPING_JSON, encoding="utf-8"))

dlidx_to_cls = {}
for dl_str, v in raw_map.items():
    dl = int(dl_str)
    if isinstance(v, dict) and "cls" in v:
        dlidx_to_cls[dl] = int(v["cls"])
    else:
        # 예: "동아가바펜틴정 800mg (cls 16)"
        s = str(v)
        cls = int(s.split("cls")[-1].strip(" )"))
        dlidx_to_cls[dl] = cls

# 🔑 방법 A: mapping 기준 nc 결정
NC = max(dlidx_to_cls.values()) + 1
print(f"[INFO] nc inferred from mapping = {NC}")

# =========================
# TRAIN / VAL SPLIT
# =========================
img_ids = [im["id"] for im in images]
random.shuffle(img_ids)

n_val = int(len(img_ids) * VAL_RATIO)
val_ids = set(img_ids[:n_val])
train_ids = set(img_ids[n_val:])

# =========================
# IMAGE SOURCE ROOTS
# (STEP 1과 동일)
# =========================
IMAGE_ROOTS = [
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터",
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/yolo_dataset/images",
    "/home/ohs3201/6team_beginner_project/data/train_images",
]

def find_image(fn):
    for r in IMAGE_ROOTS:
        p = os.path.join(r, fn)
        if os.path.exists(p):
            return p
    return None

# =========================
# CONVERT
# =========================
print("[STEP 4] Converting COCO -> YOLO")

for im in tqdm(images):
    iid = im["id"]
    fn = im["file_name"]
    W, H = im["width"], im["height"]

    split = "val" if iid in val_ids else "train"

    src = find_image(fn)
    if not src:
        continue

    dst_img = os.path.join(IMG_OUT, split, fn)
    if not os.path.exists(dst_img):
        shutil.copy2(src, dst_img)

    label_path = os.path.join(LBL_OUT, split, os.path.splitext(fn)[0] + ".txt")
    with open(label_path, "w") as f:
        for an in anns_by_img[iid]:
            dl = int(an["category_id"])
            if dl not in dlidx_to_cls:
                continue

            cls = dlidx_to_cls[dl]

            x, y, w, h = an["bbox"]

            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H

            if nw <= 0 or nh <= 0:
                continue

            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

# =========================
# WRITE data.yaml
# =========================
yaml_path = os.path.join(OUT_ROOT, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT_ROOT}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {NC}\n")
    f.write("names:\n")
    for i in range(NC):
        f.write(f"  - class_{i}\n")

print("[DONE] STEP 4 completed")
print(f"Saved to: {OUT_ROOT}")
print(f"nc = {NC}")