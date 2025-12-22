import os
import json
from collections import defaultdict
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"

COCO_JSON = os.path.join(
    PROJECT_ROOT,
    "coco",
    "train_coco_trainid_SAFE_FULL_IMAGES.json"
)

# 원천데이터 최상위 (여기 아래에 train_images, TS_* 전부 있음)
IMG_SRC_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터"

YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")
IMG_DST = os.path.join(YOLO_ROOT, "images", "all")
LBL_DST = os.path.join(YOLO_ROOT, "labels", "all")

os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

# =========================
# LOAD COCO
# =========================
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

id2img = {im["id"]: im for im in images}

print(f"[STEP 02 SAFE] COCO images      : {len(images)}")
print(f"[STEP 02 SAFE] COCO annotations : {len(annotations)}")

# =========================
# GROUP ANNS BY image_id
# =========================
imgid_to_anns = defaultdict(list)
for an in annotations:
    imgid_to_anns[an["image_id"]].append(an)

print(f"[STEP 02 SAFE] images with bbox : {len(imgid_to_anns)}")

# =========================
# BUILD SOURCE IMAGE INDEX (🔥 핵심 수정)
# =========================
print("[STEP 02 SAFE] Indexing source images (recursive)...")

src_index = {}
dup = 0

for root, _, files in tqdm(os.walk(IMG_SRC_ROOT), desc="Index source"):
    for fn in files:
        if not fn.lower().endswith(".png"):
            continue
        if fn in src_index:
            dup += 1
            continue
        src_index[fn] = os.path.join(root, fn)

print(f"[STEP 02 SAFE] indexed images : {len(src_index)} (dup ignored: {dup})")

# =========================
# WRITE YOLO
# =========================
symlinked = 0
labels_written = 0
skipped_missing = 0

for img_id, anns in tqdm(imgid_to_anns.items(), desc="COCO → YOLO"):
    im = id2img[img_id]
    fname = im["file_name"]
    W, H = im["width"], im["height"]

    src_img = src_index.get(fname)
    if src_img is None:
        skipped_missing += 1
        continue

    # 1) symlink image
    dst_img = os.path.join(IMG_DST, fname)
    if not os.path.exists(dst_img):
        os.symlink(src_img, dst_img)
        symlinked += 1

    # 2) label
    label_path = os.path.join(LBL_DST, os.path.splitext(fname)[0] + ".txt")
    lines = []

    for an in anns:
        x, y, w, h = an["bbox"]
        cls = an["category_id"]

        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H

        if nw <= 0 or nh <= 0:
            continue

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if lines:
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        labels_written += 1

print("\n[DONE] STEP 02 SAFE FINAL")
print(f" - images symlinked : {symlinked}")
print(f" - labels written  : {labels_written}")
print(f" - skipped missing : {skipped_missing}")
print(f" - YOLO root       : {YOLO_ROOT}")