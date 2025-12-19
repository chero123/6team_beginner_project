# project_ver4/file_py/step02_coco_to_yolo_symlink_SAFE.py
import os
import json
from collections import defaultdict
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"

COCO_JSON = os.path.join(PROJECT_ROOT, "coco", "train_coco_trainid_SAFE.json")

# 원천 이미지 루트(실제 png들이 있는 곳)
IMG_SRC_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터"

YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")
IMG_DST = os.path.join(YOLO_ROOT, "images", "all")
LBL_DST = os.path.join(YOLO_ROOT, "labels", "all")

os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

# =========================
# UTIL
# =========================
def safe_float(x):
    try:
        return float(x)
    except:
        return None

def clamp01(v):
    return max(0.0, min(1.0, v))

def coco_bbox_to_yolo(bbox, W, H):
    """COCO [x,y,w,h] -> YOLO [cx,cy,w,h] normalized"""
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or W <= 0 or H <= 0:
        return None

    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    nw = w / W
    nh = h / H

    # 안전 clamp (혹시 경계로 인해 약간 넘어갈 수 있어서)
    cx = clamp01(cx)
    cy = clamp01(cy)
    nw = clamp01(nw)
    nh = clamp01(nh)

    # 너무 작거나 이상치 제거
    if nw <= 0 or nh <= 0:
        return None

    return cx, cy, nw, nh

# =========================
# LOAD COCO
# =========================
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])

# image_id -> meta
id2img = {im["id"]: im for im in images}

print(f"[STEP 02 SAFE] images: {len(images)}")
print(f"[STEP 02 SAFE] annotations: {len(annotations)}")

# =========================
# GROUP ANNS BY image_id
# =========================
imgid_to_anns = defaultdict(list)
for an in annotations:
    imgid_to_anns[an["image_id"]].append(an)

print(f"[STEP 02 SAFE] images with bbox: {len(imgid_to_anns)}")

# =========================
# NEED FILE NAMES SET
# =========================
needed_files = set()
for img_id in imgid_to_anns.keys():
    im = id2img.get(img_id)
    if im:
        needed_files.add(im["file_name"])

print(f"[STEP 02 SAFE] needed image files: {len(needed_files)}")

# =========================
# INDEX SOURCE IMAGES (one pass os.walk)
# =========================
print("[STEP 02 SAFE] Indexing source images with one os.walk (only needed)...")

src_index = {}
dup = 0

for root, _, files in tqdm(os.walk(IMG_SRC_ROOT), desc="Indexing"):
    for fn in files:
        if fn not in needed_files:
            continue
        full = os.path.join(root, fn)
        if fn in src_index:
            dup += 1  # 같은 파일명이 여러 경로에 존재 (위험)
            # 기존 것을 유지 (원하면 여기서 규칙을 바꿀 수 있음)
            continue
        src_index[fn] = full

print(f"[STEP 02 SAFE] indexed: {len(src_index)} / needed {len(needed_files)} (dup ignored: {dup})")

# =========================
# WRITE YOLO
# =========================
symlink_new = 0
labels_written = 0
skipped_not_found = 0
skipped_bad_lines = 0

for img_id, anns in tqdm(imgid_to_anns.items(), desc="COCO->YOLO"):
    im = id2img.get(img_id)
    if im is None:
        skipped_bad_lines += 1
        continue

    fname = im.get("file_name")
    W = im.get("width")
    H = im.get("height")
    if not fname or not W or not H:
        skipped_bad_lines += 1
        continue

    src_img = src_index.get(fname)
    if src_img is None:
        skipped_not_found += 1
        continue

    # 1) symlink image
    dst_img = os.path.join(IMG_DST, fname)
    if not os.path.exists(dst_img):
        try:
            os.symlink(src_img, dst_img)
            symlink_new += 1
        except FileExistsError:
            pass

    # 2) write label file (file_name 기준: 덮어쓰기 X, 이 COCO는 병합 구조라 안전)
    label_path = os.path.join(LBL_DST, os.path.splitext(fname)[0] + ".txt")

    lines = []
    for an in anns:
        bbox = an.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped_bad_lines += 1
            continue

        cls = an.get("category_id")  # train_id (0-based)
        try:
            cls = int(cls)
        except:
            skipped_bad_lines += 1
            continue

        y = coco_bbox_to_yolo(bbox, W, H)
        if y is None:
            skipped_bad_lines += 1
            continue

        cx, cy, nw, nh = y
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if not lines:
        # 이 경우는 원래 COCO에서 annotation이 있었는데 다 걸러진 케이스
        # 라벨 파일을 만들지 않는게 안전
        continue

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    labels_written += 1

print("\n[DONE] STEP 02 SAFE")
print(f" - yolo root: {YOLO_ROOT}")
print(f" - images symlinked newly: {symlink_new}")
print(f" - label files written: {labels_written}")
print(f" - skipped (image not found): {skipped_not_found}")
print(f" - skipped bad bbox/lines: {skipped_bad_lines}")