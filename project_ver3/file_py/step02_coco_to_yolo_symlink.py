# =========================================
# STEP 02 (FAST + CONSISTENT)
# COCO(train_id) -> YOLO labels/all + images/all (symlink)
# - No per-image os.walk (VERY slow)
# - Build basename->fullpath index in ONE walk (only needed files)
# =========================================

import os
import json
from collections import defaultdict
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
COCO_JSON = (
    "/home/ohs3201/6team_beginner_project/project_ver3/"
    "coco/train_coco_dlidx.json"
)

IMG_SRC_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터"

YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"
IMG_DST = os.path.join(YOLO_ROOT, "images/all")
LBL_DST = os.path.join(YOLO_ROOT, "labels/all")

os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

# =========================
# LOAD COCO
# =========================
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])

print(f"[STEP 02] images: {len(images)}")
print(f"[STEP 02] annotations: {len(annotations)}")

# image_id -> image meta
img_by_id = {im["id"]: im for im in images}

# image_id -> anns
anns_by_img = defaultdict(list)
for a in annotations:
    anns_by_img[a["image_id"]].append(a)

print(f"[STEP 02] images with bbox: {len(anns_by_img)}")

# =========================
# 1) Build "needed filenames" set
# =========================
needed = set()
for img_id in anns_by_img.keys():
    needed.add(img_by_id[img_id]["file_name"])

print(f"[STEP 02] needed image files: {len(needed)}")

# =========================
# 2) One-time index build (ONLY needed files)
# =========================
print("[STEP 02] Indexing source images with one os.walk (only needed)...")
img_index = {}
dup = 0

# 원천데이터가 크더라도 "한 번만" 걷고 끝냄
for root, _, files in tqdm(os.walk(IMG_SRC_ROOT), desc="Indexing", mininterval=5):
    # files가 아주 클 수 있으니 set intersection 방식 사용
    for fn in files:
        if fn in needed:
            if fn in img_index:
                dup += 1
                continue
            img_index[fn] = os.path.join(root, fn)

print(f"[STEP 02] indexed: {len(img_index)} / needed {len(needed)} (dup ignored: {dup})")

# =========================
# 3) Convert (symlink + label)
# =========================
skipped_no_image = 0
written_labels = 0
linked_images = 0
skipped_bad_bbox = 0

def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

for img_id, anns in tqdm(anns_by_img.items(), desc="COCO->YOLO", mininterval=5):
    im = img_by_id.get(img_id)
    if im is None:
        continue

    fname = im["file_name"]
    W = float(im["width"])
    H = float(im["height"])

    src = img_index.get(fname)
    if src is None:
        skipped_no_image += 1
        continue

    # --------- symlink image
    dst_img = os.path.join(IMG_DST, fname)
    if not os.path.exists(dst_img):
        try:
            os.symlink(src, dst_img)
            linked_images += 1
        except FileExistsError:
            pass
        except OSError:
            # 권한/파일시스템 문제 등
            # 그래도 라벨은 만들 수 있으니 진행
            pass

    # --------- write label
    label_path = os.path.join(LBL_DST, os.path.splitext(fname)[0] + ".txt")

    lines = []
    for a in anns:
        bbox = a.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            skipped_bad_bbox += 1
            continue

        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            skipped_bad_bbox += 1
            continue

        # COCO bbox -> YOLO normalized
        cx = (x + bw / 2.0) / W
        cy = (y + bh / 2.0) / H
        nw = bw / W
        nh = bh / H

        # 약간의 오차 방지 clamp
        cx = clamp01(cx)
        cy = clamp01(cy)
        nw = clamp01(nw)
        nh = clamp01(nh)

        if nw <= 0 or nh <= 0:
            skipped_bad_bbox += 1
            continue

        cls = int(a["category_id"])  # ✅ 이미 train_id(0-based)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if lines:
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written_labels += 1

print("\n[DONE] STEP 02 completed")
print(f" - yolo root: {YOLO_ROOT}")
print(f" - images symlinked newly: {linked_images}")
print(f" - label files written: {written_labels}")
print(f" - skipped (image not found): {skipped_no_image}")
print(f" - skipped bad bbox lines: {skipped_bad_bbox}")