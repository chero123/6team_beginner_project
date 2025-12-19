import os
import cv2
import random
from collections import Counter, defaultdict

import albumentations as A

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")
IMG_ALL = os.path.join(YOLO_ROOT, "images", "all")
LBL_ALL = os.path.join(YOLO_ROOT, "labels", "all")

IMG_AUG = os.path.join(YOLO_ROOT, "images", "aug")
LBL_AUG = os.path.join(YOLO_ROOT, "labels", "aug")
os.makedirs(IMG_AUG, exist_ok=True)
os.makedirs(LBL_AUG, exist_ok=True)

# -----------------------
# SETTINGS (ver3 감각 유지)
# -----------------------
VAL_EXCLUDE_SUFFIX = ("_aug1", "_aug2", "_aug3")  # 혹시 이미 들어온 aug가 있다면 제외
RARE_Q = 0.20          # 하위 20% 클래스를 rare로
AUG_PER_IMAGE = 2      # 희소 이미지당 증강 2장 생성 (너무 과하면 과적합/중복 위험)
SEED = 0
random.seed(SEED)

# -----------------------
# helpers
# -----------------------
def read_yolo_label(path):
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except:
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            boxes.append([cx, cy, w, h, cls])
    return boxes

def write_yolo_label(path, boxes):
    # boxes: [cx,cy,w,h,cls]
    with open(path, "w", encoding="utf-8") as f:
        for cx, cy, w, h, cls in boxes:
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def yolo_to_albu(boxes):
    # albumentations expects bbox=(x_min,y_min,x_max,y_max) in normalized coords when format='yolo'? 아니고,
    # 여기선 format='yolo'를 쓰면 (x_center,y_center,w,h) 그대로 넣으면 됨.
    # label_fields에 cls 따로.
    b = []
    c = []
    for cx, cy, w, h, cls in boxes:
        b.append([cx, cy, w, h])
        c.append(int(cls))
    return b, c

def albu_to_yolo(bboxes, clses):
    out = []
    for (cx, cy, w, h), cls in zip(bboxes, clses):
        if w <= 0 or h <= 0:
            continue
        # clamp
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        if w <= 0 or h <= 0:
            continue
        out.append([cx, cy, w, h, int(cls)])
    return out

# -----------------------
# 1) class distribution from ALL
# -----------------------
label_files = [f for f in os.listdir(LBL_ALL) if f.endswith(".txt")]
cnt = Counter()

img2boxes = {}
for lf in label_files:
    # 혹시 이미 aug 파일이 all에 섞여있으면 제외
    stem = os.path.splitext(lf)[0]
    if any(stem.endswith(suf) for suf in VAL_EXCLUDE_SUFFIX):
        continue
    p = os.path.join(LBL_ALL, lf)
    boxes = read_yolo_label(p)
    if not boxes:
        continue
    img2boxes[lf] = boxes
    for *_, cls in boxes:
        cnt[cls] += 1

if not cnt:
    raise RuntimeError("No valid labels found in labels/all")

# rare classes = bottom RARE_Q by instances
sorted_by_freq = sorted(cnt.items(), key=lambda x: x[1])
k = max(1, int(len(sorted_by_freq) * RARE_Q))
rare_classes = set([c for c, _ in sorted_by_freq[:k]])

print("[INFO] rare classes:", sorted(list(rare_classes))[:50], "..." if len(rare_classes) > 50 else "")
print(f"[INFO] rare count: {len(rare_classes)} / total classes: {len(cnt)}")

# -----------------------
# 2) select images containing rare classes
# -----------------------
targets = []
for lf, boxes in img2boxes.items():
    if any((b[4] in rare_classes) for b in boxes):
        targets.append(lf)

print(f"[INFO] target images(contain rare): {len(targets)} / total labeled: {len(img2boxes)}")

# -----------------------
# 3) augmentation pipeline (robustness용)
# -----------------------
aug = A.Compose(
    [
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
        ], p=0.30),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        ], p=0.30),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.40),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=12, p=0.35),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.10, rotate_limit=7, border_mode=cv2.BORDER_REFLECT_101, p=0.50),
        A.CoarseDropout(max_holes=6, max_height=40, max_width=40, fill_value=0, p=0.25),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.25),
)

# -----------------------
# 4) generate
# -----------------------
made = 0
skipped = 0

for lf in targets:
    img_name = os.path.splitext(lf)[0] + ".png"
    img_path = os.path.join(IMG_ALL, img_name)
    lab_path = os.path.join(LBL_ALL, lf)

    if not os.path.exists(img_path):
        skipped += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue

    boxes = read_yolo_label(lab_path)
    if not boxes:
        skipped += 1
        continue

    bboxes, clses = yolo_to_albu(boxes)

    for j in range(1, AUG_PER_IMAGE + 1):
        out_stem = f"{os.path.splitext(img_name)[0]}_aug{j}"
        out_img = os.path.join(IMG_AUG, out_stem + ".png")
        out_lbl = os.path.join(LBL_AUG, out_stem + ".txt")

        if os.path.exists(out_img) and os.path.exists(out_lbl):
            continue

        transformed = aug(image=img, bboxes=bboxes, class_labels=clses)
        img_t = transformed["image"]
        b_t = transformed["bboxes"]
        c_t = transformed["class_labels"]

        out_boxes = albu_to_yolo(b_t, c_t)
        if not out_boxes:
            continue

        cv2.imwrite(out_img, img_t)
        write_yolo_label(out_lbl, out_boxes)
        made += 1

print("[DONE] STEP 03-2 augmentation completed")
print(f" - made: {made}")
print(f" - skipped: {skipped}")
print(f" - aug dirs: {IMG_AUG}, {LBL_AUG}")