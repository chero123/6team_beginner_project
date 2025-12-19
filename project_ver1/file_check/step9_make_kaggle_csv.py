import os
import json
import csv
import torch
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH     = "step8_yolov8_runs/yolov8l_baseline2/weights/best.pt"
TEST_IMG_DIR   = "/home/ohs3201/6team_beginner_project/data/test_images"
REMAPPING_JSON = "/home/ohs3201/work/step4_runs_remap/category_id_remap.json"
OUT_CSV        = "submission.csv"
CONF_THRES     = 0.5

# =========================
# LOAD REMAP (train_id -> dl_idx)
# =========================
remap = json.load(open(REMAPPING_JSON))
orig_to_train = {int(k): int(v) for k, v in remap["orig_to_train_id"].items()}
train_to_orig = {v: k for k, v in orig_to_train.items()}  # 🔥 핵심

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# CSV WRITE
# =========================
rows = []
ann_id = 1

test_imgs = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for fname in test_imgs:
    img_path = os.path.join(TEST_IMG_DIR, fname)
    img_id = int(os.path.splitext(fname)[0])  # 🔥 image_id 숫자만

    results = model(img_path, conf=CONF_THRES, verbose=False)

    if len(results) == 0:
        continue

    r = results[0]
    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, clss):
        train_id = cls + 1  # YOLO cls(0-based) → train_id(1-based)

        if train_id not in train_to_orig:
            continue  # 안전장치

        dl_idx = train_to_orig[train_id]

        w = x2 - x1
        h = y2 - y1

        rows.append([
            ann_id,
            img_id,
            dl_idx,
            float(x1),
            float(y1),
            float(w),
            float(h),
            float(score)
        ])
        ann_id += 1

# =========================
# SAVE CSV
# =========================
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score"
    ])
    writer.writerows(rows)

print(f"✅ Saved Kaggle submission: {OUT_CSV}")
print(f"Total predictions: {len(rows)}")