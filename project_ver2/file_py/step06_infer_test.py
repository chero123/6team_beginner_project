# file_py/step06_infer_test.py

import os
import json
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# =========================
# PATH
# =========================
MODEL = (
    "/home/ohs3201/6team_beginner_project/project_ver2/"
    "runs/detect/yolov8l_finetune_oversample3/weights/best.pt"
)

TEST_IMG_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
OUT_JSON = "/home/ohs3201/work/step6_preds.json"

# =========================
# CONFIG
# =========================
IMG_SIZE = 1280
CONF_THRES = 0.25
IOU_THRES = 0.6
MAX_DET = 10

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL)

# =========================
# IMAGE LIST
# =========================
img_files = sorted([
    f for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith(".png")
])

all_preds = []

# =========================
# INFERENCE
# =========================
for fname in tqdm(img_files):
    img_path = os.path.join(TEST_IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    image_id = int(os.path.splitext(fname)[0])  # 🔥 반드시 int

    results = model.predict(
        source=img,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        device=0,
        verbose=False
    )

    r = results[0]
    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy()

    for box, score, cls in zip(boxes, scores, clss):
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        all_preds.append({
            "image_id": image_id,                 # int
            "cls": int(cls),                      # int
            "score": float(score),                # float
            "bbox": [
                float(x1), float(y1),
                float(bw), float(bh)
            ]
        })

# =========================
# SAVE JSON
# =========================
with open(OUT_JSON, "w") as f:
    json.dump(all_preds, f, indent=2)

print(f"[DONE] Step06 inference saved to {OUT_JSON}")