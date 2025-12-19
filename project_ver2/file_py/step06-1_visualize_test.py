# file_py/step06-1_visualize_test.py

import os
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
OUT_DIR = "/home/ohs3201/work/step6_vis"   # 🔥 기존 유지
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
IMG_SIZE = 1280          # 🔥 테스트 이미지 기준
CONF_THRES = 0.25        # 누락 확인용 (낮게)
IOU_THRES = 0.6
MAX_DET = 10             # 강제 4개 X, 우선 다 보고 판단

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

print(f"[INFO] Visualizing {len(img_files)} test images")

# =========================
# INFERENCE + DRAW
# =========================
for fname in tqdm(img_files):
    img_path = os.path.join(TEST_IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

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
    clss = r.boxes.cls.cpu().numpy().astype(int)

    for box, score, cls in zip(boxes, scores, clss):
        x1, y1, x2, y2 = map(int, box)

        # bbox
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # label
        label = f"cls {cls} | {score:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, img)

print(f"[DONE] Visualization saved to {OUT_DIR}")