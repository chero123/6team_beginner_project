import os
import json
import csv
from ultralytics import YOLO
from tqdm import tqdm

# =========================
# PATH
# =========================
MODEL_PATH = (
    "/home/ohs3201/6team_beginner_project/project_ver2/"
    "runs/detect/yolov8l_finetune_lr2e-3/weights/best.pt"
)
TEST_IMG_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
MAPPING_JSON = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"

OUT_CSV = "/home/ohs3201/work/kaggle_submission_ver2.csv"

# =========================
# CONFIG
# =========================
CONF_THRES = 0.1
TOP_K = 4

# =========================
# LOAD MAPPING (cls -> dl_idx)
# =========================
raw_map = json.load(open(MAPPING_JSON, encoding="utf-8"))

cls_to_dlidx = {}
for dl_str, v in raw_map.items():
    dl = int(dl_str)
    if isinstance(v, dict) and "cls" in v:
        cls_to_dlidx[int(v["cls"])] = dl
    else:
        cls = int(str(v).split("cls")[-1].strip(" )"))
        cls_to_dlidx[cls] = dl

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# INFER
# =========================
results = model.predict(
    source=TEST_IMG_DIR,
    imgsz=1024,
    conf=CONF_THRES,
    iou=0.6,
    device=0,
    save=False,
    verbose=False
)

rows = []
ann_id = 1

for r in tqdm(results):
    img_name = os.path.basename(r.path)
    image_id = int(os.path.splitext(img_name)[0])  # 1.png -> 1

    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    preds = []
    for box, score, cls in zip(boxes, scores, clss):
        if cls not in cls_to_dlidx:
            continue

        if score < CONF_THRES:
            continue

        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        preds.append((score, cls, x1, y1, w, h))

    # 🔑 score 기준 상위 TOP_K만 사용
    preds.sort(key=lambda x: x[0], reverse=True)
    preds = preds[:TOP_K]

    for score, cls, x, y, w, h in preds:
        rows.append([
            ann_id,
            image_id,
            cls_to_dlidx[cls],   # dl_idx
            round(float(x), 2),
            round(float(y), 2),
            round(float(w), 2),
            round(float(h), 2),
            round(float(score), 4),
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
        "score",
    ])
    writer.writerows(rows)

print(f"[DONE] Kaggle CSV saved -> {OUT_CSV}")
print(f"Total rows: {len(rows)}")