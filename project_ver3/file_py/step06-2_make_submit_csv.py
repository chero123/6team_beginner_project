# file_py/step06-2_make_submit_csv.py
import os
import csv
from ultralytics import YOLO
import json

MODEL_PATH = "runs/detect/ver3_finetune_1152_final/weights/best.pt"
TEST_IMG_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
OUT_CSV = "/home/ohs3201/6team_beginner_project/project_ver3/output/submit.csv"

TRAINID2DLIDX = "/home/ohs3201/6team_beginner_project/project_ver3/mappings/trainid_to_dlidx.json"

# -----------------------
# load mapping
# -----------------------
with open(TRAINID2DLIDX, "r") as f:
    trainid_to_dlidx = {int(k): int(v) for k, v in json.load(f).items()}

model = YOLO(MODEL_PATH)

results = model.predict(
    source=TEST_IMG_DIR,
    imgsz=1152,
    conf=0.6,
    iou=0.7,
    augment=True,
    device=0
)

rows = []
annotation_id = 1

for r in results:
    fname = os.path.basename(r.path)
    image_id = os.path.splitext(fname)[0]  # 🔥 .png 제거

    boxes = r.boxes
    if boxes is None:
        continue

    # score 기준 상위 4개만
    scores = boxes.conf.cpu().tolist()
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]

    for i in order:
        cls = int(boxes.cls[i])
        dl_idx = trainid_to_dlidx[cls]

        x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
        w = x2 - x1
        h = y2 - y1

        rows.append([
            annotation_id,
            int(image_id),
            dl_idx,
            round(x1, 2),
            round(y1, 2),
            round(w, 2),
            round(h, 2),
            round(scores[i], 5)
        ])
        annotation_id += 1

# -----------------------
# sort rows
# -----------------------
rows.sort(key=lambda x: (x[1], -x[7]))  # image_id asc, score desc

# -----------------------
# write csv
# -----------------------
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

print(f"[DONE] submission csv saved -> {OUT_CSV}")