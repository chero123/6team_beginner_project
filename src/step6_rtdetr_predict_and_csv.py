import os
import csv
import json
from glob import glob
from ultralytics import YOLO
from PIL import Image

# 경로 설정
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

TEST_DIR = os.path.join(BASE_PROJECT, "data", "test_images")
OUTPUT_CSV = os.path.join(BASE_PROJECT, "results", "submission", "rtdetr_submission.csv")

RTDETR_WEIGHT = os.path.join(BASE_PROJECT, "runs_full", "rtdetr_full", "weights", "best.pt")

# category mapping 불러오기
with open(os.path.join(BASE_PROJECT, "category_mapping.json")) as f:
    mapping = json.load(f)

idx2catid = mapping["idx2catid"]  # YOLO/RTDETR 내부 idx → original category_id 매핑

# 모델 로드
print("Loading RT-DETR model...")
model = YOLO(RTDETR_WEIGHT)


# 예측 후 CSV 생성
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

annotation_id = 1
rows = []

test_images = sorted(glob(os.path.join(TEST_DIR, "*.png")))

print("Running inference...")

for img_path in test_images:
    img_name = os.path.basename(img_path)
    image_id = img_name.replace(".png", "")

    results = model.predict(img_path, conf=0.25, verbose=False)

    for r in results:
        boxes = r.boxes

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            score = float(b.conf[0])
            cls_idx = int(b.cls[0])                      # YOLO/RTDETR class index
            category_id = int(idx2catid[str(cls_idx)])   # original category_id

            # convert xyxy → xywh
            w = x2 - x1
            h = y2 - y1

            rows.append([
                annotation_id,
                image_id,
                category_id,
                int(x1), int(y1), int(w), int(h),
                round(score, 6)
            ])
            annotation_id += 1

# CSV 저장
print(f"Saving CSV → {OUTPUT_CSV}")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "annotation_id","image_id","category_id",
        "bbox_x","bbox_y","bbox_w","bbox_h","score"
    ])
    writer.writerows(rows)

print(f"🎉 RT-DETR 제출 CSV 저장 완료!\n📁 {OUTPUT_CSV}")