import os
import json
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"
DATA_DIR = f"{BASE}/data/test_images"
YOLO_WEIGHT = f"{BASE}/results_v3/yolov8l_v3/train/weights/best.pt"
CATEGORY_MAP = f"{BASE}/category_mapping.json"

# Load mapping
with open(CATEGORY_MAP, "r") as f:
    mp = json.load(f)

yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}

# Load YOLO model
print("🔄 YOLO 모델 로드 중...")
model = YOLO(YOLO_WEIGHT)

submit_rows = []
test_imgs = sorted(os.listdir(DATA_DIR))
print(f"📸 총 테스트 이미지: {len(test_imgs)}")

ann_id = 1

for img_name in tqdm(test_imgs):
    image_id = int(img_name.replace(".jpg", "").replace(".png", ""))
    img_path = os.path.join(DATA_DIR, img_name)

    results = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)

    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0].item())  # 0~27
            score = float(b.conf[0].item())

            cat = yolo2cat.get(cls, -1)  # convert to original category_id

            submit_rows.append([
                ann_id, image_id, cat,
                x1, y1, x2 - x1, y2 - y1, score
            ])
            ann_id += 1

# Save CSV
df = pd.DataFrame(submit_rows, columns=[
    "annotation_id", "image_id", "category_id",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
])

SAVE_PATH = f"{BASE}/results/submission/ver3/yolo_submit_v3.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
df.to_csv(SAVE_PATH, index=False)

print(f"🎉 YOLO 제출 CSV 저장 완료 → {SAVE_PATH}")