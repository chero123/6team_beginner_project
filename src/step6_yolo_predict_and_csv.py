import os
import csv
import json
from ultralytics import YOLO
from PIL import Image

# 기본 경로 설정
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

TEST_DIR = os.path.join(BASE_PROJECT, "data", "test_images")
WEIGHT_PATH = os.path.join(BASE_PROJECT, "runs_full", "yolov8m_full", "weights", "best.pt")
OUTPUT_CSV = os.path.join(BASE_PROJECT, "submission_yolo.csv")

# category_id 매핑 로드 (가장 중요!)
mapping_path = os.path.join(BASE_PROJECT, "category_mapping.json")
with open(mapping_path, "r") as f:
    mapping = json.load(f)

sorted_cat_ids = mapping["sorted_cat_ids"]  # YOLO 내부 0~N-1 → 실제 category_id

# 모델 로드
print("🚀 Loading YOLOv8 model...")
model = YOLO(WEIGHT_PATH)

# 테스트 이미지 리스트
test_images = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]
test_images.sort()

# CSV 파일 생성
header = [
    "annotation_id", "image_id", "category_id",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
]

annotation_id = 1
rows = []

print("🔍 Running inference on test images...\n")

for img_name in test_images:
    img_path = os.path.join(TEST_DIR, img_name)

    # image_id 추출 (파일이름 앞 숫자)
    image_id = int(img_name.replace(".png", "").split("_")[0])

    results = model(img_path, conf=0.1)[0]  # inference 결과

    for box in results.boxes:
        # YOLO → xyxy
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # bbox 변환
        bbox_x = int(x1)
        bbox_y = int(y1)
        bbox_w = int(x2 - x1)
        bbox_h = int(y2 - y1)

        # 🔥 내부 class index → 실제 category_id 변환
        cls_idx = int(box.cls[0].item())
        category_id = int(sorted_cat_ids[cls_idx])

        score = float(box.conf[0].item())

        rows.append([
            annotation_id,
            image_id,
            category_id,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            round(score, 5)
        ])

        annotation_id += 1

# CSV 저장
print(f"\n💾 Saving submission CSV to: {OUTPUT_CSV}")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("\n🎉 CSV 생성 완료!")
print(f"총 박스 수: {len(rows)}")
print(f"📁 제출 파일: {OUTPUT_CSV}")