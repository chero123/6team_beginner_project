#모델이 예측한 test 데이터를 kaggle 형식에 맞게 변환하여 csv 생성

import os
import re
import json
import pandas as pd
from ultralytics import YOLO

# -----------------------------
# 0. 경로 및 기본 설정
# -----------------------------
ROOT = r"C:\Users\sangj\workspace\6team_beginner_project"
TEST_DIR = os.path.join(ROOT, "data_ai06", "test_images")
MODEL_PATH = os.path.join(ROOT, "runs", "detect", "train17", "weights", "best.pt")
CAT_MAP_PATH = os.path.join(ROOT, "category_id_mapping.json")

CONF_THRES = 0.05  # 필요하면 0.05 ~ 0.3 사이에서 조정 가능

# -----------------------------
# 1. category_id 매핑 복원
#    value 안에 "... (cls N)" 형태로 들어있다고 가정
# -----------------------------
with open(CAT_MAP_PATH, "r", encoding="utf-8") as f:
    cat_raw = json.load(f)

idx_to_old = {}  # YOLO cls_idx -> 원래 category_id

for k, v in cat_raw.items():
    cat_id = int(k)  # "1899" -> 1899
    m = re.search(r"cls\s*(\d+)", v)
    if not m:
        continue  # cls 정보 없으면 건너뜀 (원하면 여기서 에러 내도 됨)
    cls_idx = int(m.group(1))
    idx_to_old[cls_idx] = cat_id

# -----------------------------
# 2. YOLO 모델 로드
# -----------------------------
model = YOLO(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

# -----------------------------
# 3. 테스트 이미지 목록 가져오기
# -----------------------------
image_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"✅ Found {len(image_files)} test images.")

# -----------------------------
# 4. 예측 & submission row 생성
# -----------------------------
rows = []
annotation_counter = 1  # annotation_id 1부터 시작

for filename in image_files:
    img_path = os.path.join(TEST_DIR, filename)

    # 파일명에서 숫자만 뽑아서 image_id로 사용
    m = re.findall(r"\d+", filename)
    if len(m) == 0:
        image_id = 0
    else:
        image_id = int(m[-1])  # 마지막 숫자 덩어리 사용

    results = model.predict(
        source=img_path,
        imgsz=640,
        conf=CONF_THRES,
        verbose=False
    )

    preds = results[0].boxes

    if preds is None or len(preds) == 0:
        continue

    for box in preds:
        cls_idx = int(box.cls[0].item())            # YOLO cls index
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        bbox_x = x1
        bbox_y = y1
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # cls_idx -> 원래 category_id 로 복원
        if cls_idx not in idx_to_old:
            # 매핑에 없는 클래스면 Kaggle 채점에 안 쓰이는 거니까 스킵
            continue

        category_id = idx_to_old[cls_idx]

        rows.append({
            "annotation_id": annotation_counter,
            "image_id": image_id,
            "category_id": category_id,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "score": score,
        })

        annotation_counter += 1

# -----------------------------
# 5. DataFrame -> CSV 저장
# -----------------------------
df = pd.DataFrame(rows, columns=[
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
])

output_path = os.path.join(ROOT, "submission17.csv")
df.to_csv(output_path, index=False)
print("🎉 submission17.csv 생성 완료 →", output_path)
print("총 row 수:", len(df))
