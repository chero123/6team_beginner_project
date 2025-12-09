import os
import re
import json
import pandas as pd
from ultralytics import YOLO

# -----------------------------
# 0. 경로 및 기본 설정
# -----------------------------
ROOT = "/Users/apple/Downloads/프로젝트1/ai06-level1-project"
TEST_DIR = os.path.join(ROOT, "test_images")
MODEL_PATH = os.path.join(ROOT, "runs/detect/train3/weights/best.pt")
CAT_MAP_PATH = os.path.join(ROOT, "category_id_mapping.json")

CONF_THRES = 0.1  # 필요하면 0.05 ~ 0.3 사이에서 조정 가능

# -----------------------------
# 1. category_id 매핑 복원
#    YOLO cls(0~55) -> 원래 category_id
# -----------------------------
with open(CAT_MAP_PATH, "r", encoding="utf-8") as f:
    cat_raw = json.load(f)

# 예: {"1": "xxx", "3": "yyy", ...}
old_ids = sorted([int(k) for k in cat_raw.keys()])   # [1, 3, 11, 24, ...]
# YOLO 학습 때: old_id -> 0~N-1 로 매핑했었으니
# 지금은 반대로: cls_idx(0~N-1) -> old_id 로 되돌려줌
idx_to_old = {idx: old_id for idx, old_id in enumerate(old_ids)}

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
    # 예: "0001.png" -> 1, "image_12.png" -> 12
    m = re.findall(r"\d+", filename)
    if len(m) == 0:
        # 숫자가 없으면 그냥 0이나, 파일 인덱스를 쓸 수도 있음 (필요시 조정)
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
        # 이 이미지에 대한 예측이 없으면 row를 추가하지 않아도 됨
        # (대회 룰에서 "하나도 없는 경우"에 대한 별도 규칙이 없다면 보통 OK)
        continue

    for box in preds:
        cls_idx = int(box.cls[0].item())            # 0 ~ 55
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # YOLO는 xyxy (x_min, y_min, x_max, y_max)를 주니까
        bbox_x = x1
        bbox_y = y1
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # cls_idx -> 원래 category_id 로 복원
        if cls_idx in idx_to_old:
            category_id = idx_to_old[cls_idx]
        else:
            # 혹시 인덱스가 범위를 벗어나면 일단 0 같은 값으로
            category_id = 0

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

output_path = os.path.join(ROOT, "submission.csv")
df.to_csv(output_path, index=False)
print("🎉 submission.csv 생성 완료 →", output_path)
print("총 row 수:", len(df))
