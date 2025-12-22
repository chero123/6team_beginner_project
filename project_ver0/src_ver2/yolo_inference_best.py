import os
import csv
import json
from ultralytics import YOLO
from PIL import Image
import numpy as np

# PATH 설정
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

TEST_DIR = os.path.join(BASE_PROJECT, "data", "test_images")
OUTPUT_CSV = os.path.join(BASE_PROJECT, "results", "submission", "yolov8_best_single.csv")

# 너가 FULL TRAIN 또는 CONTINUE TRAIN한 YOLO weight
YOLO_WEIGHT = os.path.join(BASE_PROJECT, "results/full/yolov8l_continue/finetune6/weights/best.pt")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# category mapping 불러오기
with open(os.path.join(BASE_PROJECT, "category_mapping.json"), "r") as f:
    mapping = json.load(f)

# YOLO 내부 class index → 실제 category_id 변환
idx2catid = mapping.get("idx2catid")
if idx2catid is None:
    # 역매핑 자동 생성 (step05 방식과 동일)
    yolo2cat = {int(k): v for k, v in mapping["yolo2cat"].items()}
    idx2catid = {str(k): int(v) for k, v in yolo2cat.items()}

# YOLO 모델 로드
print("🚀 Loading YOLO model...")
model = YOLO(YOLO_WEIGHT)

# Inference 설정
CONF_TH = 0.05     # 너무 높으면 Recall 감소 → 0.01~0.05 추천
IOU_NMS = 0.6      # NMS 강화
TOPK = 5           # 이미지당 최대 5개 박스 제한 (노이즈 제거)

# CSV 생성 준비
rows = []
annotation_id = 1

test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])

print(f"🔍 테스트 이미지 {len(test_images)}개 예측 시작...\n")

for img_name in test_images:
    img_path = os.path.join(TEST_DIR, img_name)

    # image_id robust parsing
    try:
        image_id = int(os.path.splitext(img_name)[0])
    except:
        # 혹시 숫자+문자 섞인 경우 대비
        image_id = int(os.path.splitext(img_name)[0].split("_")[0])


    # YOLO inference

    preds = model.predict(
        source=img_path,
        conf=CONF_TH,
        iou=IOU_NMS,
        imgsz=1024,
        verbose=False
    )[0]

    boxes = preds.boxes

    # Score 기준으로 TOP-K 선택
    if len(boxes) > TOPK:
        scores = boxes.conf.cpu().numpy()
        top_idx = np.argsort(-scores)[:TOPK]
        boxes = boxes[top_idx]


    # 박스 처리

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        score = float(b.conf[0])
        cls_idx = int(b.cls[0])

        # 🔥 YOLO class index → 실제 category_id 변환
        category_id = int(idx2catid[str(cls_idx)])

        # xyxy → xywh
        w = x2 - x1
        h = y2 - y1

        rows.append([
            annotation_id,
            image_id,
            category_id,
            int(round(x1)),
            int(round(y1)),
            int(round(w)),
            int(round(h)),
            round(score, 6)
        ])
        annotation_id += 1

# CSV 저장
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    writer.writerows(rows)

print("\n🎉 YOLO 단독 inference 완료!")
print(f"📁 저장된 CSV: {OUTPUT_CSV}")
print(f"총 박스 수: {len(rows)}")