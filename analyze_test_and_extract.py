import os
from pathlib import Path
from collections import Counter
import shutil

import cv2
from ultralytics import YOLO


# -----------------------------
# 0. 경로 및 설정
# -----------------------------
BASE_DIR = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

MODEL_PATH = BASE_DIR / r"runs\detect\train5\weights\best.pt"  # 네 모델 경로 맞게 수정
TEST_DIR = BASE_DIR / r"data_ai06\test_images"                      # 테스트 이미지 폴더
OUT_DIR_NO_DET = BASE_DIR / "test_no_detections"                    # 박스 하나도 없는 이미지 복사 위치

OUT_DIR_NO_DET.mkdir(exist_ok=True)

CONF_THRES = 0.1  # 필요하면 조절


# -----------------------------
# 1. 모델 로드
# -----------------------------
print(f"[INFO] 모델 로드 중: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("[INFO] model.names:", model.names)


# -----------------------------
# 2. 테스트 이미지 목록 수집
# -----------------------------
test_imgs = []


test_imgs = sorted(TEST_DIR.glob("*.png"))
print(f"[INFO] 테스트 이미지 개수: {len(test_imgs)}")


# -----------------------------
# 3. 테스트 전체 예측 및 클래스/박스 통계
# -----------------------------
class_counter = Counter()
num_images_no_det = 0

for idx, img_path in enumerate(test_imgs, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 이미지 로드 실패: {img_path}")
        continue

    # YOLO 예측
    results = model.predict(img, conf=CONF_THRES, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        # 박스가 하나도 없는 이미지 → 따로 저장
        num_images_no_det += 1
        dst_path = OUT_DIR_NO_DET / img_path.name
        shutil.copy2(img_path, dst_path)
        print(f"[NO DET] {img_path.name} -> {dst_path}")
        continue

    # 박스가 있으면 클래스 카운트
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    class_counter.update(cls_ids)

    if idx % 20 == 0 or idx == len(test_imgs):
        print(f"[PROGRESS] {idx}/{len(test_imgs)} 처리 중...")


# -----------------------------
# 4. 결과 출력
# -----------------------------
print("\n=== 테스트 데이터 클래스 분포 (모델 예측 기준) ===")
for cls_id, count in sorted(class_counter.items()):
    name = model.names.get(cls_id, f"class_{cls_id}")
    print(f"  class {cls_id} ({name}): {count} boxes")

print(f"\n[INFO] 박스 0개(탐지 실패) 이미지 수: {num_images_no_det}")
print(f"[INFO] 탐지 실패 이미지는 여기 복사됨: {OUT_DIR_NO_DET}")
