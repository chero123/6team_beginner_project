# make_pseudo.py
from ultralytics import YOLO
import os

# 현재 파일 기준으로 프로젝트 루트 계산
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) 학습된 모델(best.pt) 경로
MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs", "detect", "train3", "weights", "best.pt"
)

# 2) 라벨 없는 이미지들이 모여 있는 폴더
UNLABELED_DIR = os.path.join(BASE_DIR, "unlabeled_images")

print("모델 경로 :", MODEL_PATH)
print("라벨 없는 이미지 폴더 :", UNLABELED_DIR)

# 모델 로드
model = YOLO(MODEL_PATH)

# 3) pseudo label 생성
results = model.predict(
    source=UNLABELED_DIR,          # 라벨 없는 이미지들
    project=os.path.join(BASE_DIR, "runs", "detect"),
    name="pseudo1",                # 결과 폴더 이름: runs/detect/pseudo1
    save_txt=True,                 # YOLO txt 라벨 저장
    save_conf=False,               # 학습용이므로 conf 컬럼은 저장 X (5컬럼 유지)
    conf=0.7,                      # ✨ 확신 높은 것만 pseudo label로 사용
    imgsz=640
)

print("\n✅ 완료! pseudo label txt 파일들은")
print(os.path.join(BASE_DIR, "runs", "detect", "pseudo1", "labels"))
print("여기에 생성됨.")
