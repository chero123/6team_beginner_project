from ultralytics import YOLO
import os

# ===== 경로 설정 =====
PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver1"

WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "step8_yolov8_runs/yolov8l_baseline2/weights/best.pt"
)

VAL_IMG_DIR = "/home/ohs3201/work/step7_yolov8/images/val"
OUT_DIR = os.path.join(PROJECT_ROOT, "step8_yolov8_runs/vis")

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 모델 로드 =====
model = YOLO(WEIGHTS)

# ===== 추론 & 저장 =====
model.predict(
    source=VAL_IMG_DIR,
    conf=0.25,
    iou=0.7,
    save=True,
    project=OUT_DIR,
    name="val_vis",
    exist_ok=True
)

print("✅ STEP 8.5 visualization done")