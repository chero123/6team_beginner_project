import os
import random
from ultralytics import YOLO

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"

# 최종 모델 (원하면 1024 best로 바꿔도 됨)
MODEL = f"{PROJECT_ROOT}/runs/detect/ver4_finetune_1152_final/weights/best.pt"

TEST_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "test_vis")
os.makedirs(OUT_DIR, exist_ok=True)

CONF = 0.25
IMGSZ = 1152
SAMPLE_N = 50

imgs = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".png")]
random.shuffle(imgs)
imgs = imgs[:min(SAMPLE_N, len(imgs))]
paths = [os.path.join(TEST_DIR, f) for f in imgs]

model = YOLO(MODEL)

# save=True면 자동으로 예측 이미지 저장됨
model.predict(
    source=paths,
    imgsz=IMGSZ,
    conf=CONF,
    iou=0.7,
    device=0,
    save=True,
    project=OUT_DIR,
    name="pred",
)
print("[DONE] test visualization saved to:", OUT_DIR)