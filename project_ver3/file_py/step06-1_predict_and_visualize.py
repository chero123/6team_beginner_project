# file_py/step06-1_predict_and_visualize.py
from ultralytics import YOLO
import os

MODEL_PATH = "runs/detect/ver3_finetune_1152_final/weights/best.pt"
TEST_IMG_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
OUT_VIS_DIR = "/home/ohs3201/6team_beginner_project/project_ver3/output/vis"

os.makedirs(OUT_VIS_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model.predict(
    source=TEST_IMG_DIR,
    imgsz=1152,
    conf=0.25,
    iou=0.7,
    augment=True,      # 🔥 TTA (flip)
    save=True,
    save_txt=False,
    project=OUT_VIS_DIR,
    name="pred",
    device=0
)

print("[DONE] Test prediction & visualization saved")