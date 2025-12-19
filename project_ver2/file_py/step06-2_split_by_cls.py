import os
import shutil
from ultralytics import YOLO

MODEL = (
    "/home/ohs3201/6team_beginner_project/project_ver2/"
    "runs/detect/yolov8l_finetune_oversample3/weights/best.pt"
)

TEST_IMG_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
OUT_ROOT = "/home/ohs3201/work/step6_vis_by_class"

CONF_THRES = 0.25
IMG_SIZE = 1280

os.makedirs(OUT_ROOT, exist_ok=True)
model = YOLO(MODEL)

for img_name in os.listdir(TEST_IMG_DIR):
    if not img_name.endswith(".png"):
        continue

    img_path = os.path.join(TEST_IMG_DIR, img_name)

    results = model.predict(
        source=img_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=0,
        verbose=False
    )

    r = results[0]
    if r.boxes is None:
        continue

    clss = r.boxes.cls.cpu().numpy().astype(int)

    for cls in set(clss):
        cls_dir = os.path.join(OUT_ROOT, f"cls_{cls:02d}")
        os.makedirs(cls_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(cls_dir, img_name))