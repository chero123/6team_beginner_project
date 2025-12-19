# file_py/step05-1_finetune_freeze.py
from ultralytics import YOLO

DATA_YAML = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo/data.yaml"
BASE_MODEL = "runs/detect/ver3_yolov8l_baseline/weights/best.pt"

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=40,
    batch=8,
    device=0,

    # 🔥 핵심
    lr0=0.002,
    optimizer="SGD",
    freeze=10,          # backbone 대부분 freeze
    mosaic=0.0,
    close_mosaic=0,

    patience=8,
    name="ver3_finetune_1024_freeze",
)