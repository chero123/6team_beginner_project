# file_py/step05-2_finetune_unfreeze.py
from ultralytics import YOLO

DATA_YAML = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo/data.yaml"
MODEL = "runs/detect/ver3_finetune_1024_freeze/weights/best.pt"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=30,
    batch=8,
    device=0,

    # 🔥 더 낮은 lr
    lr0=0.001,
    optimizer="SGD",

    mosaic=0.0,
    close_mosaic=0,

    patience=6,
    name="ver3_finetune_1024_unfreeze",
)