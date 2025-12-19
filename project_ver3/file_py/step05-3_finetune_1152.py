# file_py/step05-3_finetune_1152.py
from ultralytics import YOLO

DATA_YAML = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo/data.yaml"
MODEL = "runs/detect/ver3_finetune_1024_unfreeze/weights/best.pt"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    imgsz=1152,
    epochs=12,          # 🔥 길게 하지 말 것
    batch=6,         
    device=0,

    lr0=0.0007,         # 🔥 낮게
    optimizer="SGD",

    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    copy_paste=0.0,

    patience=6,
    name="ver3_finetune_1152_final",
)