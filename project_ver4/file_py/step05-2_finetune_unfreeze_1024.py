from ultralytics import YOLO

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
DATA_YAML = f"{PROJECT_ROOT}/work/yolo/data.yaml"
MODEL = f"{PROJECT_ROOT}/runs/detect/ver4_finetune_1024_freeze/weights/best.pt"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=30,
    batch=8,
    device=0,
    lr0=0.001,
    optimizer="SGD",
    mosaic=0.0,
    close_mosaic=0,
    patience=6,
    workers=8,
    name="ver4_finetune_1024_unfreeze",
)