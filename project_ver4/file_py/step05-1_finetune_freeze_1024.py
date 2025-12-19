from ultralytics import YOLO

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
DATA_YAML = f"{PROJECT_ROOT}/work/yolo/data.yaml"
BASE_MODEL = f"{PROJECT_ROOT}/runs/detect/ver4_yolov8l_baseline_896/weights/best.pt"

model = YOLO(BASE_MODEL)

model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=40,
    batch=8,
    device=0,
    lr0=0.002,
    optimizer="SGD",
    freeze=10,
    mosaic=0.0,
    close_mosaic=0,
    patience=8,
    workers=8,
    name="ver4_finetune_1024_freeze",
)