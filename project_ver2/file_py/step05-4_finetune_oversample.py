# file_py/step05-3_finetune_oversample.py
from ultralytics import YOLO

MODEL = (
    "/home/ohs3201/6team_beginner_project/project_ver2/"
    "runs/detect/yolov8l_e120/weights/best.pt"
)

DATA = "/home/ohs3201/work/step4_yolov8/data_oversample.yaml"

model = YOLO(MODEL)

model.train(
    data=DATA,
    imgsz=1024,
    epochs=50,
    batch=8,
    lr0=0.0015,
    mosaic=0.0,
    close_mosaic=0,
    optimizer="SGD",
    patience=10,
    device=0,
    name="yolov8l_finetune_oversample",
)