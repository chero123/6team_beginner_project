from ultralytics import YOLO
import torch

MODEL = (
    "/home/ohs3201/6team_beginner_project/project_ver2/"
    "runs/detect/yolov8l_finetune_oversample3/weights/best.pt"
)
DATA_YAML = "/home/ohs3201/work/step4_yolov8/data.yaml"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=30,          # 🔽 짧게
    batch=8,
    device=0,

    lr0=0.001,          # 🔽 매우 낮게
    mosaic=0.0,         # 🔥 안정화
    close_mosaic=0,
    patience=5,

    resume=False,
    name="yolov8l_final_touch",
    verbose=True
)