from ultralytics import YOLO

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
DATA_YAML = f"{PROJECT_ROOT}/work/yolo/data.yaml"

model = YOLO("yolov8l.pt")

model.train(
    data=DATA_YAML,
    imgsz=896,
    epochs=120,
    batch=8,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=15,
    warmup_epochs=3,
    mosaic=0.5,
    close_mosaic=10,
    mixup=0.0,
    copy_paste=0.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    workers=8,
    name="ver4_yolov8l_baseline_896",
)