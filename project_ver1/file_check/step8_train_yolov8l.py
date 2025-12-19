from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="/home/ohs3201/6team_beginner_project/project_ver1/file_check/step7_yolov8/data.yaml",
    imgsz=1024,
    epochs=100,
    batch=8,
    device=0,
    patience=15,
    project="step8_yolov8_runs",
    name="yolov8l_baseline",
)