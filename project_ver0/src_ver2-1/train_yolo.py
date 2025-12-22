from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data="/.../yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=1e-3,
    patience=20
)