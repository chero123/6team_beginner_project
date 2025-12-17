from ultralytics import YOLO
 
model = YOLO(r"runs\detect\train17\weights\best.pt")

model.train(
    data="data.yml",
    epochs=20,
    imgsz=640,
    batch=16,
    lr0=0.0005,
    patience=10,
    device=0,   # GPU면 0, 안되면 "cpu"
    workers=0,
    name="train16_add_minor50 ",
)
