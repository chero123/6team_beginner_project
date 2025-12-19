from ultralytics import YOLO
import torch

# =========================
# ENV CHECK
# =========================
print("[INFO] torch:", torch.__version__)
print("[INFO] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] gpu:", torch.cuda.get_device_name(0))

# =========================
# PATH
# =========================
DATA_YAML = "/home/ohs3201/work/step4_yolov8/data.yaml"

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8l.pt")

# =========================
# TRAIN
# =========================
model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=120,
    batch=8,
    device=0,          # GPU 0
    workers=8,         # WSL에서 안정적인 값
    patience=20,
    optimizer="SGD",   # 기본값 (안 바꿈)
    lr0=0.01,          # 기본값
    weight_decay=5e-4, # 기본값
    name="yolov8l_e120",
    verbose=True
)