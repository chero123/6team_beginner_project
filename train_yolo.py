from ultralytics import YOLO
import torch

# CUDA 사용 여부 확인
if torch.cuda.is_available():
    device_str = 0
    print("🔥 NVIDIA CUDA GPU 사용!")
else:
    device_str = "cpu"
    print("⚠️ GPU 없음 → CPU 사용")
    

model = YOLO("yolov8n.pt")

model.train(
    data="data.yml",
    epochs=50,
    batch=16,
    imgsz=640,
    device=device_str,
    workers=0
)
