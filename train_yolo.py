from ultralytics import YOLO
import torch

# M1이면 mps, 아니면 cpu
if torch.backends.mps.is_available():
    device_str = "mps"
    print("🔥 Using Apple M1 GPU (MPS)")
else:
    device_str = "cpu"
    print("⚠️ MPS 사용 불가, CPU로 학습합니다.")

# 가장 가벼운 YOLOv8n 모델 사용
model = YOLO("yolov8n.pt")

model.train(
    data="data.yml",
    epochs=50,
    batch=16,
    imgsz=640,
    device=device_str,   # ✅ 여기!
)
