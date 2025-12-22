from ultralytics import YOLO
import os

BASE = "/home/ohs3201/6team_beginner_project"

PRETRAINED = f"{BASE}/results/full_training/yolov8l_full/weights/best.pt"
DATASET = f"{BASE}/yolo_dataset/data.yaml"
OUTPUT_DIR = f"{BASE}/results/full/yolov8l_continue"

LR = 1e-4
EPOCHS = 120

print("🔄 Loading YOLOv8-L pretrained weight...")
model = YOLO(PRETRAINED)

print(f"🚀 Starting YOLOv8-L continue training for {EPOCHS} epochs")
print(f"📌 Learning Rate = {LR}")

model.train(
    data=DATASET,
    imgsz=800,
    epochs=EPOCHS,
    lr0=LR,
    patience=30,
    batch=12,
    device=0,
    project=OUTPUT_DIR,
    name="finetune",
    deterministic=True,
)

print("🎉 YOLOv8-L continue training completed!")
print(f"📁 Results saved in: {OUTPUT_DIR}/finetune")