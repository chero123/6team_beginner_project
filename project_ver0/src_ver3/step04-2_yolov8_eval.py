# step04-2_yolov8_eval.py
import os
from ultralytics import YOLO

BASE = "/home/ohs3201/6team_beginner_project"
YOLO_MODEL = f"{BASE}/results_v3/yolov8l_v3/train/weights/best.pt"
DATASET_YAML = f"{BASE}/dataset_v3.yaml"

def main():
    print("\n========== STEP04-2: YOLOv8 Validation ==========")
    print(f"📌 Load model: {YOLO_MODEL}")

    model = YOLO(YOLO_MODEL)

    results = model.val(
        data=DATASET_YAML,
        imgsz=800,
        batch=8,
        save_json=True,  # COCO metrics용
        save_hybrid=False
    )

    print("\n📌 YOLO Valid metrics:")
    print(results)

if __name__ == "__main__":
    main()