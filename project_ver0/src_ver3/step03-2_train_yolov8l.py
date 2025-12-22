import os
from ultralytics import YOLO

BASE = "/home/ohs3201/6team_beginner_project"

DATA_YAML = f"{BASE}/dataset_v3.yaml"   # 이 파일 안에 nc: 28, 경로: yolo_dataset_v3로 되어 있어야 함
RESULT_DIR = f"{BASE}/results_v3/yolov8l_v3"
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    print("\n========== STEP03-2: YOLOv8L Training (v3 dataset, 28 classes) ==========")
    print(f"- DATA YAML : {DATA_YAML}")
    print(f"- RESULT DIR: {RESULT_DIR}")

    model = YOLO("yolov8l.pt")  # COCO pretrained

    model.train(
        data=DATA_YAML,
        epochs=120,
        imgsz=800,
        batch=8,
        optimizer="AdamW",
        lr0=1e-3,
        patience=20,
        device=0,
        project=RESULT_DIR,
        name="train",
        exist_ok=True,
        deterministic=True,
        workers=4,
        verbose=True,
    )

    print("\n🎉 YOLOv8L v3 Training 완료")
    print(f"📁 결과 디렉토리: {RESULT_DIR}/train")


if __name__ == "__main__":
    main()