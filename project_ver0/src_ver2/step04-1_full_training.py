import os
import json
from ultralytics import YOLO
from fasterrcnn_full_train import train_fasterrcnn_full

BASE = "/home/ohs3201/6team_beginner_project"
CV_DIR = f"{BASE}/results/cv"
FULL_DIR = f"{BASE}/results/full_training"
YOLO_DATA = f"{BASE}/yolo_dataset/data.yaml"

os.makedirs(FULL_DIR, exist_ok=True)


# Best model info 로드
def load_best_info():
    info_path = f"{CV_DIR}/best_model_info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    return info["model_name"], info["weight"]


# YOLO 계열 Full Train
def train_yolo(model_name, init_weight, epochs, save_name):
    print(f"\n🚀 YOLO Full Training 시작 → {model_name}")

    model = YOLO(init_weight)

    model.train(
        data=YOLO_DATA,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        project=FULL_DIR,
        name=save_name,
        exist_ok=True,
        verbose=True
    )

    final_weight = f"{FULL_DIR}/{save_name}/weights/best.pt"
    return final_weight


# MAIN
def main():
    print("\n🔥 Step04: Full Training 시작 🔥")

    best_model, best_weight = load_best_info()
    print(f"⭐ 선택된 BEST MODEL: {best_model}")
    print(f"⭐ Pretrained weight: {best_weight}")

    final_paths = {}

    # 1) Best 모델 Full Train
    if best_model == "fasterrcnn":
        print("\n🚀 FasterRCNN Full Training 시작")
        out = train_fasterrcnn_full(
            pretrained_weight=best_weight,
            output_dir=f"{FULL_DIR}/fasterrcnn_full"
        )
        final_paths["fasterrcnn"] = out

    elif best_model in ["yolov8m", "rtdetr"]:
        epochs = 40
        out = train_yolo(
            model_name=best_model,
            init_weight=best_weight,
            epochs=epochs,
            save_name=f"{best_model}_full"
        )
        final_paths[best_model] = out


    # 2) YOLOv8l Always Full Train
    print("\n🚀 YOLOv8l Full Training 시작 (무조건 수행)")

    yolov8l_weight = "yolov8l.pt"
    yolov8l_out = train_yolo(
        model_name="yolov8l",
        init_weight=yolov8l_weight,
        epochs=40,
        save_name="yolov8l_full"
    )

    final_paths["yolov8l"] = yolov8l_out


    # 3) 결과 저장
    out_json = f"{FULL_DIR}/final_models.json"
    with open(out_json, "w") as f:
        json.dump(final_paths, f, indent=2)

    print(f"\n📁 Full Training 완료 → {out_json}")
    print("📌 다음 단계: step04-2_yolov8l_continue_train.py 실행")


if __name__ == "__main__":
    main()