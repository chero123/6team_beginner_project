# file_py/step05_5_finetune_yolov8.py

from ultralytics import YOLO
import torch

# =========================
# PATH
# =========================
BEST_MODEL = (
    "/home/ohs3201/6team_beginner_project/project_ver2/"
    "runs/detect/yolov8l_e120/weights/best.pt"
)
DATA_YAML = "/home/ohs3201/work/step4_yolov8/data.yaml"

# =========================
# ENV CHECK
# =========================
print("[INFO] torch:", torch.__version__)
print("[INFO] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] gpu:", torch.cuda.get_device_name(0))

# =========================
# LOAD MODEL (best.pt)
# =========================
model = YOLO(BEST_MODEL)

# =========================
# FINETUNE CONFIG
# =========================
model.train(
    data=DATA_YAML,
    imgsz=1024,
    epochs=50,          # 🔥 추가 파인튜닝
    batch=8,
    device=0,

    # 🔥 핵심 파라미터
    lr0=0.002,          # 낮은 learning rate
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # 🔥 conf 안정화용
    mosaic=0.2,         # 👉 더 안정 원하면 0.0
    close_mosaic=5,
    mixup=0.0,
    copy_paste=0.0,

    # 학습 제어
    patience=15,
    pretrained=True,
    resume=False,       # ❗ resume 아님 (best.pt를 초기값으로 새 run)
    deterministic=True,

    # 로그
    name="yolov8l_finetune_lr2e-3",
    plots=True,
    verbose=True
)

print("[DONE] Finetuning completed")