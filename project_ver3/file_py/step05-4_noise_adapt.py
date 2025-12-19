from ultralytics import YOLO

DATA_YAML = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo/data.yaml"
MODEL = "runs/detect/ver3_finetune_1152_final/weights/best.pt"

model = YOLO(MODEL)

model.train(
    data=DATA_YAML,

    imgsz=1152,
    epochs=15,          # 🔥 짧게 (과적합 방지)
    batch=6,
    device=0,

    optimizer="SGD",
    lr0=0.0005,         # 🔥 매우 낮게
    momentum=0.937,
    weight_decay=0.0005,

    patience=5,

    # ❌ 구조 증강 OFF
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    copy_paste=0.0,

    # ✅ YOLOv8 공식 지원 "현실 대응" 증강
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.4,

    erasing=0.6,        # 부분 가림
    translate=0.1,      # 위치 흔들림
    scale=0.5,          # 크기 변화
    degrees=5.0,        # 미세 회전
    perspective=0.0005,

    fliplr=0.5,

    box=7.5,
    cls=0.5,
    dfl=1.5,

    workers=8,
    name="ver3_finetune_1152_noise"
)