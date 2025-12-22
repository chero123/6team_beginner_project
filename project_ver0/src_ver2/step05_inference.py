import os
import json
from typing import List

from ultralytics import YOLO
from fasterrcnn_full_infer import infer_fasterrcnn

BASE = "/home/ohs3201/6team_beginner_project"
TEST_DIR = f"{BASE}/data/test_images"
MAP_PATH = f"{BASE}/category_mapping.json"

# 📁 저장 폴더: results/submission/ver2
SUBMIT_DIR = f"{BASE}/results/submission/ver2"
os.makedirs(SUBMIT_DIR, exist_ok=True)

# 📌 weight paths
FRCNN_WEIGHT = f"{BASE}/results/full_training/fasterrcnn_full/best.pth"
YOLO_CONTINUE_WEIGHT = f"{BASE}/results/full/yolov8l_continue/finetune6/weights/best.pt"

# 📌 output CSV paths
CSV_FRCNN = f"{SUBMIT_DIR}/FasterRCNN_ver2.csv"
CSV_YOLO  = f"{SUBMIT_DIR}/YOLOv8L_continue_ver2.csv"

# ---------- category_mapping.json 로드 ----------
with open(MAP_PATH, "r") as f:
    mp = json.load(f)

# YOLO class index → original category_id
yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}

# YOLO Inference 파라미터
YOLO_CONF = 0.05   # 모델 자체 conf threshold (낮게 두고 후처리에서 다시 필터)
YOLO_IOU  = 0.55   # NMS IoU
YOLO_MAX_DET = 15  # 이미지당 최대 박스 수 (YOLO 내부)
MIN_BOX = 5        # 초소형 박스 제거 기준
TOPK_PER_IMAGE = 5  # 최종 이미지당 상위 K개만 사용


def run_yolo_inference():
    print("\n🚀 YOLOv8L_continue inference 시작")
    if not os.path.exists(YOLO_CONTINUE_WEIGHT):
        raise FileNotFoundError(f"❌ YOLOv8L weight 없음: {YOLO_CONTINUE_WEIGHT}")

    model = YOLO(YOLO_CONTINUE_WEIGHT)

    preds = model.predict(
        source=TEST_DIR,
        imgsz=800,  
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        max_det=YOLO_MAX_DET,
        save=False,
        verbose=False,
        device=0,
    )

    rows: List[list] = []

    for img_pred in preds:
        img_name = os.path.basename(img_pred.path)
        img_id = int(os.path.splitext(img_name)[0])

        W, H = img_pred.orig_shape[1], img_pred.orig_shape[0]

        # 각 이미지 내부에서 score desc 정렬 후 TOPK_PER_IMAGE만 사용
        boxes_list = []
        for b in img_pred.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1

            # 너무 작은 박스 제거
            if w < MIN_BOX or h < MIN_BOX:
                continue

            if cls not in yolo2cat:
                continue

            category_id = yolo2cat[cls]

            boxes_list.append([
                conf,
                img_id,
                int(category_id),
                x1, y1, w, h,
            ])

        if not boxes_list:
            continue

        # score 내림차순 정렬 후 상위 TOPK_PER_IMAGE만 사용
        boxes_list.sort(key=lambda x: x[0], reverse=True)
        boxes_list = boxes_list[:TOPK_PER_IMAGE]

        # 정리해서 rows에 추가
        for conf, img_id, category_id, x1, y1, w, h in boxes_list:
            # 좌표 정리 (int + clamp)
            x1 = max(0, min(W - 1, int(round(x1))))
            y1 = max(0, min(H - 1, int(round(y1))))
            w = max(1, min(W - x1, int(round(w))))
            h = max(1, min(H - y1, int(round(h))))

            rows.append([
                img_id,
                category_id,
                x1, y1, w, h,
                conf,
            ])

    # CSV 저장
    with open(CSV_YOLO, "w") as f:
        f.write("image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"💾 YOLOv8L_continue CSV 저장 완료 → {CSV_YOLO}")


def main():
    print("\n==============================")
    print(" STEP05 : Inference Start (ver2)")
    print("==============================")

    # 1) 🔥 FasterRCNN Inference
    print(f"\n🚀 FasterRCNN inference 시작 → {CSV_FRCNN}")
    infer_fasterrcnn(
        weight_path=FRCNN_WEIGHT,
        csv_path=CSV_FRCNN,
        test_dir=TEST_DIR,
        num_classes=56,
    )

    # 2) 🔥 YOLOv8L Continue Inference
    run_yolo_inference()

    print("\n🎉 STEP05 ver2 완료!")


if __name__ == "__main__":
    main()