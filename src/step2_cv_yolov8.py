import os
import json
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 기본 경로
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

DATA_DIR = os.path.join(BASE_PROJECT, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_ANN_DIR = os.path.join(DATA_DIR, "train_annotations")

YOLO_BASE = os.path.join(BASE_PROJECT, "yolo_dataset")
RESULT_CV_PATH = os.path.join(BASE_PROJECT, "results", "cv")
os.makedirs(RESULT_CV_PATH, exist_ok=True)

# category mapping
with open(os.path.join(BASE_PROJECT, "category_mapping.json"), "r") as f:
    mapping = json.load(f)

sorted_cat_ids = mapping["sorted_cat_ids"]
catid2idx = mapping["catid2idx"]

# folds
with open(os.path.join(BASE_PROJECT, "folds_5.json"), "r") as f:
    folds = json.load(f)

# annotation 파일 매핑 캐시 (이미 Step1에서 스캔됨)
json_paths = {}
for root, dirs, files in os.walk(TRAIN_ANN_DIR):
    for f in files:
        if f.endswith(".json"):
            json_paths[f] = os.path.join(root, f)


# YOLO 라벨 파일 작성 함수
def write_yolo_label(img_name, dst_path):
    """img_name 기준으로 YOLO txt 생성"""

    # annotation JSON 찾기
    json_name = img_name.replace(".png", ".json")
    if json_name not in json_paths:
        return

    with open(json_paths[json_name], "r") as f:
        data = json.load(f)

    anns = data["annotations"]
    img_info = data["images"][0]

    img_path = os.path.join(TRAIN_IMG_DIR, img_name)
    w, h = Image.open(img_path).size

    lines = []
    for ann in anns:
        cid = ann["category_id"]
        if str(cid) not in catid2idx:
            continue

        cls_idx = catid2idx[str(cid)]
        x, y, bw, bh = ann["bbox"]

        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        bw_n = bw / w
        bh_n = bh / h

        lines.append(f"{cls_idx} {cx} {cy} {bw_n} {bh_n}")

    if lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(lines))


# Fold별 YOLO dataset 생성
def prepare_yolo_fold_dataset(fold_idx):
    fold = folds[fold_idx]
    train_imgs = fold["train"]
    val_imgs = fold["val"]

    # 기존 폴더 초기화
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(os.path.join(YOLO_BASE, sub), ignore_errors=True)
        os.makedirs(os.path.join(YOLO_BASE, sub), exist_ok=True)

    train_txt_list = []
    val_txt_list = []

    # Train 복사
    for img_name in tqdm(train_imgs, desc=f"Fold{fold_idx} Train 이미지 처리"):
        src = os.path.join(TRAIN_IMG_DIR, img_name)
        dst = os.path.join(YOLO_BASE, "images/train", img_name)
        shutil.copy2(src, dst)

        label_dst = os.path.join(YOLO_BASE, "labels/train", img_name.replace(".png", ".txt"))
        write_yolo_label(img_name, label_dst)

        train_txt_list.append(dst)

    # Val 복사
    for img_name in tqdm(val_imgs, desc=f"Fold{fold_idx} Val 이미지 처리"):
        src = os.path.join(TRAIN_IMG_DIR, img_name)
        dst = os.path.join(YOLO_BASE, "images/val", img_name)
        shutil.copy2(src, dst)

        label_dst = os.path.join(YOLO_BASE, "labels/val", img_name.replace(".png", ".txt"))
        write_yolo_label(img_name, label_dst)

        val_txt_list.append(dst)

    # train.txt, val.txt
    train_txt_path = os.path.join(YOLO_BASE, f"fold{fold_idx}_train.txt")
    val_txt_path = os.path.join(YOLO_BASE, f"fold{fold_idx}_val.txt")

    with open(train_txt_path, "w") as f:
        f.write("\n".join(train_txt_list))
    with open(val_txt_path, "w") as f:
        f.write("\n".join(val_txt_list))

    # YAML 생성
    names = [str(cid) for cid in sorted_cat_ids]
    dataset_yaml = {
        "path": YOLO_BASE,
        "train": train_txt_path,
        "val": val_txt_path,
        "names": {i: n for i, n in enumerate(names)},
        "nc": len(names)
    }

    yaml_path = os.path.join(YOLO_BASE, f"dataset_fold{fold_idx}.yaml")

    import yaml
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True)

    return yaml_path


# YOLOv8m 교차검증
def run_yolov8_cv(epochs=2):
    model_name = "yolov8m"
    pretrained = "yolov8m.pt"

    fold_scores = []

    for fold_idx in range(5):
        print(f"\n==============================")
        print(f"   Fold {fold_idx} 시작")
        print(f"==============================")

        yaml_path = prepare_yolo_fold_dataset(fold_idx)

        model = YOLO(pretrained)

        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            device=0,
            project=os.path.join(BASE_PROJECT, "runs_cv"),
            name=f"{model_name}_fold{fold_idx}",
            exist_ok=True,
            verbose=False
        )

        # mAP50 추출
        metrics = getattr(results, "results_dict", {})
        score = metrics.get("metrics/mAP50(B)", None) \
             or metrics.get("metrics/mAP50", None)

        if score is None:
            print("⚠ mAP50 키를 찾지 못함 → 0으로 처리")
            score = 0.0

        print(f"Fold {fold_idx} mAP50:", score)
        fold_scores.append(float(score))

    avg_score = float(np.mean(fold_scores))

    # 결과 저장
    out_path = os.path.join(RESULT_CV_PATH, "yolov8m.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": "yolov8m",
            "fold_scores": fold_scores,
            "avg_score": avg_score
        }, f, indent=2)

    print("\n🎉 YOLOv8m 5-Fold CV 완료!")
    print("📌 평균 mAP50:", avg_score)
    print("📁 저장 위치:", out_path)


# 실행부
if __name__ == "__main__":
    run_yolov8_cv(epochs=2)