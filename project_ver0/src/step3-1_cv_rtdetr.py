import os
import json
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# 기본 경로 설정
HOME = os.path.expanduser("~")
BASE_PROJECT = os.path.join(HOME, "6team_beginner_project")

DATA_DIR = os.path.join(BASE_PROJECT, "data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_ANN_DIR = os.path.join(DATA_DIR, "train_annotations")

YOLO_BASE = os.path.join(BASE_PROJECT, "yolo_dataset")
RESULT_CV_PATH = os.path.join(BASE_PROJECT, "results", "cv")
os.makedirs(RESULT_CV_PATH, exist_ok=True)

# category & folds 불러오기
with open(os.path.join(BASE_PROJECT, "category_mapping.json")) as f:
    mapping = json.load(f)
sorted_cat_ids = mapping["sorted_cat_ids"]
catid2idx = mapping["catid2idx"]

with open(os.path.join(BASE_PROJECT, "folds_5.json")) as f:
    folds = json.load(f)

# annotation 위치 캐시
json_paths = {}
for root, dirs, files in os.walk(TRAIN_ANN_DIR):
    for f in files:
        if f.endswith(".json"):
            json_paths[f] = os.path.join(root, f)

# YOLO 형식 라벨 생성 함수
def write_yolo_label(img_name, dst_path):
    json_name = img_name.replace(".png", ".json")
    if json_name not in json_paths:
        return

    with open(json_paths[json_name], "r") as f:
        data = json.load(f)
    anns = data["annotations"]

    w, h = Image.open(os.path.join(TRAIN_IMG_DIR, img_name)).size

    lines = []
    for ann in anns:
        cid = ann["category_id"]
        x, y, bw, bh = ann["bbox"]

        cls_idx = catid2idx[str(cid)]
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        bw_n = bw / w
        bh_n = bh / h

        lines.append(f"{cls_idx} {cx} {cy} {bw_n} {bh_n}")

    if lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(lines))

# Fold dataset 준비
def prepare_dataset_for_fold(fold_idx):
    fold = folds[fold_idx]
    train_imgs = fold["train"]
    val_imgs = fold["val"]

    # 기존 YOLO 디렉토리 클린업
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(os.path.join(YOLO_BASE, sub), ignore_errors=True)
        os.makedirs(os.path.join(YOLO_BASE, sub), exist_ok=True)

    train_txt_list, val_txt_list = [], []

    # Train 이미지 복사 & 라벨 생성
    for img_name in tqdm(train_imgs, desc=f"[Fold {fold_idx}] Train"):
        src = os.path.join(TRAIN_IMG_DIR, img_name)
        dst = os.path.join(YOLO_BASE, "images/train", img_name)
        shutil.copy2(src, dst)

        label_dst = os.path.join(YOLO_BASE, "labels/train", img_name.replace(".png", ".txt"))
        write_yolo_label(img_name, label_dst)

        train_txt_list.append(dst)

    # Val 이미지 복사 & 라벨 생성
    for img_name in tqdm(val_imgs, desc=f"[Fold {fold_idx}] Val"):
        src = os.path.join(TRAIN_IMG_DIR, img_name)
        dst = os.path.join(YOLO_BASE, "images/val", img_name)
        shutil.copy2(src, dst)

        label_dst = os.path.join(YOLO_BASE, "labels/val", img_name.replace(".png", ".txt"))
        write_yolo_label(img_name, label_dst)

        val_txt_list.append(dst)

    # Train, Val 목록 txt 저장
    train_txt_path = os.path.join(YOLO_BASE, f"fold{fold_idx}_train.txt")
    val_txt_path = os.path.join(YOLO_BASE, f"fold{fold_idx}_val.txt")

    with open(train_txt_path, "w") as f:
        f.write("\n".join(train_txt_list))
    with open(val_txt_path, "w") as f:
        f.write("\n".join(val_txt_list))

    # YAML 생성
    import yaml
    names = [str(cid) for cid in sorted_cat_ids]
    dataset_yaml = {
        "path": YOLO_BASE,
        "train": train_txt_path,
        "val": val_txt_path,
        "names": {i: n for i, n in enumerate(names)},
        "nc": len(names),
    }

    yaml_path = os.path.join(YOLO_BASE, f"dataset_fold{fold_idx}.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, allow_unicode=True)

    return yaml_path

# RT-DETR 5-Fold CV
def run_rtdetr_cv(epochs=2):
    model_name = "rtdetr-l"
    pretrained_path = os.path.join(BASE_PROJECT, "rtdetr-l.pt")

    fold_scores = []

    for fold_idx in range(5):
        print("\n====================================")
        print(f"   RT-DETR Fold {fold_idx} 시작")
        print("====================================")

        yaml_path = prepare_dataset_for_fold(fold_idx)

        model = YOLO(pretrained_path)

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

        # mAP50 가져오기 (최신 버전 대응)
        metrics = getattr(results, "results_dict", {})

        score = (
            metrics.get("metrics/mAP50(B)")
            or metrics.get("metrics/mAP50")
            or metrics.get("mAP50")
            or 0.0
        )

        print(f"Fold {fold_idx} mAP50 = {score}")
        fold_scores.append(score)

    avg_score = float(np.mean(fold_scores))

    out_path = os.path.join(RESULT_CV_PATH, "rtdetr-l.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "fold_scores": fold_scores,
            "avg_score": avg_score,
        }, f, indent=2)

    print("\n🎉 RT-DETR 5-Fold CV 완료!")
    print("📌 평균 mAP50:", avg_score)
    print("📁 저장 위치:", out_path)


if __name__ == "__main__":
    run_rtdetr_cv(epochs=2)