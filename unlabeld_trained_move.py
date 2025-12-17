#라벨을 새로 만든 이미지들만 골라서 YOLO 학습 데이터셋(train)에 편입시키는 코드

import os
import shutil
from pathlib import Path

# ---------------------------------------
# 1) 경로 설정 (네 환경에 맞게 수정)
# ---------------------------------------
BASE = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

UNLABELED_IMG_DIR = BASE / "unlabeled_images"
SELF_LABEL_DIR = BASE / "self_labels"

YOLO_DATASET = BASE / "yolo_dataset"
YOLO_IMG_TRAIN = YOLO_DATASET / "images" / "train"
YOLO_LBL_TRAIN = YOLO_DATASET / "labels" / "train"

# 폴더 생성
YOLO_IMG_TRAIN.mkdir(parents=True, exist_ok=True)
YOLO_LBL_TRAIN.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2) self_labels 안 txt 파일 스캔 → stem 목록 만들기
# ---------------------------------------
txt_files = list(SELF_LABEL_DIR.glob("*.txt"))
label_stems = {txt.stem for txt in txt_files}

print(f"[INFO] self_labels TXT 개수: {len(txt_files)}")

# ---------------------------------------
# 3) unlabeled_images 안 이미지 스캔 후 매핑되는 것만 이동
# ---------------------------------------
img_exts = [".jpg", ".jpeg", ".png"]
moved_count = 0

for img_path in UNLABELED_IMG_DIR.iterdir():
    if img_path.suffix.lower() not in img_exts:
        continue

    stem = img_path.stem

    # self_labels에 같은 이름의 txt가 있을 때만 이동
    if stem in label_stems:
        txt_path = SELF_LABEL_DIR / f"{stem}.txt"

        # 이미지 이동
        shutil.move(str(img_path), YOLO_IMG_TRAIN / img_path.name)
        # 라벨 이동
        shutil.move(str(txt_path), YOLO_LBL_TRAIN / txt_path.name)

        moved_count += 1

print(f"[INFO] 매핑된 (이미지 + txt) 쌍 {moved_count}개 이동 완료!")
print(f"→ images/train: {YOLO_IMG_TRAIN}")
print(f"→ labels/train: {YOLO_LBL_TRAIN}")
