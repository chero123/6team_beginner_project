#저장된 학습 데이터를 클래스별로 시각화해서 보는 streamlit

import os
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

# -----------------------------
# 0. 기본 경로 설정
# -----------------------------
BASE_DIR = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

YOLO_ROOT = BASE_DIR / "yolo_dataset_aihub+orig(4img)"
IMG_DIRS = [
    YOLO_ROOT / "images" / "train",
    YOLO_ROOT / "images" / "val",
]
LABEL_DIRS = [
    YOLO_ROOT / "labels" / "train",
    YOLO_ROOT / "labels" / "val",
]

MODEL_PATH = BASE_DIR / r"runs\detect\train17\weights\best.pt"
IMG_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
EXAMPLES_PER_CLASS = 4

# -----------------------------
# 1. 모델 class 이름 불러오기
# -----------------------------
model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names  # dict: {0:'pill_0', ...} or list
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------
# Helper: 이미지 경로 찾기
# -----------------------------
def find_image_path(stem: str):
    for img_dir in IMG_DIRS:
        for ext in IMG_EXTS:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None

# -----------------------------
# Helper: YOLO → xyxy 변환
# -----------------------------
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    return x1, y1, x2, y2


# =====================================================================================
# 1) GT 카운트 계산
# 2) 클래스별 예시 이미지 추출
# =====================================================================================

# 클래스별 GT box 개수
class_count = {i: 0 for i in range(NUM_CLASSES)}

# 클래스별 예시 이미지 저장
class_examples = {i: [] for i in range(NUM_CLASSES)}

# 라벨 파일 수집
label_files = []
for lbl_dir in LABEL_DIRS:
    if lbl_dir.exists():
        label_files.extend(lbl_dir.glob("*.txt"))

# 라벨 파일 하나씩 처리
for lbl_path in label_files:
    stem = lbl_path.stem
    img_path = find_image_path(stem)

    # 이미지 로드
    img = None
    img_h = img_w = None
    if img_path is not None:
        img = cv2.imread(str(img_path))
        if img is not None:
            img_h, img_w = img.shape[:2]

    # 라벨 읽기
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])

        # GT count 증가
        class_count[cls_id] += 1

        # 이미 예시가 4개 있으면 스킵
        if len(class_examples[cls_id]) >= EXAMPLES_PER_CLASS:
            continue

        # 이미지 없으면 crop 불가
        if img is None:
            continue

        cx, cy, w, h = map(float, parts[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        class_examples[cls_id].append(crop_rgb)


# =====================================================================================
# Streamlit UI
# =====================================================================================
st.title("💊 YOLO Class Example Viewer (GT 기반 학습 데이터 시각화)")

# 전체 클래스 출력
for class_id in range(NUM_CLASSES):
    total_boxes = class_count[class_id]

    st.markdown(
        f"## 🏷 Class {class_id} — {CLASS_NAMES[class_id]} (**{total_boxes}개**)"
    )

    examples = class_examples[class_id]

    if len(examples) == 0:
        st.info("📭 예시 없음 (해당 클래스가 GT에 존재하지 않음)")
    else:
        st.image(examples, width=200)
