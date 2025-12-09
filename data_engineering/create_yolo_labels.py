"""
JSON 어노테이션을 YOLO 형식 라벨 파일로 변환하는 스크립트
"""

import os
import json
import glob
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_bbox_to_yolo(x, y, w, h, img_w, img_h):
    """COCO 형식 [x, y, w, h]를 YOLO 형식 [cx, cy, w, h] (정규화)로 변환"""
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return cx, cy, w_norm, h_norm


def create_yolo_labels(base_dir, train_ann_dir, yolo_dir):
    """
    JSON 어노테이션을 YOLO 라벨 파일로 변환
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
        train_ann_dir: JSON 어노테이션 디렉토리
        yolo_dir: YOLO 데이터셋 디렉토리
    """
    # Category mapping 로드
    category_mapping_path = os.path.join(base_dir, "category_mapping.json")
    with open(category_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    cat2idx = {int(k): int(v) for k, v in mapping["cat2idx"].items()}
    
    # 디렉토리 생성
    train_label_dir = os.path.join(yolo_dir, "labels", "train")
    val_label_dir = os.path.join(yolo_dir, "labels", "val")
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    train_img_dir = os.path.join(yolo_dir, "images", "train")
    val_img_dir = os.path.join(yolo_dir, "images", "val")
    
    train_images = set([f.replace(".png", "") for f in os.listdir(train_img_dir) if f.endswith(".png")])
    val_images = set([f.replace(".png", "") for f in os.listdir(val_img_dir) if f.endswith(".png")])
    
    # JSON 파일 찾기
    json_files = glob.glob(os.path.join(train_ann_dir, "**/*.json"), recursive=True)
    
    print(f"총 {len(json_files)}개의 JSON 파일 발견")
    print(f"Train 이미지: {len(train_images)}개")
    print(f"Val 이미지: {len(val_images)}개")
    
    converted_train = 0
    converted_val = 0
    skipped = 0
    
    for json_path in tqdm(json_files, desc="변환 중"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 이미지 정보
            img_info = data["images"][0]
            img_name = img_info["file_name"]
            img_name_no_ext = img_name.replace(".png", "").replace(".jpg", "")
            img_w = img_info["width"]
            img_h = img_info["height"]
            
            # Train 또는 Val 결정
            if img_name_no_ext in train_images:
                label_dir = train_label_dir
                converted_train += 1
            elif img_name_no_ext in val_images:
                label_dir = val_label_dir
                converted_val += 1
            else:
                skipped += 1
                continue
            
            # 라벨 파일 생성
            label_path = os.path.join(label_dir, f"{img_name_no_ext}.txt")
            
            with open(label_path, "w", encoding="utf-8") as lf:
                for ann in data["annotations"]:
                    x, y, w, h = ann["bbox"]
                    category_id = int(ann["category_id"])
                    
                    # Category ID를 클래스 인덱스로 변환
                    if category_id not in cat2idx:
                        continue
                    
                    class_idx = cat2idx[category_id]
                    
                    # YOLO 형식으로 변환
                    cx, cy, w_norm, h_norm = convert_bbox_to_yolo(x, y, w, h, img_w, img_h)
                    
                    # 라벨 파일에 쓰기 (class_idx cx cy w h)
                    lf.write(f"{class_idx} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        except Exception as e:
            print(f"\n⚠️ JSON 파싱 실패: {json_path} - {e}")
            continue
    
    print(f"\n=== 변환 완료 ===")
    print(f"Train 라벨 생성: {converted_train}개")
    print(f"Val 라벨 생성: {converted_val}개")
    print(f"건너뜀: {skipped}개")
    
    # 생성된 라벨 파일 수 확인
    train_labels = len([f for f in os.listdir(train_label_dir) if f.endswith(".txt")])
    val_labels = len([f for f in os.listdir(val_label_dir) if f.endswith(".txt")])
    
    print(f"\n실제 생성된 라벨 파일:")
    print(f"Train: {train_labels}개")
    print(f"Val: {val_labels}개")


if __name__ == "__main__":
    # 경로 설정
    BASE = r"D:/스프린트AI엔지니어 부트캠프/part2_kaggle/6team_beginner_project"
    TRAIN_ANN_DIR = os.path.join(BASE, "train_annotations")
    YOLO_DIR = os.path.join(BASE, "yolo_multiclass")
    
    print("=== YOLO 라벨 파일 생성 ===")
    print(f"BASE: {BASE}")
    print(f"Train Annotations: {TRAIN_ANN_DIR}")
    print(f"YOLO Directory: {YOLO_DIR}\n")
    
    create_yolo_labels(BASE, TRAIN_ANN_DIR, YOLO_DIR)

