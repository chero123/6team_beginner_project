"""
Iterative Pseudo-labeling 스크립트

전략:
1. 레이블이 있는 데이터로 초기 모델 학습
2. 높은 confidence의 pseudo-label 생성 (신뢰할 수 있는 예측만)
3. 레이블 + pseudo-label로 모델 재학습
4. 반복 (점진적으로 confidence threshold 낮춤)

이렇게 하면 점진적으로 더 많은 데이터로 모델을 개선할 수 있습니다.
"""

import os
import json
import glob
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO
from PIL import Image
import subprocess

# 현재 스크립트 디렉토리를 sys.path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def validate_pseudo_labels_on_val_set(model_path, val_img_dir, val_label_dir, 
                                     category_mapping_path, conf_threshold=0.7):
    """
    Validation set에서 pseudo-label의 품질 검증
    
    Returns:
        mAP50 점수 (높을수록 좋음)
    """
    try:
        model = YOLO(model_path)
        
        # Category mapping 로드
        with open(category_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Validation set으로 검증
        dataset_yaml = os.path.join(os.path.dirname(val_img_dir), "yolo_multiclass", "dataset.yaml")
        
        metrics = model.val(
            data=dataset_yaml,
            imgsz=800,
            conf=conf_threshold,
            iou=0.5,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        
        return metrics.box.map50 if hasattr(metrics, 'box') else 0.0
    except Exception as e:
        print(f"⚠️ 검증 중 오류: {e}")
        return 0.0


def generate_high_confidence_pseudo_labels(model_path, missing_images, category_mapping_path,
                                           output_dir, conf_threshold=0.7, min_boxes=1):
    """
    높은 confidence의 pseudo-label만 생성
    
    Args:
        conf_threshold: 높은 confidence threshold (기본값: 0.7)
        min_boxes: 최소 검출 박스 개수
    """
    model = YOLO(model_path)
    
    with open(category_mapping_path, 'r', encoding='utf-8') as f:
        category_mapping = json.load(f)
    
    device = 0 if torch.cuda.is_available() else "cpu"
    generated_count = 0
    skipped_count = 0
    ann_id = 20000  # 기존과 겹치지 않도록
    
    print(f"\n높은 confidence pseudo-label 생성 (threshold={conf_threshold})...")
    
    for img_info in tqdm(missing_images, desc="High-confidence labeling"):
        img_path = img_info['path']
        img_name_no_ext = img_info['name']
        
        try:
            # 예측 수행 (낮은 threshold로 모든 예측 확인)
            results = model.predict(
                img_path,
                imgsz=800,
                conf=0.01,  # 모든 예측 확인
                iou=0.5,
                max_det=300,
                device=device,
                verbose=False
            )[0]
            
            # 높은 confidence만 필터링
            high_conf_boxes = []
            for box in results.boxes:
                if float(box.conf) >= conf_threshold:
                    high_conf_boxes.append(box)
            
            if len(high_conf_boxes) < min_boxes:
                skipped_count += 1
                continue
            
            # JSON 어노테이션 생성
            from generate_pseudo_labels import create_annotation_json
            annotation_data = create_annotation_json(
                img_path, results, category_mapping, ann_id, conf_threshold
            )
            
            if len(annotation_data['annotations']) < min_boxes:
                skipped_count += 1
                continue
            
            # JSON 파일 저장
            json_filename = f"{img_name_no_ext}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
            generated_count += 1
            ann_id += 1
            
        except Exception as e:
            skipped_count += 1
            continue
    
    return generated_count, skipped_count


def iterative_pseudo_labeling(base_dir, train_img_dir, train_ann_dir, 
                              yolo_dir, category_mapping_path,
                              initial_epochs=50, iterations=3):
    """
    Iterative Pseudo-labeling 수행
    
    Args:
        iterations: 반복 횟수 (기본값: 3)
    """
    print("="*60)
    print("Iterative Pseudo-labeling 시작")
    print("="*60)
    
    # 어노테이션이 없는 이미지 찾기
    from generate_pseudo_labels import find_images_without_annotations
    missing_images = find_images_without_annotations(base_dir, train_img_dir, train_ann_dir)
    print(f"\n어노테이션이 없는 이미지: {len(missing_images)}개")
    
    if len(missing_images) == 0:
        print("어노테이션이 없는 이미지가 없습니다.")
        return
    
    # Iteration별 confidence threshold (점진적으로 낮춤)
    conf_thresholds = [0.8, 0.7, 0.6]  # 첫 번째 iteration은 매우 높은 threshold
    
    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"{'='*60}")
        
        # 1. 모델 학습
        print("\n1단계: 모델 학습...")
        model_name = f"pill_yolo_iter{iteration + 1}"
        
        # 학습 스크립트 실행
        train_script = os.path.join(base_dir, "model_architecture", "train_improved.py")
        
        print(f"학습 스크립트 실행: {train_script}")
        print("⚠️ 주의: 모델 학습은 수동으로 실행해야 합니다.")
        print(f"   명령어: python {train_script} --force-train")
        print("\n학습이 완료되면 Enter를 눌러 계속하세요...")
        input()
        
        # 모델 경로 찾기 (기본 모델 이름 사용)
        possible_paths = [
            os.path.join(base_dir, "runs", "detect", "pill_yolo_improved", "weights", "best.pt"),
            os.path.join(base_dir, "pill_yolo_improved", "weights", "best.pt"),
            os.path.join(base_dir, "runs", "detect", "pill_yolo_improved2", "weights", "best.pt"),
            os.path.join(base_dir, "pill_yolo_improved2", "weights", "best.pt"),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("⚠️ 학습된 모델을 찾을 수 없습니다.")
            print("   모델 경로를 직접 입력하세요:")
            model_path = input().strip()
            if not os.path.exists(model_path):
                print("❌ 모델 파일을 찾을 수 없습니다. 중단합니다.")
                break
        
        print(f"✅ 모델 경로 확인: {model_path}")
        
        # 2. Validation set으로 모델 품질 확인
        print("\n2단계: 모델 품질 검증...")
        val_map50 = validate_pseudo_labels_on_val_set(
            model_path, 
            os.path.join(yolo_dir, "images", "val"),
            os.path.join(yolo_dir, "labels", "val"),
            category_mapping_path,
            conf_threshold=0.5
        )
        print(f"Validation mAP50: {val_map50:.4f}")
        
        # 모델이 너무 나쁘면 중단
        if val_map50 < 0.1 and iteration > 0:
            print("⚠️ 모델 품질이 너무 낮습니다. 중단합니다.")
            break
        
        # 3. 높은 confidence의 pseudo-label 생성
        conf_threshold = conf_thresholds[min(iteration, len(conf_thresholds) - 1)]
        print(f"\n3단계: 높은 confidence pseudo-label 생성 (threshold={conf_threshold})...")
        
        generated, skipped = generate_high_confidence_pseudo_labels(
            model_path,
            missing_images,
            category_mapping_path,
            train_ann_dir,  # 출력 디렉토리
            conf_threshold=conf_threshold,
            min_boxes=1
        )
        
        print(f"생성된 pseudo-label: {generated}개")
        print(f"건너뜀: {skipped}개")
        
        if generated == 0:
            print("⚠️ 생성된 pseudo-label이 없습니다. threshold를 낮추거나 중단합니다.")
            if iteration == 0:
                print("❌ 첫 번째 iteration에서 pseudo-label이 생성되지 않았습니다.")
                print("   레이블이 있는 데이터로 모델을 더 많이 학습하거나,")
                print("   confidence threshold를 낮춰야 합니다.")
            break
        
        # 4. YOLO 레이블 재생성
        print("\n4단계: YOLO 레이블 재생성...")
        from create_yolo_labels import create_yolo_labels
        create_yolo_labels(base_dir, train_ann_dir, yolo_dir)
        
        # 다음 iteration을 위해 missing_images 업데이트
        missing_images = find_images_without_annotations(base_dir, train_img_dir, train_ann_dir)
        print(f"\n남은 어노테이션 없는 이미지: {len(missing_images)}개")
        
        if len(missing_images) == 0:
            print("✅ 모든 이미지에 어노테이션이 생성되었습니다!")
            break
    
    print(f"\n{'='*60}")
    print("Iterative Pseudo-labeling 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(script_dir)
    
    TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
    TRAIN_ANN_DIR = os.path.join(BASE, "train_annotations")
    YOLO_DIR = os.path.join(BASE, "yolo_multiclass")
    CATEGORY_MAPPING = os.path.join(BASE, "category_mapping.json")
    
    print("=== Iterative Pseudo-labeling ===")
    print(f"BASE: {BASE}")
    print(f"Train 이미지: {TRAIN_IMG_DIR}")
    print(f"Train 어노테이션: {TRAIN_ANN_DIR}")
    print(f"YOLO 디렉토리: {YOLO_DIR}\n")
    
    iterative_pseudo_labeling(
        BASE,
        TRAIN_IMG_DIR,
        TRAIN_ANN_DIR,
        YOLO_DIR,
        CATEGORY_MAPPING,
        initial_epochs=50,
        iterations=3
    )

