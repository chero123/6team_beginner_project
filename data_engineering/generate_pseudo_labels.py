"""
어노테이션이 없는 이미지에 대해 pseudo-labeling을 수행하는 스크립트

사용 방법:
1. 먼저 레이블이 있는 이미지로 모델을 학습
2. 학습된 모델을 사용하여 어노테이션이 없는 이미지에 예측 수행
3. 높은 confidence의 예측만 JSON 어노테이션 파일로 저장
"""

import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from ultralytics import YOLO
from PIL import Image


def find_images_without_annotations(base_dir, train_img_dir, train_ann_dir):
    """
    어노테이션이 없는 이미지 찾기
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
        train_img_dir: 학습 이미지 디렉토리
        train_ann_dir: 어노테이션 디렉토리
    
    Returns:
        어노테이션이 없는 이미지 경로 리스트
    """
    # 모든 이미지 파일 찾기
    all_images = set()
    for ext in ['.png', '.jpg', '.jpeg']:
        all_images.update([f.replace(ext, '') for f in os.listdir(train_img_dir) 
                          if f.lower().endswith(ext)])
    
    # 어노테이션이 있는 이미지 찾기
    json_files = glob.glob(os.path.join(train_ann_dir, "**/*.json"), recursive=True)
    annotated_images = set()
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'images' in data and len(data['images']) > 0:
                    img_name = data['images'][0]['file_name']
                    img_name_no_ext = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    annotated_images.add(img_name_no_ext)
        except:
            continue
    
    # 어노테이션이 없는 이미지 찾기
    missing_images = []
    for img_name_no_ext in all_images:
        if img_name_no_ext not in annotated_images:
            # 실제 이미지 파일 찾기
            for ext in ['.png', '.jpg', '.jpeg']:
                img_path = os.path.join(train_img_dir, f"{img_name_no_ext}{ext}")
                if os.path.exists(img_path):
                    missing_images.append({
                        'name': img_name_no_ext,
                        'path': img_path,
                        'ext': ext
                    })
                    break
    
    return missing_images


def create_annotation_json(img_path, predictions, category_mapping, ann_id_start=1, conf_threshold=0.5):
    """
    예측 결과를 JSON 어노테이션 형식으로 변환
    
    Args:
        img_path: 이미지 경로
        predictions: YOLO 예측 결과
        category_mapping: 카테고리 매핑 딕셔너리
        ann_id_start: 어노테이션 ID 시작 번호
        conf_threshold: Confidence threshold
    
    Returns:
        JSON 어노테이션 딕셔너리
    """
    # 이미지 정보 가져오기
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    # 이미지 파일 이름
    img_name = os.path.basename(img_path)
    img_name_no_ext = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    
    # 기본 이미지 정보 (실제 데이터에서 가져올 수 있으면 더 좋음)
    image_info = {
        "file_name": img_name,
        "width": img_w,
        "height": img_h,
        "imgfile": img_name,
        "id": ann_id_start
    }
    
    # 어노테이션 생성
    annotations = []
    categories = []
    category_ids_used = set()
    
    idx2cat = category_mapping.get('idx2cat', {})
    
    annotation_idx = 0
    for box in predictions.boxes:
        cls = int(box.cls)
        score = float(box.conf)
        
        # Confidence threshold 체크
        if score < conf_threshold:
            continue
        
        # 클래스 인덱스를 카테고리 ID로 변환
        if str(cls) not in idx2cat:
            continue
        
        category_id = int(idx2cat[str(cls)])
        category_ids_used.add(category_id)
        
        # YOLO 형식 (정규화된 좌표)을 COCO 형식 (절대 좌표)으로 변환
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h
        
        annotation = {
            "area": int(area),
            "iscrowd": 0,
            "bbox": [int(x), int(y), int(w), int(h)],
            "category_id": category_id,
            "ignore": 0,
            "segmentation": [],
            "id": ann_id_start + annotation_idx,
            "image_id": ann_id_start
        }
        annotations.append(annotation)
        annotation_idx += 1
    
    # 카테고리 정보 생성
    # 실제 JSON 파일에서 카테고리 이름을 가져올 수 없으므로 기본값 사용
    cat2name = category_mapping.get('cat2name', {})
    for cat_id in category_ids_used:
        category = {
            "supercategory": "pill",
            "id": cat_id,
            "name": cat2name.get(str(cat_id), f"category_{cat_id}")
        }
        categories.append(category)
    
    # JSON 구조 생성
    annotation_data = {
        "images": [image_info],
        "type": "instances",
        "annotations": annotations,
        "categories": categories
    }
    
    return annotation_data


def generate_pseudo_labels(model_path, base_dir, train_img_dir, train_ann_dir, 
                          category_mapping_path, output_dir=None, 
                          conf_threshold=0.5, min_boxes=1):
    """
    어노테이션이 없는 이미지에 대해 pseudo-labeling 수행
    
    Args:
        model_path: 학습된 YOLO 모델 경로
        base_dir: 프로젝트 기본 디렉토리
        train_img_dir: 학습 이미지 디렉토리
        train_ann_dir: 어노테이션 디렉토리
        category_mapping_path: 카테고리 매핑 JSON 파일 경로
        output_dir: 출력 디렉토리 (None이면 train_ann_dir 사용)
        conf_threshold: Confidence threshold (기본값: 0.5)
        min_boxes: 최소 검출 박스 개수 (이보다 적으면 저장하지 않음)
    """
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # Category mapping 로드
    with open(category_mapping_path, 'r', encoding='utf-8') as f:
        category_mapping = json.load(f)
    
    # 어노테이션이 없는 이미지 찾기
    print("\n어노테이션이 없는 이미지 찾는 중...")
    missing_images = find_images_without_annotations(base_dir, train_img_dir, train_ann_dir)
    print(f"어노테이션이 없는 이미지: {len(missing_images)}개")
    
    if len(missing_images) == 0:
        print("어노테이션이 없는 이미지가 없습니다.")
        return
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = train_ann_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 디바이스 설정
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {'GPU' if device == 0 else 'CPU'}")
    
    # Pseudo-labeling 수행
    generated_count = 0
    skipped_count = 0
    ann_id = 10000  # 기존 어노테이션과 겹치지 않도록 높은 ID 사용
    
    # 디버깅: 예측 통계
    total_predictions = 0
    max_confidences = []
    
    print(f"\nPseudo-labeling 시작 (conf_threshold={conf_threshold}, min_boxes={min_boxes})...")
    
    for idx, img_info in enumerate(tqdm(missing_images, desc="Pseudo-labeling")):
        img_path = img_info['path']
        img_name_no_ext = img_info['name']
        
        try:
            # 예측 수행 (confidence threshold를 낮춰서 모든 예측 확인)
            results = model.predict(
                img_path,
                imgsz=800,
                conf=0.01,  # 매우 낮은 threshold로 모든 예측 확인
                iou=0.5,
                max_det=300,
                device=device,
                verbose=False
            )[0]
            
            # 디버깅: 첫 번째 이미지의 예측 결과 출력
            if idx == 0:
                print(f"\n[디버깅] 첫 번째 이미지: {img_name_no_ext}")
                print(f"  - 검출된 박스 개수: {len(results.boxes)}")
                if len(results.boxes) > 0:
                    confidences = [float(box.conf) for box in results.boxes]
                    print(f"  - Confidence 범위: {min(confidences):.4f} ~ {max(confidences):.4f}")
                    print(f"  - 평균 Confidence: {sum(confidences)/len(confidences):.4f}")
                    print(f"  - {conf_threshold} 이상인 박스: {sum(1 for c in confidences if c >= conf_threshold)}개")
            
            # Confidence threshold로 필터링
            filtered_boxes = []
            for box in results.boxes:
                if float(box.conf) >= conf_threshold:
                    filtered_boxes.append(box)
            
            # 검출된 박스가 최소 개수 이상인지 확인
            if len(filtered_boxes) < min_boxes:
                skipped_count += 1
                if len(results.boxes) > 0:
                    max_confidences.append(max([float(box.conf) for box in results.boxes]))
                continue
            
            # JSON 어노테이션 생성 (confidence threshold는 함수 내에서 체크)
            annotation_data = create_annotation_json(
                img_path, results, category_mapping, ann_id, conf_threshold
            )
            
            # 어노테이션이 실제로 생성되었는지 확인
            if len(annotation_data['annotations']) < min_boxes:
                skipped_count += 1
                continue
            
            # JSON 파일 저장 (기존 구조 유지)
            # 파일 이름은 이미지 이름과 동일하게
            json_filename = f"{img_name_no_ext}.json"
            json_path = os.path.join(output_dir, json_filename)
            
            # 기존 디렉토리 구조를 유지하려면 더 복잡한 로직 필요
            # 일단 간단하게 output_dir에 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)
            
            generated_count += 1
            ann_id += 1
            
        except Exception as e:
            print(f"\n⚠️ 오류 발생 ({img_name_no_ext}): {e}")
            skipped_count += 1
            continue
    
    print(f"\n=== Pseudo-labeling 완료 ===")
    print(f"생성된 어노테이션: {generated_count}개")
    print(f"건너뜀: {skipped_count}개")
    print(f"출력 디렉토리: {output_dir}")
    
    if max_confidences:
        print(f"\n[통계] 예측이 있었지만 threshold 미달인 이미지들:")
        print(f"  - 평균 최대 confidence: {sum(max_confidences)/len(max_confidences):.4f}")
        print(f"  - 최대 confidence: {max(max_confidences):.4f}")
        print(f"  - 최소 confidence: {min(max_confidences):.4f}")
        print(f"  - {conf_threshold} 이상인 이미지: {sum(1 for c in max_confidences if c >= conf_threshold)}개")
        print(f"\n💡 Tip: Confidence threshold를 낮추면 더 많은 어노테이션이 생성될 수 있습니다.")
        print(f"   예: python generate_pseudo_labels.py <model_path> 0.3")


if __name__ == "__main__":
    import sys
    
    # 경로 설정 (스크립트 파일 위치를 기준으로 동적으로 설정)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(script_dir)  # data_engineering의 상위 디렉토리 = 프로젝트 루트
    
    TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
    TRAIN_ANN_DIR = os.path.join(BASE, "train_annotations")
    CATEGORY_MAPPING = os.path.join(BASE, "category_mapping.json")
    
    # 모델 경로 (기본값: 최근 학습된 모델)
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    else:
        # 기본 모델 경로 찾기
        possible_models = [
            os.path.join(BASE, "pill_yolo_improved2", "weights", "best.pt"),
            os.path.join(BASE, "pill_yolo_improved", "weights", "best.pt"),
            os.path.join(BASE, "runs", "detect", "pill_yolo_improved2", "weights", "best.pt"),
        ]
        MODEL_PATH = None
        for path in possible_models:
            if os.path.exists(path):
                MODEL_PATH = path
                break
        
        if MODEL_PATH is None:
            print("❌ 모델 파일을 찾을 수 없습니다.")
            print("사용법: python generate_pseudo_labels.py <model_path>")
            print("또는 모델을 먼저 학습하세요.")
            sys.exit(1)
    
    # Confidence threshold 설정
    conf_threshold = 0.5
    if len(sys.argv) > 2:
        conf_threshold = float(sys.argv[2])
    
    print("=== Pseudo-labeling 시작 ===")
    print(f"모델: {MODEL_PATH}")
    print(f"Train 이미지 디렉토리: {TRAIN_IMG_DIR}")
    print(f"Train 어노테이션 디렉토리: {TRAIN_ANN_DIR}")
    print(f"Confidence threshold: {conf_threshold}\n")
    
    generate_pseudo_labels(
        MODEL_PATH,
        BASE,
        TRAIN_IMG_DIR,
        TRAIN_ANN_DIR,
        CATEGORY_MAPPING,
        conf_threshold=conf_threshold,
        min_boxes=1  # 최소 1개 이상 검출되어야 저장
    )

