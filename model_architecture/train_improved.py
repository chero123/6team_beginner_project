"""
개선된 YOLO 모델 학습 스크립트

주요 개선 사항:
1. 더 큰 모델: YOLOv8m → YOLOv8l
2. 더 많은 epochs: 20 → 50
3. 더 큰 이미지 크기: 640 → 800
4. 개선된 Augmentation
5. 학습률 스케줄링
6. Early Stopping
7. TTA (Test Time Augmentation)
"""

import os
import json
import random
import re
import glob
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import yaml

# OpenMP 중복 초기화 문제 해결 (Windows)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def set_seed(seed=42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_improved_model(base_dir, yolo_dir, device=0, epochs=50, model_name="pill_yolo_improved"):
    """
    개선된 하이퍼파라미터로 YOLO 모델 학습
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
        yolo_dir: YOLO 데이터셋 디렉토리
        device: GPU 디바이스 번호 또는 'cpu'
        epochs: 학습 에포크 수
        model_name: 모델 이름 (저장 폴더명)
    
    Returns:
        학습 결과와 모델 경로를 포함한 딕셔너리
    """
    dataset_yaml = os.path.join(yolo_dir, "dataset.yaml")
    
    # dataset.yaml 파일의 path를 동적으로 업데이트
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f) or {}
        
        # path를 yolo_dir의 절대 경로로 설정
        dataset_config['path'] = os.path.abspath(yolo_dir)
        
        # 업데이트된 설정을 파일에 저장
        with open(dataset_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    # YOLOv8x 모델 사용 (가장 큰 모델, 최고 정확도)
    # GPU 메모리가 부족하면 yolov8l.pt로 변경
    try:
        model = YOLO("yolov8x.pt")
        print("✅ YOLOv8x 모델 사용 (최고 성능)")
    except:
        model = YOLO("yolov8l.pt")
        print("⚠️ YOLOv8x 로드 실패, YOLOv8l 사용")
    
    # 개선된 하이퍼파라미터로 학습
    results = model.train(
        data=dataset_yaml,
        
        # 모델 설정
        epochs=epochs,              # 기본값 사용 (더 충분한 학습)
        imgsz=1024,                # 800 → 1024 (더 큰 이미지로 작은 객체 검출 개선)
        batch=4,                    # 8 → 4 (더 큰 이미지로 인한 메모리 절약)
        device=device,
        name=model_name,
        project=base_dir,          # 프로젝트 디렉토리 명시적으로 지정 (경로 문제 해결)
        
        # 학습률 설정 (Cosine Annealing)
        lr0=0.0005,                # 0.001 → 0.0005 (더 안정적인 학습)
        lrf=0.01,                  # 최종 학습률 비율
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,         # 3.0 → 5.0 (더 긴 warmup)
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,               # Cosine learning rate scheduler 활성화
        
        # Augmentation (약 이미지에 최적화, 더 강화)
        hsv_h=0.02,                # 0.015 → 0.02 (색조 변화 증가)
        hsv_s=0.7,                 # 채도 변화 (약의 색상 다양성 반영)
        hsv_v=0.4,                 # 명도 변화
        degrees=15,                 # 10 → 15 (회전 각도 증가)
        translate=0.15,            # 0.1 → 0.15 (이동 증가)
        scale=0.6,                 # 0.5 → 0.6 (크기 변화 범위 증가)
        shear=8,                   # 5 → 8 (전단 변환 증가)
        perspective=0.0002,        # 0.0001 → 0.0002 (원근 변환 증가)
        fliplr=0.5,                # 좌우 반전
        flipud=0.0,                # 상하 반전 (약 이미지에는 부적절)
        mosaic=1.0,                # Mosaic augmentation
        mixup=0.15,                # 0.1 → 0.15 (Mixup augmentation 증가)
        copy_paste=0.15,           # 0.1 → 0.15 (Copy-paste augmentation 증가)
        erasing=0.4,               # Random erasing 추가
        auto_augment="randaugment", # Auto augmentation 활성화
        
        # 학습 설정
        patience=20,               # 15 → 20 (Early stopping patience 증가)
        save=True,
        save_period=5,             # 10 → 5 (더 자주 체크포인트 저장)
        val=True,
        plots=True,
        close_mosaic=10,           # 마지막 10 epoch에서 mosaic 비활성화
        
        # 재현성
        seed=42,
        deterministic=True,
        
        # 기타
        workers=0,                 # Windows 멀티프로세싱 문제 해결 (0 = 메인 프로세스만 사용)
        amp=True,                  # Automatic Mixed Precision (속도 향상)
        fraction=1.0,              # 전체 데이터셋 사용
        profile=False,
        freeze=None,
        multi_scale=False,        # Multi-scale training (메모리 절약)
        
        # Loss 가중치 (더 정교한 튜닝)
        box=7.5,                   # Box loss 가중치
        cls=0.5,                   # Classification loss 가중치
        dfl=1.5,                   # Distribution Focal Loss 가중치
        
        # NMS 설정
        iou=0.7,                   # NMS IoU threshold
        conf=0.25,                 # Confidence threshold
        max_det=300,               # 최대 검출 개수
        
        # 추가 최적화
        optimizer="AdamW",         # SGD → AdamW (더 나은 수렴)
        nbs=64,                    # Nominal batch size
        overlap_mask=True,         # Overlap mask 활성화
    )
    
    # 모델 경로 반환 (YOLO가 실제로 저장한 경로 사용)
    # YOLO는 project 파라미터를 지정하지 않으면 현재 작업 디렉토리 기준으로 저장
    # 여러 가능한 경로를 확인하여 실제 존재하는 경로 찾기
    
    # results 객체에서 실제 저장 경로 확인
    if hasattr(results, "save_dir") and results.save_dir:
        # YOLO가 반환한 실제 저장 디렉토리 사용
        model_path = os.path.join(results.save_dir, "weights", "best.pt")
        if os.path.exists(model_path):
            print(f"✅ 모델 파일 발견 (results.save_dir): {model_path}")
            return {
                "results": results,
                "model_path": model_path,
                "model_name": model_name
            }
    
    # 여러 가능한 경로 확인
    possible_paths = [
        os.path.join(base_dir, "runs", "detect", model_name, "weights", "best.pt"),
        os.path.join(os.getcwd(), "runs", "detect", model_name, "weights", "best.pt"),
    ]
    
    # 모델 이름 변형도 확인 (pill_yolo_improved2 등)
    model_name_variants = [model_name, f"{model_name}2", f"{model_name}_2"]
    for variant in model_name_variants:
        possible_paths.extend([
            os.path.join(base_dir, "runs", "detect", variant, "weights", "best.pt"),
            os.path.join(os.getcwd(), "runs", "detect", variant, "weights", "best.pt"),
        ])
    
    # 상위 디렉토리에서도 검색 (D:\AI \part2_kaggle 같은 경우 대비)
    parent_dirs = [
        os.path.dirname(base_dir),  # base_dir의 상위 디렉토리
        os.path.dirname(os.getcwd()),  # 현재 작업 디렉토리의 상위 디렉토리
    ]
    for parent_dir in parent_dirs:
        if parent_dir and os.path.exists(parent_dir):
            for variant in model_name_variants:
                possible_paths.append(
                    os.path.join(parent_dir, "runs", "detect", variant, "weights", "best.pt")
                )
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ 모델 파일 발견: {model_path}")
            break
    
    if model_path is None:
        # 기본 경로 사용 (나중에 에러 처리)
        model_path = os.path.join(base_dir, "runs", "detect", model_name, "weights", "best.pt")
        print(f"⚠️ 모델 파일을 찾지 못했습니다. 예상 경로: {model_path}")
        print(f"   다음 경로들을 확인했습니다:")
        for path in possible_paths[:10]:  # 처음 10개만 출력
            print(f"   - {path}")
    
    return {
        "results": results,
        "model_path": model_path,
        "model_name": model_name
    }


def validate_model(model_path, dataset_yaml, device=0):
    """모델 검증"""
    model = YOLO(model_path)
    
    metrics = model.val(
        data=dataset_yaml,
        imgsz=1024,                # 800 → 1024 (학습과 동일한 크기)
        conf=0.25,
        iou=0.7,
        device=device,
    )
    
    print(f"\n=== 검증 결과 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


def save_training_summary(model_path, metrics, output_path):
    """
    학습 결과 요약을 CSV로 저장
    
    Args:
        model_path: 모델 경로
        metrics: 검증 메트릭
        output_path: 출력 파일 경로
    """
    summary = {
        "model_path": [model_path],
        "mAP50": [metrics.box.map50],
        "mAP50-95": [metrics.box.map],
        "Precision": [metrics.box.mp],
        "Recall": [metrics.box.mr],
    }
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(output_path, index=False)
    print(f"학습 결과 요약 저장: {output_path}")


def predict_with_tta(model, img_path, conf_threshold=0.5, iou_threshold=0.5, max_det=300):
    """
    Test Time Augmentation을 사용한 추론
    
    Args:
        model: YOLO 모델
        img_path: 이미지 경로
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        max_det: 최대 검출 개수
    
    Returns:
        추론 결과
    """
    results = model.predict(
        img_path,
        imgsz=1024,                # 800 → 1024 (학습과 동일한 크기)
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        augment=True,  # TTA 활성화
        verbose=False
    )
    
    return results[0]


def generate_submission(model_path, test_img_dir, category_mapping_path, 
                       output_path, conf_threshold=0.5, use_tta=False, iou_threshold=0.5, max_det=300):
    """
    Kaggle 제출용 CSV 파일 생성
    
    Args:
        model_path: 학습된 모델 경로
        test_img_dir: 테스트 이미지 디렉토리
        category_mapping_path: Category mapping JSON 파일 경로
        output_path: 출력 CSV 파일 경로
        conf_threshold: Confidence threshold (기본값: 0.5, 더 높은 정확도)
        use_tta: TTA 사용 여부 (기본값: False, TTA는 때때로 성능을 떨어뜨림)
        iou_threshold: NMS IoU threshold (기본값: 0.5, 더 엄격한 필터링)
        max_det: 최대 검출 개수
    """
    # 모델 로드
    model = YOLO(model_path)
    
    # Category mapping 로드
    with open(category_mapping_path, "r") as f:
        mapping = json.load(f)
    idx2cat = {int(k): v for k, v in mapping["idx2cat"].items()}
    
    # 제출 파일 생성
    submission = []
    ann_id = 1
    
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])
    total_images = len(test_images)
    
    print(f"총 {total_images}개의 테스트 이미지 처리 시작...")
    print(f"설정: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, use_tta={use_tta}")
    
    for idx, img_name in enumerate(test_images, 1):
        if idx % 10 == 0 or idx == total_images:
            print(f"진행 중... ({idx}/{total_images} 이미지 처리 완료)")
        img_id = int(img_name.replace(".png", ""))
        img_path = os.path.join(test_img_dir, img_name)
        
        # 추론 (더 엄격한 설정)
        if use_tta:
            results = predict_with_tta(model, img_path, conf_threshold, iou_threshold, max_det)
        else:
            results = model.predict(
                img_path,
                imgsz=1024,              # 800 → 1024 (학습과 동일한 크기)
                conf=conf_threshold,      # Confidence threshold
                iou=iou_threshold,        # NMS IoU threshold
                max_det=max_det,          # 최대 검출 개수
                verbose=False
            )[0]
        
        # 디버깅: 첫 번째 이미지에서 예측 결과 확인
        if idx == 1:
            print(f"\n[디버깅] 첫 번째 이미지 예측 결과:")
            print(f"  - 검출된 박스 개수: {len(results.boxes)}")
            if len(results.boxes) > 0:
                print(f"  - 첫 번째 박스 confidence: {float(results.boxes[0].conf):.4f}")
                print(f"  - 첫 번째 박스 class: {int(results.boxes[0].cls)}")
        
        # Score 기반으로 추가 필터링 (confidence가 낮은 예측 제거)
        for box in results.boxes:
            cls = int(box.cls)
            score = float(box.conf)
            
            # Confidence가 threshold보다 낮으면 제외
            if score < conf_threshold:
                continue
                
            orig_cid = idx2cat[cls]
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            
            # Bounding box 유효성 검사 (너무 작거나 음수인 경우 제외)
            if w <= 0 or h <= 0 or x1 < 0 or y1 < 0:
                continue
            
            submission.append([
                ann_id, img_id, orig_cid,
                float(x1), float(y1), float(w), float(h), score
            ])
            ann_id += 1
    
    # CSV 저장
    df = pd.DataFrame(submission, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    
    df.to_csv(output_path, index=False)
    print(f"제출 파일 저장 완료: {output_path}")
    print(f"총 예측 개수: {len(submission)}")
    
    return df


def extract_model_name_from_path(model_path):
    """
    모델 경로에서 모델 이름 추출
    
    Args:
        model_path: 모델 파일 경로
    
    Returns:
        모델 이름
    """
    # 경로를 정규화
    normalized_path = model_path.replace("\\", "/")
    path_parts = normalized_path.split("/")
    
    # pill_yolo_improved* 패턴 찾기
    for part in path_parts:
        if "pill_yolo_improved" in part:
            return part
    
    # runs/detect/모델이름/weights/best.pt 형식
    if "runs" in path_parts and "detect" in path_parts:
        detect_idx = path_parts.index("detect")
        if detect_idx + 1 < len(path_parts):
            return path_parts[detect_idx + 1]
    
    # 기본값
    return "pill_yolo_improved"


def find_existing_model(base_dir, model_name="pill_yolo_improved"):
    """
    기존에 학습된 모델 파일을 찾기 (가장 최근 모델 우선)
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
        model_name: 모델 이름 (기본값, 모든 모델 검색)
    
    Returns:
        가장 최근에 수정된 모델 경로 또는 None
    """
    found_models = []
    
    # 모든 가능한 경로에서 모델 검색
    search_dirs = [
        base_dir,
        os.getcwd(),
        os.path.dirname(base_dir),
        os.path.dirname(os.getcwd()),
    ]
    
    for search_dir in search_dirs:
        if not search_dir or not os.path.exists(search_dir):
            continue
        
        # runs/detect/*/weights/best.pt 패턴 검색
        runs_detect_pattern = os.path.join(search_dir, "runs", "detect", "*", "weights", "best.pt")
        found_models.extend(glob.glob(runs_detect_pattern))
        
        # 직접 모델 디렉토리 패턴 검색 (pill_yolo_improved*/weights/best.pt)
        model_pattern = os.path.join(search_dir, "pill_yolo_improved*", "weights", "best.pt")
        found_models.extend(glob.glob(model_pattern))
    
    if not found_models:
        return None
    
    # 중복 제거 및 실제 존재하는 파일만 필터링
    found_models = list(set([f for f in found_models if os.path.exists(f)]))
    
    if not found_models:
        return None
    
    # 수정 시간 기준으로 정렬 (가장 최근 것 우선)
    found_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 가장 최근 모델 반환
    latest_model = found_models[0]
    print(f"📌 발견된 모델 개수: {len(found_models)}개")
    print(f"📌 가장 최근 모델 선택: {os.path.basename(os.path.dirname(os.path.dirname(latest_model)))}")
    
    return latest_model


if __name__ == "__main__":
    import sys
    
    # 경로 설정 (스크립트 파일 위치를 기준으로 동적으로 설정)
    # 현재 스크립트 파일의 디렉토리: model_architecture/
    # 프로젝트 루트: 상위 디렉토리 1개
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(script_dir)  # model_architecture의 상위 디렉토리 = 프로젝트 루트
    
    # 데이터셋 우선순위: 병합된 데이터셋 > yolo_dataset > yolo_multiclass
    yolo_merged_path = os.path.join(BASE, "yolo_dataset_merged")
    yolo_dataset_path = os.path.join(BASE, "yolo_dataset")
    yolo_multiclass_path = os.path.join(BASE, "yolo_multiclass")
    
    if os.path.exists(yolo_merged_path) and os.path.exists(os.path.join(yolo_merged_path, "images", "train")):
        YOLO_DIR = yolo_merged_path
        print("✅ 병합된 데이터셋(yolo_dataset_merged) 사용")
        print(f"   - Train: {len([f for f in os.listdir(os.path.join(yolo_merged_path, 'images', 'train')) if f.endswith(('.png', '.jpg', '.jpeg'))])}개")
        print(f"   - Val: {len([f for f in os.listdir(os.path.join(yolo_merged_path, 'images', 'val')) if f.endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(os.path.join(yolo_merged_path, 'images', 'val')) else 0}개")
    elif os.path.exists(yolo_dataset_path) and os.path.exists(os.path.join(yolo_dataset_path, "dataset.yaml")):
        YOLO_DIR = yolo_dataset_path
        print("✅ 수동 라벨링된 yolo_dataset 사용")
    else:
        YOLO_DIR = yolo_multiclass_path
        print("⚠️ yolo_dataset을 찾을 수 없어 yolo_multiclass 사용")
    
    TEST_IMG_DIR = os.path.join(BASE, "test_images")
    CATEGORY_MAPPING = os.path.join(BASE, "category_mapping.json")
    
    # 경로 확인 출력
    print(f"📁 프로젝트 루트 경로: {BASE}")
    print(f"📁 YOLO 데이터셋 경로: {YOLO_DIR}")
    print(f"📁 테스트 이미지 경로: {TEST_IMG_DIR}")
    print(f"📁 카테고리 매핑 파일: {CATEGORY_MAPPING}")
    
    # 명령줄 인자 확인 (--skip-training 또는 --inference-only)
    skip_training = "--skip-training" in sys.argv or "--inference-only" in sys.argv
    # --force-train 옵션이 있으면 강제로 학습
    force_train = "--force-train" in sys.argv
    
    # 시드 고정
    set_seed(42)
    
    # 디바이스 설정
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {'GPU' if device == 0 else 'CPU'}")
    
    # 기존 모델 찾기
    existing_model = find_existing_model(BASE)
    
    # 모델 학습 또는 기존 모델 사용
    if force_train:
        print("\n🔄 --force-train 옵션: 강제로 새로 학습합니다.")
        skip_training = False
    elif skip_training and existing_model:
        # --skip-training 옵션이 있고 기존 모델이 있는 경우만 사용
        print(f"\n✅ 기존 모델 발견: {existing_model}")
        print("--skip-training 옵션에 따라 기존 모델을 사용합니다.")
        best_model_path = existing_model
        skip_training = True
        # 모델 이름 추출 (경로에서 자동 추출)
        model_name = extract_model_name_from_path(best_model_path)
    elif existing_model:
        # 기존 모델이 있지만 제대로 학습되지 않았을 수 있으므로 경고
        print(f"\n⚠️ 기존 모델 발견: {existing_model}")
        print("⚠️ 경고: 이 모델은 레이블이 거의 없는 상태에서 학습되었을 수 있습니다.")
        print("⚠️ 새로 학습하는 것을 권장합니다. (--force-train 옵션 사용)")
        print("\n💡 기본적으로 새로 학습을 시작합니다.")
        print("   (기존 모델을 사용하려면 --skip-training 옵션을 사용하세요)")
        skip_training = False
    elif skip_training:
        # --skip-training 옵션이 있지만 모델이 없는 경우
        print("⚠️ --skip-training 옵션이 있지만 기존 모델을 찾을 수 없습니다.")
        print("학습을 시작합니다.")
        skip_training = False
    else:
        # 기존 모델도 없고 옵션도 없는 경우 → 학습 시작
        print("\n💡 기존 모델을 찾을 수 없습니다. 학습을 시작합니다.")
        skip_training = False
    
    if not skip_training:
        # 1. 모델 학습
        print("\n=== 모델 학습 시작 ===")
        train_result = train_improved_model(BASE, YOLO_DIR, device=device, epochs=50)
        best_model_path = train_result["model_path"]
        model_name = train_result["model_name"]
        
        print(f"\n✅ 학습 완료! 모델 저장 위치: {best_model_path}")
        
        # 2. 검증
        print("\n=== 모델 검증 ===")
        if os.path.exists(best_model_path):
            metrics = validate_model(best_model_path, os.path.join(YOLO_DIR, "dataset.yaml"), device)
            
            # 학습 결과 요약 저장
            summary_path = os.path.join(BASE, f"training_summary_{model_name}.csv")
            save_training_summary(best_model_path, metrics, summary_path)
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {best_model_path}")
            print("학습이 완료되지 않았거나 경로가 잘못되었습니다.")
            exit(1)
    else:
        # 기존 모델 사용 시 검증은 선택사항
        print("\n💡 기존 모델을 사용합니다. 검증은 건너뜁니다.")
        print(f"   모델 경로: {best_model_path}")
    
    # 3. 제출 파일 생성
    print("\n=== 제출 파일 생성 ===")
    
    # 테스트 이미지 디렉토리 확인
    if not os.path.exists(TEST_IMG_DIR):
        print(f"⚠️ 테스트 이미지 디렉토리가 없습니다: {TEST_IMG_DIR}")
        print("테스트 이미지를 준비한 후 다시 실행하세요.")
    else:
        # 제출 파일 경로 (버전 번호 자동 추가)
        base_filename = f"kaggle_submission_{model_name}"
        
        # 기존 파일에서 최대 버전 번호 찾기 (두 가지 패턴 모두 확인)
        pattern1 = os.path.join(BASE, f"{base_filename}_ver*.csv")
        pattern2 = os.path.join(BASE, "kaggle_submission_ver*.csv")
        existing_files = glob.glob(pattern1) + glob.glob(pattern2)
        
        # ver 뒤의 숫자 추출
        max_version = 0
        for file in existing_files:
            filename = os.path.basename(file)
            # kaggle_submission_pill_yolo_improved_ver1.csv 또는 kaggle_submission_ver2.csv 형식에서 숫자 추출
            match = re.search(r'_ver(\d+)\.csv$', filename)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)
        
        # 다음 버전 번호
        next_version = max_version + 1
        output_path = os.path.join(BASE, f"{base_filename}_ver{next_version}.csv")
        
        print(f"테스트 이미지 디렉토리: {TEST_IMG_DIR}")
        print(f"출력 파일: {output_path} (버전 {next_version})")
        
        try:
            # 개선된 파라미터로 제출 파일 생성
            # 기존 모델이 제대로 학습되지 않았을 수 있으므로 confidence threshold를 낮춤
            df = generate_submission(
                best_model_path,
                TEST_IMG_DIR,
                CATEGORY_MAPPING,
                output_path,
                conf_threshold=0.25,     # 0.5 → 0.25 (낮춰서 예측 확인)
                use_tta=False,           # TTA 비활성화
                iou_threshold=0.5,      # NMS IoU threshold
                max_det=300             # 최대 검출 개수
            )
            
            # 결과 요약 출력
            print(f"\n📊 제출 파일 요약:")
            print(f"  - 총 예측 개수: {len(df)}")
            print(f"  - 고유 이미지 수: {df['image_id'].nunique()}")
            print(f"  - 고유 카테고리 수: {df['category_id'].nunique()}")
            print(f"  - 평균 Confidence: {df['score'].mean():.4f}")
            print(f"  - 최소 Confidence: {df['score'].min():.4f}")
            print(f"  - 최대 Confidence: {df['score'].max():.4f}")
            
        except Exception as e:
            print(f"❌ 제출 파일 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("✅ 모든 작업 완료!")
    print("="*50)

