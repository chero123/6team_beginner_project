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
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

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
    
    # YOLOv8l 모델 사용 (더 큰 모델, 더 높은 정확도)
    model = YOLO("yolov8l.pt")
    
    # 개선된 하이퍼파라미터로 학습
    results = model.train(
        data=dataset_yaml,
        
        # 모델 설정
        epochs=epochs,              # 20 → 50 (더 충분한 학습)
        imgsz=800,                 # 640 → 800 (작은 객체 검출 개선)
        batch=8,                   # GPU 메모리에 맞게 조정
        device=device,
        name=model_name,
        project=base_dir,          # 프로젝트 디렉토리 명시적으로 지정 (경로 문제 해결)
        
        # 학습률 설정
        lr0=0.001,                 # 초기 학습률 (더 낮게 시작)
        lrf=0.01,                  # 최종 학습률 비율
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation (약 이미지에 최적화)
        hsv_h=0.015,               # 색조 변화
        hsv_s=0.7,                 # 채도 변화 (약의 색상 다양성 반영)
        hsv_v=0.4,                 # 명도 변화
        degrees=10,                 # 회전 각도 (5 → 10)
        translate=0.1,             # 이동 (0.05 → 0.1)
        scale=0.5,                 # 크기 변화
        shear=5,                   # 전단 변환 추가
        perspective=0.0001,         # 원근 변환 추가
        fliplr=0.5,                # 좌우 반전
        flipud=0.0,                # 상하 반전 (약 이미지에는 부적절)
        mosaic=1.0,                # Mosaic augmentation (0.7 → 1.0)
        mixup=0.1,                 # Mixup augmentation (0.05 → 0.1)
        copy_paste=0.1,            # Copy-paste augmentation 추가
        
        # 학습 설정
        patience=15,               # Early stopping patience
        save=True,
        save_period=10,            # 10 epoch마다 체크포인트 저장
        val=True,
        plots=True,
        
        # 재현성
        seed=42,
        deterministic=True,
        
        # 기타
        workers=0,                 # Windows 멀티프로세싱 문제 해결 (0 = 메인 프로세스만 사용)
        amp=True,                  # Automatic Mixed Precision (속도 향상)
        fraction=1.0,              # 전체 데이터셋 사용
        profile=False,
        freeze=None,
        
        # Loss 가중치
        box=7.5,                   # Box loss 가중치
        cls=0.5,                   # Classification loss 가중치
        dfl=1.5,                   # Distribution Focal Loss 가중치
        
        # NMS 설정
        iou=0.7,                   # NMS IoU threshold
        conf=0.25,                 # Confidence threshold
        max_det=300,               # 최대 검출 개수
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
        imgsz=800,
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
        imgsz=800,
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
                imgsz=800,
                conf=conf_threshold,      # Confidence threshold
                iou=iou_threshold,        # NMS IoU threshold
                max_det=max_det,          # 최대 검출 개수
                verbose=False
            )[0]
        
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


def find_existing_model(base_dir, model_name="pill_yolo_improved"):
    """
    기존에 학습된 모델 파일을 찾기
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
        model_name: 모델 이름
    
    Returns:
        모델 경로 또는 None
    """
    # 여러 가능한 경로 확인
    possible_paths = []
    
    # runs/detect 아래의 여러 모델 이름 변형 확인
    model_name_variants = [model_name, f"{model_name}2", f"{model_name}3", f"{model_name}_2"]
    
    # base_dir 기준 경로
    for variant in model_name_variants:
        possible_paths.append(
            os.path.join(base_dir, "runs", "detect", variant, "weights", "best.pt")
        )
        possible_paths.append(
            os.path.join(base_dir, variant, "weights", "best.pt")
        )
    
    # 현재 작업 디렉토리 기준 경로
    for variant in model_name_variants:
        possible_paths.append(
            os.path.join(os.getcwd(), "runs", "detect", variant, "weights", "best.pt")
        )
    
    # 상위 디렉토리에서도 검색
    parent_dirs = [
        os.path.dirname(base_dir),
        os.path.dirname(os.getcwd()),
    ]
    for parent_dir in parent_dirs:
        if parent_dir and os.path.exists(parent_dir):
            for variant in model_name_variants:
                possible_paths.append(
                    os.path.join(parent_dir, "runs", "detect", variant, "weights", "best.pt")
                )
    
    # 실제 존재하는 경로 찾기
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


if __name__ == "__main__":
    import sys
    
    # 경로 설정 (Windows 환경에 맞게 수정 필요)
    BASE = r"D:/스프린트AI엔지니어 부트캠프/part2_kaggle/6team_beginner_project"
    YOLO_DIR = os.path.join(BASE, "yolo_multiclass")
    TEST_IMG_DIR = os.path.join(BASE, "test_images")
    CATEGORY_MAPPING = os.path.join(BASE, "category_mapping.json")
    
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
    elif existing_model:
        # 기존 모델이 있으면 자동으로 사용 (학습 건너뛰기)
        print(f"\n✅ 기존 모델 발견: {existing_model}")
        print("학습을 건너뛰고 기존 모델을 사용합니다.")
        print("   (새로 학습하려면 --force-train 옵션을 사용하세요)")
        best_model_path = existing_model
        skip_training = True  # 명시적으로 설정
        # 모델 이름 추출 (경로에서)
        model_name = "pill_yolo_improved"  # 기본값
        for variant in ["pill_yolo_improved3", "pill_yolo_improved2", "pill_yolo_improved"]:
            if variant in best_model_path:
                model_name = variant
                break
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
        # 제출 파일 경로 (모델 이름 포함)
        output_path = os.path.join(BASE, f"kaggle_submission_{model_name}.csv")
        
        print(f"테스트 이미지 디렉토리: {TEST_IMG_DIR}")
        print(f"출력 파일: {output_path}")
        
        try:
            # 개선된 파라미터로 제출 파일 생성
            # conf_threshold를 0.5로 높여 False Positive 감소
            # use_tta를 False로 설정 (TTA는 때때로 성능을 떨어뜨림)
            # iou_threshold를 0.5로 설정하여 더 엄격한 NMS
            df = generate_submission(
                best_model_path,
                TEST_IMG_DIR,
                CATEGORY_MAPPING,
                output_path,
                conf_threshold=0.5,      # 0.25 → 0.5 (더 높은 정확도)
                use_tta=False,           # TTA 비활성화 (성능 개선)
                iou_threshold=0.5,      # 0.7 → 0.5 (더 엄격한 NMS)
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

