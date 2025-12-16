"""
두 개의 YOLO 데이터셋을 병합하는 스크립트

yolo_dataset_aihub+orig와 yolo_dataset을 병합하여
더 큰 데이터셋으로 학습할 수 있도록 합니다.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import time
import yaml


def merge_yolo_datasets(base_dir, dataset1_dir, dataset2_dir, output_dir):
    """
    두 개의 YOLO 데이터셋을 병합
    
    Args:
        base_dir: 프로젝트 기본 디렉토리
        dataset1_dir: 첫 번째 데이터셋 디렉토리 (yolo_dataset_aihub+orig)
        dataset2_dir: 두 번째 데이터셋 디렉토리 (yolo_dataset)
        output_dir: 출력 디렉토리 (병합된 데이터셋)
    """
    start_time = time.time()
    print("="*60)
    print("YOLO 데이터셋 병합 시작")
    print("="*60)
    
    # 출력 디렉토리 생성
    output_images_train = os.path.join(output_dir, "images", "train")
    output_images_val = os.path.join(output_dir, "images", "val")
    output_labels_train = os.path.join(output_dir, "labels", "train")
    output_labels_val = os.path.join(output_dir, "labels", "val")
    
    for dir_path in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 이미지와 레이블 파일 복사 (중복 제거)
    copied_images = set()
    copied_count = {"train": 0, "val": 0}
    
    datasets = [
        ("Dataset 1 (AIHub+Orig)", dataset1_dir),
        ("Dataset 2 (Original)", dataset2_dir),
    ]
    
    for dataset_name, dataset_dir in datasets:
        print(f"\n{dataset_name} 처리 중...")
        
        # Train 데이터
        train_img_dir = os.path.join(dataset_dir, "images", "train")
        train_label_dir = os.path.join(dataset_dir, "labels", "train")
        
        if os.path.exists(train_img_dir):
            train_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in tqdm(train_images, desc=f"{dataset_name} Train"):
                img_stem = os.path.splitext(img_name)[0]
                
                # 중복 체크 (이미지 이름 기준)
                if img_stem in copied_images:
                    continue
                
                img_src = os.path.join(train_img_dir, img_name)
                label_src = os.path.join(train_label_dir, f"{img_stem}.txt")
                
                # 이미지와 레이블이 모두 있어야 복사
                if os.path.exists(img_src) and os.path.exists(label_src):
                    img_dst = os.path.join(output_images_train, img_name)
                    label_dst = os.path.join(output_labels_train, f"{img_stem}.txt")
                    
                    # 이미 존재하면 스킵
                    if os.path.exists(img_dst) and os.path.exists(label_dst):
                        copied_images.add(img_stem)
                        continue
                    
                    # 파일 복사 (더 빠른 방법)
                    try:
                        shutil.copy2(img_src, img_dst)
                        shutil.copy2(label_src, label_dst)
                        copied_images.add(img_stem)
                        copied_count["train"] += 1
                    except Exception as e:
                        print(f"\n경고: {img_name} 복사 실패 - {e}")
                        continue
        
        # Val 데이터
        val_img_dir = os.path.join(dataset_dir, "images", "val")
        val_label_dir = os.path.join(dataset_dir, "labels", "val")
        
        if os.path.exists(val_img_dir):
            val_images = [f for f in os.listdir(val_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in tqdm(val_images, desc=f"{dataset_name} Val"):
                img_stem = os.path.splitext(img_name)[0]
                
                # 중복 체크
                if img_stem in copied_images:
                    continue
                
                img_src = os.path.join(val_img_dir, img_name)
                label_src = os.path.join(val_label_dir, f"{img_stem}.txt")
                
                if os.path.exists(img_src) and os.path.exists(label_src):
                    img_dst = os.path.join(output_images_val, img_name)
                    label_dst = os.path.join(output_labels_val, f"{img_stem}.txt")
                    
                    # 이미 존재하면 스킵
                    if os.path.exists(img_dst) and os.path.exists(label_dst):
                        copied_images.add(img_stem)
                        continue
                    
                    # 파일 복사
                    try:
                        shutil.copy2(img_src, img_dst)
                        shutil.copy2(label_src, label_dst)
                        copied_images.add(img_stem)
                        copied_count["val"] += 1
                    except Exception as e:
                        print(f"\n경고: {img_name} 복사 실패 - {e}")
                        continue
    
    # 최종 통계
    final_train_imgs = len([f for f in os.listdir(output_images_train) if f.endswith(('.png', '.jpg', '.jpeg'))])
    final_train_labels = len([f for f in os.listdir(output_labels_train) if f.endswith('.txt')])
    final_val_imgs = len([f for f in os.listdir(output_images_val) if f.endswith(('.png', '.jpg', '.jpeg'))])
    final_val_labels = len([f for f in os.listdir(output_labels_val) if f.endswith('.txt')])
    
    # dataset.yaml 생성 (기존 yolo_dataset의 dataset.yaml 사용)
    dataset_yaml_src = os.path.join(dataset2_dir, "dataset.yaml")
    dataset_yaml_dst = os.path.join(output_dir, "dataset.yaml")
    
    if os.path.exists(dataset_yaml_src):
        # 기존 dataset.yaml 복사
        with open(dataset_yaml_src, 'r', encoding='utf-8') as f:
            dataset_config = yaml.safe_load(f) or {}
        
        # path 업데이트
        dataset_config['path'] = os.path.abspath(output_dir)
        
        # train, val 경로 확인
        dataset_config['train'] = 'images/train'
        dataset_config['val'] = 'images/val'
        
        # dataset.yaml 저장
        with open(dataset_yaml_dst, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"\n✅ dataset.yaml 생성 완료: {dataset_yaml_dst}")
    else:
        print(f"\n⚠️ {dataset_yaml_src}를 찾을 수 없어 dataset.yaml을 생성하지 않았습니다.")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("병합 완료!")
    print(f"{'='*60}")
    print(f"소요 시간: {elapsed_time:.1f}초")
    print(f"출력 디렉토리: {output_dir}")
    print(f"\nTrain:")
    print(f"  - 이미지: {final_train_imgs}개")
    print(f"  - 레이블: {final_train_labels}개")
    print(f"\nVal:")
    print(f"  - 이미지: {final_val_imgs}개")
    print(f"  - 레이블: {final_val_labels}개")
    print(f"\n총 데이터: {final_train_imgs + final_val_imgs}개")
    
    return output_dir


if __name__ == "__main__":
    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(script_dir)
    
    DATASET1 = os.path.join(BASE, "yolo_dataset_aihub+orig")
    DATASET2 = os.path.join(BASE, "yolo_dataset")
    OUTPUT = os.path.join(BASE, "yolo_dataset_merged")
    
    print(f"BASE: {BASE}")
    print(f"Dataset 1: {DATASET1}")
    print(f"Dataset 2: {DATASET2}")
    print(f"Output: {OUTPUT}\n")
    
    merge_yolo_datasets(BASE, DATASET1, DATASET2, OUTPUT)

