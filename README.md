이 프로젝트는 알약 객체 탐지(Object Detection)를 목표로 하며, dl_idx를 유일한 클래스 기준으로 사용하는 YOLOv8 파이프라인입니다.

> 핵심 원칙  
> - cls 기준 분류 X  
> - dl_idx 1:1 매핑만 사용 O  
> - JSON / YOLO txt 혼합 데이터 → dl_idx 기준 통합  
> - 증강 데이터는 train 전용, val에는 절대 포함하지 않음

## 프로젝트 구조 (project_ver3)

project_ver3/

├── file_py/ # STEP별 실행 스크립트

│ ├── step01-1_make_dlidx_mapping.py

│ ├── step01-2_make_coco_from_dlidx.py

│ ├── step02_coco_to_yolo_symlink.py

│ ├── step03-1_yolo_class_stats.py

│ ├── step03-2_augment_rare_classes.py

│ ├── step03-3_merge_augmented.py

│ ├── step03-4_make_train_val_split.py

│ ├── step04-1_sanity_check_yolo.py

│ ├── step04-2_make_data_yaml.py

│ ├── step04-3_train_yolo.py

│ ├── step05-1_finetune_freeze.py

│ ├── step05-2_finetune_unfreeze.py

│ ├── step05-3_finetune_1152.py

│ ├── step05-4_noise_adapt.py

│ ├── step06-1_predict_and_visualize.py

│ └── step06-2_make_submit_csv.py

│

├── mappings/

│ ├── dlidx_to_trainid.json

│ └── trainid_to_dlidx.json

│

├── coco/


│ └── train_coco_dlidx.json

│

├── work/

│ └── yolo/

│ ├── images/ (all / aug / train / val)

│ ├── labels/ (all / aug / train / val)

│ └── data.yaml

│

├── runs/ # 학습 결과 (git 미포함)

├── .gitignore

└── README.md

## STEP 01 – dl_idx 매핑 생성

### STEP 01-1
모든 JSON + YOLO txt + category_id_mapping.json을 스캔하여 dl_idx → train_id (0-based) 매핑 생성
총 118개의 dl_idx확인

python file_py/step01-1_make_dlidx_mapping.py

### STEP 01-2
dl_idx 기준으로 COCO 생성 (불량 bbox 제거)

python file_py/step01-2_make_coco_from_dlidx.py

## STEP 02 – COCO → YOLO 변환 (고속, symlink 방식)
JSON 기준 이미지 중 bbox 있는 것만 변환
이미지 파일은 symlink 사용 (복사 X)

python file_py/step02_coco_to_yolo_symlink.py

## STEP 03 – 클래스 불균형 처리
### STEP 03-1
YOLO label 기준 클래스 분포 분석

python file_py/step03-1_yolo_class_stats.py

### STEP 03-2
희소 클래스 증강 (train 전용)

python file_py/step03-2_augment_rare_classes.py

### STEP 03-3
원본 + 증강 데이터 pool 병합 (all)

python file_py/step03-3_merge_augmented.py

### STEP 03-4
Train / Val split (Val = 10%, 증강 데이터 제외)

python file_py/step03-4_make_train_val_split.py

## STEP 04 – 학습 전 검증 + Baseline 학습
### STEP 04-1
데이터 sanity check

python file_py/step04-1_sanity_check_yolo.py
확인 항목:
이미지 / 라벨 누락
bbox / cls 이상 여부
val에 aug 포함 여부

### STEP 04-3
YOLOv8 baseline 학습

python file_py/step04-2_train_yolo.py
설정 요약:
imgsz: 896
epochs: 최대 120 (EarlyStopping 적용)
optimizer: SGD

## STEP 05 – 다단계 파인튜닝
Freeze 학습 (1024)
- baseline 모델의 backbone(처음 10개 레이어)을 freeze=10으로 동결
- COCO에서 학습된 저수준 특징(엣지, 텍스처)을 유지하며 head 부분만 데이터셋에 적응
- lr0=0.002, mosaic=0.0으로 과적합 방지하며 1024x1024 해상도에서 40 epochs 학습(EarlyStopping 적용)

Unfreeze 학습 (1024)
- 첫 번째 결과 모델을 불러와 freeze 없이 전체 레이어를 학습
- 학습률을 lr0=0.001로 낮춰 안정적으로 세밀 조정
- 동일한 1024 해상도에서 30 epochs 진행(EarlyStopping 적용)

고해상도 파인튜닝 (1152)
- 두 단계 모델을 1152x1152 고해상도로 학습
- batch=6으로 메모리 최적화
- 첫 부분(12 epochs, lr0=0.0007)은 기본 튜닝, 후반(15 epochs, lr0=0.0005)은 현실 대응 증강(hsv_*, erasing=0.6, translate=0.1 등)을 활성화해 노이즈/변형에 강인하게 만듭니다.
- mosaic/mixup=0.0으로 고해상도에서 불안정성을 피합니다.

Noise / domain adaptation (진행중)
- 고해상도(1152x1152) 모델에 현실 세계 노이즈와 변형을 견디는 robustness를 강화
- ver3_finetune_1152_final 모델을 기반으로 매우 낮은 학습률(lr0=0.0005, SGD optimizer)과 epochs=15, patience=5로 과적합 없이 세밀 조정
- mosaic=0.0, mixup=0.0, copy_paste=0.0 등 구조적 데이터 변형을 모두 끄고, 고해상도에서 발생하는 artifact를 방지
- YOLOv8 공식 지원 증강만 활성화하여 실제 환경 대응력을 높입니다.
  - 색상/조명 변형: hsv_h=0.02, hsv_s=0.5, hsv_v=0.4
  - 물리적 변형: erasing=0.6(부분 가림), translate=0.1(위치 이동), scale=0.5(크기 변화), degrees=5.0(미세 회전), perspective=0.0005(원근 왜곡)
  - 기타: fliplr=0.5(좌우 반전)
- box=7.5, cls=0.5, dfl=1.5로 bbox/클래스 예측을 강조하며, 결과는 ver3_finetune_1152_noise로 저장

## STEP 06 – 테스트 추론 & 제출 파일 생성
python file_py/step06_infer_test.py
python file_py/step06-2_make_kaggle_csv.py

image_id = 테스트 이미지 파일명
annotation_id 순차 증가
category_id = dl_idx
이미지당 최대 3~4 bbox 제한
conf=0.6 이상으로 설정(애매하게 잘못 예측한 것 안나오게)

## 핵심 인사이트
cls 기반 파이프라인은 서로 다른 dl_idx를 하나로 묶는 치명적 문제 발생
dl_idx 단일 기준 + 증강 분리 + 고해상도 파인튜닝이 가장 안정적
초기 mAP이 높은 경우 → 데이터 중복 / val 누수 반드시 점검

---

## Project_ver4 – 정석 YOLO 파이프라인 (SAFE COCO 구조)
### 프로젝트 개요
project_ver4 프로젝트는 기존 project_ver3에서 발견된 구조적 위험 요소를 제거하고, COCO → YOLO 정석 구조에 맞게 파이프라인을 재설계한 버전입니다.

### 구조적 반성 & 개선점 (project_ver3 → project_ver4)

문제 요약
- project_ver3 초기 파이프라인에서는 COCO/YOLO의 “정석 구조(1 이미지 = 1 image_id, 그 이미지의 모든 객체를 한 라벨 파일에 포함)” 원칙이 흔들릴 수 있는 위험이 있었습니다.
- 특히 라벨 JSON이 ‘약(클래스) 단위로 분리되어 저장’된 데이터 특성 때문에, JSON 단위로 이미지를 등록하면 동일한 픽셀 이미지가 여러 image_id로 분리될 수 있고, 이후 YOLO 변환 단계에서 file_name 기반 라벨(txt) 작성 시 덮어쓰기/소실 문제가 발생할 가능성이 있었습니다. 
- 또한 oversampling(복사 기반 증강)을 수행할 경우, 구조적 오류가 존재하면 그 오류가 반복/증폭될 위험이 있었습니다.

> 목표  
> - COCO 철학(1 image = 모든 객체)을 지키는 정석 데이터 구조
> - dl_idx 기반 클래스 일관성 유지
> - Kaggle 점수뿐 아니라 재현성·안정성·확장성까지 확보

### 핵심 원칙

- 1 image = 1 image_id
- 한 이미지에 존재하는 모든 객체(annotation)는 하나의 image_id에 병합
- dl_idx → train_id 매핑은 전역에서 단 1회만 생성
- JSON 기준으로 이미지를 쪼개지 않음
- file_name 기준 라벨 덮어쓰기 구조 제거

## ver3 vs ver4 구조 비교 (핵심 차이)
### project_ver3 (위험 구조)
같은 PNG 이미지

├─ JSON A → image_id 1 → bbox 1개

├─ JSON B → image_id 2 → bbox 1개

├─ JSON C → image_id 3 → bbox 1개


문제점
- 같은 픽셀 이미지가 여러 image_id로 학습됨
- “한 이미지에 여러 객체가 동시에 존재한다”는 현실 구조를 학습 못 함
- YOLO 변환 시 file_name 기준 라벨 덮어쓰기 가능
- oversampling 시 오류 구조가 증폭될 위험
- Kaggle 점수는 나올 수 있으나, 구조적으로 매우 위험

### project_ver4 (정석 구조)
같은 PNG 이미지

└─ image_id 1

├─ bbox 1 (dl_idx A)

├─ bbox 2 (dl_idx B)

├─ bbox 3 (dl_idx C)

개선점
- COCO 철학 완전 준수
- 모든 annotation이 정확히 병합
- YOLO 변환 시 덮어쓰기/소실 불가
- 증강/통계/분할 단계에서 구조 안정성 확보

## 전체 파이프라인 단계 (ver4)

### STEP 01 – 클래스 매핑 & COCO 생성 (SAFE)
```bash
# dl_idx → train_id 매핑 (annotation 기준 포함)
python step01-1_make_dlidx_mapping_FIXED118.py

# file_name 기준 이미지 병합 COCO 생성
python step01-2_make_coco_SAFE_FIXED118.py
```

### STEP 02 – COCO → YOLO 변환 (SAFE)
```bash
python file_py/step02_coco_to_yolo_SAFE.py
```
- 동일 file_name에 대해 라벨 덮어쓰기 없음
- image_id 기준 정확안 bbox 생성

### STEP 03 – 통계 & 희소 클래스 증강
```bash
# YOLO 클래스 분포 분석
python file_py/step03_1_yolo_class_stats.py

# 희소 클래스 증강 (albumentations)
python file_py/step03_2_augment_rare_classes.py

# 원본 + 증강 pool 병합
python file_py/step03_3_merge_augmented_pool.py
```
### STEP 03-4 – Train / Val Split (aug 제외 val)
```bash
python file_py/step03_4_make_train_val_split_noaugval.py
```
- val에는 증강 이미지 절대 포함 안 함
- Data leakage 방지

### STEP 04 – 학습 준비 & 베이스라인 학습
```bash
# data.yaml 생성
python file_py/step04_0_make_data_yaml.py

# sanity check (bbox / cls / 누락 검증)
python file_py/step04_1_sanity_check_yolo.py

# YOLOv8 baseline 학습
python file_py/step04_2_train_yolo.py
```

### STEP 05 – 파인튜닝 (ver3 기준 계승)
```bash
# 1) Backbone freeze
python file_py/step05_1_finetune_freeze.py

# 2) Unfreeze fine-tune
python file_py/step05_2_finetune_unfreeze.py

# 3) High-res (1152) fine-tune
python file_py/step05_3_finetune_1152.py

# 4) Noise adaptation (현실 환경 대응)
python file_py/step05_4_noise_adapt.py
```

### STEP 06 – 추론 & Kaggle 제출
```bash
# 테스트 이미지 시각화
python file_py/step06_1_predict_and_visualize.py

# Kaggle 제출 CSV 생성
python file_py/step06_2_make_submit_csv.py
```
제출 규칙
- image_id = 테스트 이미지 파일명 (확장자 제거)
- annotation_id = 순차 증가
- category_id = dl_idx
- 이미지당 최대 3~4 bbox 제한
- conf threshold 적용 (예: 0.6)

## 구조적 반성 & 개선점
- project_ver3에서는 Json 단위로 image_id를 생성하면서 같은 이미지가 여러 학습 샘플로 분리되는 구조적 문제가 존재했습니다.
- 이로 인해 COCO 철학(1 image = 모든 객체)이 깨졌고, YOLO 변화 단계에서 라벨 덮어쓰기 및 annotation 소실 가능성이 생겼습니다.
- project_ver4에서는 file_name 기준으로 이미지를 병합하여 supervision 구조를 복원하였고, 장기적으로 재사용, 확장 가능한 파이프라인으로 개선하였습니다.