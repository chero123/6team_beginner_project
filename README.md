이 프로젝트는 알약 객체 탐지(Object Detection)를 목표로 하며, dl_idx를 유일한 클래스 기준으로 사용하는 YOLOv8 파이프라인입니다.

> 핵심 원칙  
> - cls 기준 분류 X  
> - dl_idx 1:1 매핑만 사용 O  
> - JSON / YOLO txt 혼합 데이터 → dl_idx 기준 통합  
> - 증강 데이터는 train 전용, val에는 절대 포함하지 않음

---

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