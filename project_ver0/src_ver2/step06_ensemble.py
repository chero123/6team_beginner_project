import os
import pandas as pd
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

BASE = "/home/ohs3201/6team_beginner_project"
TEST_DIR = f"{BASE}/data/test_images"

SUBMIT_DIR = f"{BASE}/results/submission/ver2"
os.makedirs(SUBMIT_DIR, exist_ok=True)

# STEP05에서 생성한 CSV
CSV_FRCNN = f"{SUBMIT_DIR}/FasterRCNN_ver2.csv"
CSV_YOLO  = f"{SUBMIT_DIR}/YOLOv8L_continue_ver2.csv"

# WBF 결과
OUT_WBF = f"{SUBMIT_DIR}/final_ensemble_WBF_ver2.csv"

# Kaggle 제출용 (annotation_id 추가)
OUT_SUBMIT = f"{SUBMIT_DIR}/final_ensemble_WBF_submission_ver2.csv"

print("\n==============================")
print(" STEP06 : High-Performance WBF Ensemble (ver2)")
print("==============================")

if not os.path.exists(CSV_FRCNN):
    raise FileNotFoundError(f"❌ {CSV_FRCNN} 없음. Step05 먼저 실행하세요.")
if not os.path.exists(CSV_YOLO):
    raise FileNotFoundError(f"❌ {CSV_YOLO} 없음. Step05 먼저 실행하세요.")

# 1) CSV 로드
df_fr = pd.read_csv(CSV_FRCNN)
df_yo = pd.read_csv(CSV_YOLO)

# 타입 정리
for df in (df_fr, df_yo):
    df["image_id"] = df["image_id"].astype(int)
    df["category_id"] = df["category_id"].astype(int)
    for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]:
        df[col] = pd.to_numeric(df[col])

# 전체 이미지 id 집합
image_ids = sorted(set(df_fr["image_id"].unique()) | set(df_yo["image_id"].unique()))
print(f"🔎 총 이미지 수: {len(image_ids)}")

final_rows = []

# WBF 파라미터
iou_thr = 0.55        # IoU threshold (0.5~0.6 권장)
skip_box_thr = 0.001  # 이보다 낮은 score 박스는 WBF 입력에서 제외
weights = [1.0, 1.0]  # [FRCNN, YOLOv8L] 동일 가중치

# 최종 score 필터링 (WBF 후)
FINAL_SCORE_THR = 0.30

for img_id in image_ids:
    img_filename = f"{img_id}.png"
    img_path = os.path.join(TEST_DIR, img_filename)
    if not os.path.exists(img_path):
        continue

    with Image.open(img_path) as img:
        W, H = img.size

    df_fr_img = df_fr[df_fr["image_id"] == img_id]
    df_yo_img = df_yo[df_yo["image_id"] == img_id]

    fr_bboxes, fr_scores, fr_labels = [], [], []
    yo_bboxes, yo_scores, yo_labels = [], [], []

    # FRCNN 박스들 (0~1로 정규화)
    for _, row in df_fr_img.iterrows():
        x1 = row["bbox_x"]
        y1 = row["bbox_y"]
        x2 = x1 + row["bbox_w"]
        y2 = y1 + row["bbox_h"]
        fr_bboxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
        fr_scores.append(float(row["score"]))
        fr_labels.append(int(row["category_id"]))

    # YOLO 박스들
    for _, row in df_yo_img.iterrows():
        x1 = row["bbox_x"]
        y1 = row["bbox_y"]
        x2 = x1 + row["bbox_w"]
        y2 = y1 + row["bbox_h"]
        yo_bboxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
        yo_scores.append(float(row["score"]))
        yo_labels.append(int(row["category_id"]))

    if len(fr_bboxes) == 0 and len(yo_bboxes) == 0:
        continue

    bboxes_list = [fr_bboxes, yo_bboxes]
    scores_list = [fr_scores, yo_scores]
    labels_list = [fr_labels, yo_labels]

    # WBF 수행
    wb, ws, wl = weighted_boxes_fusion(
        bboxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    for bbox, score, label in zip(wb, ws, wl):
        if score < FINAL_SCORE_THR:
            continue

        x1 = bbox[0] * W
        y1 = bbox[1] * H
        x2 = bbox[2] * W
        y2 = bbox[3] * H

        x1 = max(0, min(W - 1, int(round(x1))))
        y1 = max(0, min(H - 1, int(round(y1))))
        x2 = max(0, min(W - 1, int(round(x2))))
        y2 = max(0, min(H - 1, int(round(y2))))

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        final_rows.append([
            int(img_id),
            int(label),
            x1, y1, w, h,
            float(score),
        ])

# DataFrame으로 변환
df_final = pd.DataFrame(
    final_rows,
    columns=[
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ],
)

# 타입 강제
df_final["image_id"] = df_final["image_id"].astype(int)
df_final["category_id"] = df_final["category_id"].astype(int)
for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
    df_final[col] = df_final[col].astype(int)
df_final["score"] = df_final["score"].astype(float)

# score 기준 내림차순 정렬
df_final = df_final.sort_values(by="score", ascending=False)

# 1) WBF 결과만 저장 (annotation_id 없음)
df_final.to_csv(OUT_WBF, index=False)
print(f"\n💾 WBF 결과 CSV 저장 완료 → {OUT_WBF}")
print(f"📝 총 박스 수: {len(df_final)}")

# 2) Kaggle 제출용: annotation_id 추가
df_submit = df_final.copy().reset_index(drop=True)
df_submit.insert(0, "annotation_id", df_submit.index + 1)  # 1부터 시작

df_submit.to_csv(OUT_SUBMIT, index=False)
print(f"💾 최종 Kaggle 제출용 CSV 저장 완료 → {OUT_SUBMIT}")
print(f"📝 총 박스 수(제출): {len(df_submit)}")
print("\n🎉 STEP06 ver2 완료!")