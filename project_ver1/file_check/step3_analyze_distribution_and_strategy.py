import os, json, math
from collections import Counter, defaultdict
import numpy as np

# =========================
# CONFIG
# =========================
UNIFIED_JSON = "/home/ohs3201/work/step2_unified_coco/unified.json"  # STEP2 결과 경로
OUT_DIR = "/home/ohs3201/work/step3_stats"
os.makedirs(OUT_DIR, exist_ok=True)

# 자동 전략 파라미터
MIN_SAMPLES_RARE = 30          # 이 미만이면 rare
MIN_SAMPLES_ULTRA_RARE = 10    # 이 미만이면 ultra_rare
CAP_WEIGHT = 20.0              # 샘플러 가중치 상한 (너무 과하게 뽑히는 것 방지)
TEMP = 1.0                     # (count^-TEMP) 형태. TEMP=1이면 역빈도, 0.5면 완만
SMOOTH = 1.0                   # 0 division 방지

# =========================
# LOAD
# =========================
with open(UNIFIED_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])
categories = coco.get("categories", [])

cat_id_to_name = {}
for c in categories:
    cid = int(c["id"])
    cat_id_to_name[cid] = c.get("name", str(cid))

print("[INFO] images:", len(images))
print("[INFO] annotations:", len(annotations))
print("[INFO] categories:", len(categories))

# =========================
# COUNT DISTRIBUTIONS
# =========================
ann_cnt = Counter()
img_has_cat = defaultdict(set)   # image_id -> {category_id}
cat_to_imgs = defaultdict(set)   # category_id -> {image_id}

bad_ann = 0
for a in annotations:
    cid = a.get("category_id", None)
    iid = a.get("image_id", None)
    bbox = a.get("bbox", None)

    # bbox가 YOLO(정규화)로 들어왔을 수도 있고, JSON(px)일 수도 있음
    # 여기선 "존재만" 확인하고 카운트는 한다.
    if cid is None or iid is None or bbox is None:
        bad_ann += 1
        continue

    cid = int(cid)
    iid = int(iid)
    ann_cnt[cid] += 1
    img_has_cat[iid].add(cid)
    cat_to_imgs[cid].add(iid)

# 이미지 기준으로 "대표 클래스"를 하나 정해서(가장 빈도가 낮은 클래스 우선) 샘플러 weight 만들기
#  - detection은 이미지에 여러 클래스가 있을 수 있는데,
#    imbalance 대응을 위해 이미지 weight를 "그 이미지가 가진 클래스들 중 가장 희소한 클래스 기준"으로 잡는다.
cat_img_cnt = {cid: len(s) for cid, s in cat_to_imgs.items()}

# =========================
# SUMMARY TABLE 만들기
# =========================
rows = []
for cid in sorted(cat_id_to_name.keys()):
    a = ann_cnt.get(cid, 0)
    im = cat_img_cnt.get(cid, 0)
    name = cat_id_to_name.get(cid, str(cid))
    rows.append((cid, name, a, im))

# 저장: csv
csv_path = os.path.join(OUT_DIR, "class_distribution.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("category_id,name,ann_count,img_count\n")
    for cid, name, a, im in rows:
        f.write(f"{cid},{name},{a},{im}\n")

# 상위/하위 출력
rows_sorted_ann = sorted(rows, key=lambda x: x[2], reverse=True)
print("\n[TOP 15 by ann_count]")
for r in rows_sorted_ann[:15]:
    print(r[0], r[2], "imgs", r[3], "-", r[1])

print("\n[BOTTOM 15 by ann_count]")
for r in rows_sorted_ann[-15:]:
    print(r[0], r[2], "imgs", r[3], "-", r[1])

# =========================
# IMBALANCE STRATEGY 자동 결정
# =========================
ann_counts = np.array([r[2] for r in rows], dtype=np.float32)
nonzero = ann_counts[ann_counts > 0]
if len(nonzero) == 0:
    raise RuntimeError("❌ No valid annotations found. unified.json이 비어있거나 category_id가 이상합니다.")

median = float(np.median(nonzero))
p10 = float(np.percentile(nonzero, 10))
p90 = float(np.percentile(nonzero, 90))

rare_cats = [cid for cid, _, a, _ in rows if 0 < a < MIN_SAMPLES_RARE]
ultra_rare_cats = [cid for cid, _, a, _ in rows if 0 < a < MIN_SAMPLES_ULTRA_RARE]
zero_cats = [cid for cid, _, a, _ in rows if a == 0]

strategy = {
    "stats": {
        "categories_total": len(rows),
        "categories_with_ann": int((ann_counts > 0).sum()),
        "median_ann_count": median,
        "p10_ann_count": p10,
        "p90_ann_count": p90,
        "rare_threshold": MIN_SAMPLES_RARE,
        "ultra_rare_threshold": MIN_SAMPLES_ULTRA_RARE,
    },
    "lists": {
        "rare_category_ids": rare_cats,
        "ultra_rare_category_ids": ultra_rare_cats,
        "zero_annotation_category_ids": zero_cats,
    },
    "recommendations": {
        # 학습에서 바로 적용 가능한 권장안
        "sampler": {
            "type": "WeightedRandomSampler",
            "image_weight_rule": "min_class_freq_in_image",
            "weight_formula": "w = min( CAP_WEIGHT, (median / (count+SMOOTH))**TEMP )",
            "TEMP": TEMP,
            "SMOOTH": SMOOTH,
            "CAP_WEIGHT": CAP_WEIGHT,
        },
        "augmentation": {
            "rare_classes": [
                "RandomHorizontalFlip",
                "ColorJitter(weak)",
                "RandomAffine(small)",
                "RandomResize(scale 0.8~1.2)"
            ],
            "common_classes": [
                "RandomHorizontalFlip",
                "RandomResize(scale 0.9~1.1)"
            ]
        },
        "training": {
            "note": "Detection은 class_weight loss보다 sampler가 더 효과적인 경우가 많음. rare가 심하면 sampler + strong aug 조합 권장.",
            "if_ultra_rare_many": "ultra_rare가 너무 많으면, 1차는 common+mid만 학습 → 2차 fine-tune에서 rare 샘플러 강하게",
        }
    }
}

strategy_path = os.path.join(OUT_DIR, "imbalance_strategy.json")
with open(strategy_path, "w", encoding="utf-8") as f:
    json.dump(strategy, f, ensure_ascii=False, indent=2)

print("\n[INFO] Strategy saved:", strategy_path)

# =========================
# IMAGE-LEVEL WEIGHTS 생성 (학습에서 즉시 사용)
# =========================
# class_count 기준 weight 계산: 희소 클래스일수록 weight↑
count_by_cid = {cid: ann_cnt.get(cid, 0) for cid in cat_id_to_name.keys()}
def class_weight(cid: int):
    c = float(count_by_cid.get(cid, 0))
    if c <= 0:
        return 0.0
    w = (median / (c + SMOOTH)) ** TEMP
    return float(min(CAP_WEIGHT, w))

# 각 이미지 weight = 그 이미지가 포함한 클래스들 중 "가장 큰 weight" (== 가장 희소한 클래스 기준)
# (희소 클래스가 포함된 이미지를 더 자주 뽑게 됨)
image_id_list = []
image_weight_list = []

valid_img = 0
for img in images:
    iid = int(img["id"])
    cats = list(img_has_cat.get(iid, []))
    if len(cats) == 0:
        continue
    ws = [class_weight(c) for c in cats if class_weight(c) > 0]
    if len(ws) == 0:
        continue
    w_img = max(ws)  # 희소 클래스 기준
    image_id_list.append(iid)
    image_weight_list.append(w_img)
    valid_img += 1

weights_out = {
    "image_ids": image_id_list,
    "weights": image_weight_list,
    "note": "Use these weights to build a WeightedRandomSampler over the TRAIN dataset indices that correspond to these image_ids."
}

weights_path = os.path.join(OUT_DIR, "train_image_weights.json")
with open(weights_path, "w", encoding="utf-8") as f:
    json.dump(weights_out, f, ensure_ascii=False)

print("[INFO] Image weights saved:", weights_path)
print("[INFO] Valid images with weights:", valid_img)

print("\n[STEP 3 DONE]")
print(" - class_distribution.csv")
print(" - imbalance_strategy.json")
print(" - train_image_weights.json")
print("Saved in:", OUT_DIR)