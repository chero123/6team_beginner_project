import json
from collections import Counter
from pycocotools.coco import COCO

BASE = "/home/ohs3201/6team_beginner_project"
COCO_JSON = f"{BASE}/yolo_dataset/coco/train.json"
MAP_JSON = f"{BASE}/category_mapping.json"

print("📌 COCO category 검증 시작\n")

# Load mapping
with open(MAP_JSON, "r") as f:
    mp = json.load(f)

yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}
cat2yolo = {v: k for k, v in yolo2cat.items()}

print(f"총 매핑 클래스 수: {len(yolo2cat)}\n")

# Load COCO
coco = COCO(COCO_JSON)

ann_cat_ids = [ann["category_id"] for ann in coco.dataset["annotations"]]
cat_freq = Counter(ann_cat_ids)

print("📌 COCO에서 실제로 등장한 category_id 개수:", len(cat_freq))
print("📌 상위 등장 category 10개:")
for cid, cnt in cat_freq.most_common(10):
    print(f" - cid={cid}: {cnt} boxes")

print("\n📌 매핑에 없는데 COCO에 등장한 category_id:")
missing_in_mapping = [cid for cid in cat_freq if cid not in cat2yolo]
print(missing_in_mapping)

print("\n📌 COCO에 등장하지 않지만 매핑에는 있는 category:")
unused_categories = [cat for cat in cat2yolo if cat not in cat_freq]
print(unused_categories)

# Round-trip 검증
print("\n📌 Round-trip 검증 (YOLO→cat→YOLO):")
bad_roundtrip = []

for yidx, cid in yolo2cat.items():
    if cat2yolo.get(cid, None) != yidx:
        bad_roundtrip.append((yidx, cid))

if bad_roundtrip:
    print("⚠ 문제 있음:", bad_roundtrip)
else:
    print("✅ Round-trip 매핑 정상")

print("\n🎉 검증 완료!")