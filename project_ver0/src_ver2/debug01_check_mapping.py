import os
import json
from collections import Counter

BASE = "/home/ohs3201/6team_beginner_project"
COCO_TRAIN = f"{BASE}/yolo_dataset/coco/train.json"
MAP_PATH = f"{BASE}/category_mapping.json"

print("📌 debug01_check_mapping.py 시작")

with open(MAP_PATH, "r") as f:
    mp = json.load(f)

cat2yolo = {int(k): int(v) for k, v in mp["cat2yolo"].items()}
yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}

print(f"- cat2yolo 개수: {len(cat2yolo)}")
print(f"- yolo2cat 개수: {len(yolo2cat)}")

# 역매핑 일관성 체크
ok_roundtrip = True
for orig, idx in cat2yolo.items():
    back = yolo2cat.get(idx, None)
    if back != orig:
        print(f"⚠ roundtrip 불일치: orig={orig} -> idx={idx} -> {back}")
        ok_roundtrip = False

if ok_roundtrip:
    print("✅ cat2yolo / yolo2cat roundtrip 일관성 OK")

# COCO train.json 내부 category / annotation 검사
with open(COCO_TRAIN, "r") as f:
    coco = json.load(f)

ann_cats = [ann["category_id"] for ann in coco["annotations"]]
unique_ann_cats = sorted(set(ann_cats))
print(f"- COCO train.json annotation에 등장하는 category_id 개수: {len(unique_ann_cats)}")
print(f"  (앞 20개): {unique_ann_cats[:20]}")

# cat2yolo key와 비교
missing_in_map = [c for c in unique_ann_cats if c not in cat2yolo]
extra_in_map = [c for c in cat2yolo.keys() if c not in unique_ann_cats]

if missing_in_map:
    print("⚠ cat2yolo에 없는 category_id (COCO에는 있는데 매핑에는 없음):", missing_in_map)
else:
    print("✅ COCO annotation의 모든 category_id가 cat2yolo에 존재")

if extra_in_map:
    print("⚠ COCO에서 안 쓰는데 cat2yolo에는 있는 category_id:", extra_in_map)
else:
    print("✅ cat2yolo key들은 모두 COCO에서 사용됨")

# 카테고리별 등장 빈도 (앞 10개만)
cnt = Counter(ann_cats)
print("\n🧾 COCO train category 등장 빈도 TOP 10:")
for cid, num in cnt.most_common(10):
    print(f"  - cid={cid}: {num} boxes")

print("\n🎉 debug01_check_mapping.py 완료")