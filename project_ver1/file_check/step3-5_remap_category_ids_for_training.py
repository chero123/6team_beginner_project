import os, json
from collections import defaultdict

UNIFIED = "/home/ohs3201/work/step2_unified_coco/unified.json"
TRAIN_IN = "/home/ohs3201/work/step4_runs/train.json"
VAL_IN   = "/home/ohs3201/work/step4_runs/val.json"
OUT_DIR  = "/home/ohs3201/work/step4_runs_remap"
os.makedirs(OUT_DIR, exist_ok=True)

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

unified = load(UNIFIED)

# 1) unified의 category 원래 id 목록 (예: 1899, 3350...) -> 연속 id(1..K)로 매핑
orig_ids = sorted([c["id"] for c in unified.get("categories", [])])
id_map = {oid: i+1 for i, oid in enumerate(orig_ids)}  # 1..K
K = len(orig_ids)

# 새 categories 생성 (id만 바꾸고 name 유지)
new_categories = []
for c in unified.get("categories", []):
    new_categories.append({
        "id": id_map[c["id"]],
        "name": c.get("name", str(c["id"])),
        "supercategory": c.get("supercategory", "")
    })
new_categories = sorted(new_categories, key=lambda x: x["id"])

def remap(split_json):
    # images 그대로
    out = {
        "images": split_json["images"],
        "annotations": [],
        "categories": new_categories
    }
    bad = 0
    for a in split_json["annotations"]:
        oid = a.get("category_id", None)
        if oid not in id_map:
            bad += 1
            continue
        na = dict(a)
        na["category_id"] = id_map[oid]
        out["annotations"].append(na)
    return out, bad

train = load(TRAIN_IN)
val = load(VAL_IN)

train_out, bad_train = remap(train)
val_out, bad_val = remap(val)

# 2) 검증: category_id 범위 체크
def check(out, name):
    mx = max([a["category_id"] for a in out["annotations"]], default=-1)
    mn = min([a["category_id"] for a in out["annotations"]], default=999999)
    print(f"[{name}] anns={len(out['annotations'])}, cat_id range: {mn}..{mx}, K={K}")

check(train_out, "TRAIN")
check(val_out, "VAL")
print(f"[DROP] train anns removed (unknown category_id): {bad_train}")
print(f"[DROP] val   anns removed (unknown category_id): {bad_val}")

# 3) 저장
train_p = os.path.join(OUT_DIR, "train_remap.json")
val_p   = os.path.join(OUT_DIR, "val_remap.json")
map_p   = os.path.join(OUT_DIR, "category_id_remap.json")

with open(train_p, "w", encoding="utf-8") as f:
    json.dump(train_out, f, ensure_ascii=False)
with open(val_p, "w", encoding="utf-8") as f:
    json.dump(val_out, f, ensure_ascii=False)
with open(map_p, "w", encoding="utf-8") as f:
    json.dump({"orig_to_train_id": id_map, "K": K}, f, ensure_ascii=False, indent=2)

print("\n[STEP 3.5 DONE]")
print("Saved:", train_p)
print("Saved:", val_p)
print("Saved:", map_p)