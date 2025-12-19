# project_ver4/file_py/step01_2_make_coco_SAFE_FINAL.py
import os
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict

# =========================
# PATH CONFIG
# =========================
PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"

JSON_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터"

MAP_JSON = os.path.join(PROJECT_ROOT, "mappings", "dlidx_to_trainid.json")

OUT_DIR  = os.path.join(PROJECT_ROOT, "coco")
OUT_JSON = os.path.join(OUT_DIR, "train_coco_trainid_SAFE.json")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD MAPPING
# =========================
with open(MAP_JSON, "r", encoding="utf-8") as f:
    dl2tid = {int(k): int(v) for k, v in json.load(f).items()}

# =========================
# UTIL
# =========================
def clamp_bbox(x, y, w, h, W, H):
    """
    bbox를 이미지 경계 내로 안전하게 보정
    COCO format: [x, y, w, h]
    """
    if w <= 1 or h <= 1:
        return None

    x1 = max(0.0, min(float(x), W - 1))
    y1 = max(0.0, min(float(y), H - 1))
    x2 = max(1.0, min(float(x + w), W))
    y2 = max(1.0, min(float(y + h), H))

    nw = x2 - x1
    nh = y2 - y1
    if nw <= 1 or nh <= 1:
        return None

    return [x1, y1, nw, nh]

# =========================
# SCAN JSON & MERGE BY FILE_NAME
# =========================
json_paths = glob(os.path.join(JSON_ROOT, "**", "*.json"), recursive=True)
print(f"[STEP 01-2 SAFE FINAL] JSON files: {len(json_paths)}")

img_meta = {}                  # image_key -> {file_name, width, height}
ann_pool = defaultdict(list)   # image_key -> list of annotations
stats = defaultdict(int)

for jp in tqdm(json_paths, desc="Build COCO SAFE"):
    try:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        stats["skip_load_err"] += 1
        continue

    images = data.get("images", [])
    anns   = data.get("annotations", [])

    if not images or not anns:
        stats["skip_empty_json"] += 1
        continue

    im = images[0]
    raw_name = im.get("file_name")
    W = im.get("width")
    H = im.get("height")

    if not raw_name or not W or not H:
        stats["skip_bad_image_meta"] += 1
        continue

    # 🔑 image_key: file_name 기준 병합
    image_key = os.path.normpath(raw_name).replace("\\", "/")

    if image_key not in img_meta:
        img_meta[image_key] = {
            "file_name": os.path.basename(image_key),
            "width": int(W),
            "height": int(H),
        }

    for a in anns:
        bbox = a.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            stats["skip_bad_bbox_format"] += 1
            continue

        try:
            dl = int(a.get("category_id"))
        except:
            stats["skip_bad_dlidx"] += 1
            continue

        if dl not in dl2tid:
            stats["skip_unmapped_dlidx"] += 1
            continue

        cb = clamp_bbox(bbox[0], bbox[1], bbox[2], bbox[3], W, H)
        if cb is None:
            stats["skip_bad_bbox_value"] += 1
            continue

        ann_pool[image_key].append({
            "bbox": cb,
            "category_id": dl2tid[dl],
        })
        stats["ann_added"] += 1

# =========================
# COCO EXPORT
# =========================
coco = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [],
}

# categories (train_id 기준)
for dl, tid in sorted(dl2tid.items(), key=lambda x: x[1]):
    coco["categories"].append({
        "id": tid,
        "name": f"dl_{dl}",
        "supercategory": "pill",
    })

img_id = 1
ann_id = 1

for image_key, meta in img_meta.items():
    anns = ann_pool.get(image_key)
    if not anns:
        continue

    coco["images"].append({
        "id": img_id,
        "file_name": meta["file_name"],
        "width": meta["width"],
        "height": meta["height"],
    })

    for a in anns:
        x, y, w, h = a["bbox"]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": a["category_id"],
            "bbox": [x, y, w, h],
            "area": float(w * h),
            "iscrowd": 0,
        })
        ann_id += 1

    img_id += 1

# =========================
# SAVE
# =========================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print("\n[DONE] STEP 01-2 SAFE FINAL")
print(f" - images      : {len(coco['images'])}")
print(f" - annotations : {len(coco['annotations'])}")
print(f" - classes     : {len(coco['categories'])}")
print(f" - saved to    : {OUT_JSON}")

print("\n[STATS]")
for k in sorted(stats.keys()):
    print(f"  {k}: {stats[k]}")