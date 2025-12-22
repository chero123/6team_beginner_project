# project_ver4/file_py/step01-2_make_coco_SAFE_FULL_IMAGES.py
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
MAP_JSON  = os.path.join(PROJECT_ROOT, "mappings", "dlidx_to_trainid.json")

OUT_DIR  = os.path.join(PROJECT_ROOT, "coco")
OUT_JSON = os.path.join(OUT_DIR, "train_coco_trainid_SAFE_FULL_IMAGES.json")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD MAPPING (dl_idx -> train_id)
# =========================
with open(MAP_JSON, "r", encoding="utf-8") as f:
    dl2tid = {int(k): int(v) for k, v in json.load(f).items()}

# =========================
# UTIL
# =========================
def norm_name(p):
    """경로 차이 제거용: file_name 정규화"""
    return os.path.basename(p.replace("\\", "/"))

def clamp_bbox(x, y, w, h, W, H):
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
# COLLECT JSON FILES
# =========================
json_paths = glob(os.path.join(JSON_ROOT, "**", "*.json"), recursive=True)
print(f"[SAFE FULL] JSON files: {len(json_paths)}")

# =========================
# GLOBAL CONTAINERS
# =========================
img_id_by_name = {}          # file_name -> global image_id
img_meta_by_id = {}          # image_id -> (W,H,file_name)
ann_pool = defaultdict(list)

next_img_id = 1
next_ann_id = 1

stats = defaultdict(int)

# =========================
# MAIN LOOP
# =========================
for jp in tqdm(json_paths, desc="Build SAFE COCO (FULL IMAGES)"):
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

    # -------------------------
    # 1) local image_id → global image_id
    # -------------------------
    local2global = {}

    for im in images:
        raw_name = im.get("file_name")
        W = im.get("width")
        H = im.get("height")
        local_id = im.get("id")

        if not raw_name or not W or not H or local_id is None:
            stats["skip_bad_image_meta"] += 1
            continue

        fname = norm_name(raw_name)

        if fname not in img_id_by_name:
            gid = next_img_id
            img_id_by_name[fname] = gid
            img_meta_by_id[gid] = (int(W), int(H), fname)
            next_img_id += 1
            stats["image_added"] += 1
        else:
            gid = img_id_by_name[fname]

        local2global[local_id] = gid

    if not local2global:
        continue

    # -------------------------
    # 2) annotations
    # -------------------------
    for an in anns:
        lid = an.get("image_id")
        if lid not in local2global:
            continue

        gid = local2global[lid]
        W, H, _ = img_meta_by_id[gid]

        bbox = an.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            stats["skip_bad_bbox_format"] += 1
            continue

        try:
            dl = int(an.get("category_id"))
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

        ann_pool[gid].append({
            "id": next_ann_id,
            "image_id": gid,
            "category_id": dl2tid[dl],
            "bbox": cb,
            "area": float(cb[2] * cb[3]),
            "iscrowd": 0,
        })
        next_ann_id += 1
        stats["ann_added"] += 1

# =========================
# BUILD COCO
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

# images + annotations
for gid, (W, H, fname) in img_meta_by_id.items():
    anns = ann_pool.get(gid)
    if not anns:
        continue

    coco["images"].append({
        "id": gid,
        "file_name": fname,
        "width": W,
        "height": H,
    })
    coco["annotations"].extend(anns)

# =========================
# SAVE
# =========================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print("\n[DONE] SAFE FULL COCO CREATED")
print(f" - images      : {len(coco['images'])}")
print(f" - annotations : {len(coco['annotations'])}")
print(f" - classes     : {len(coco['categories'])}")
print(f" - saved to    : {OUT_JSON}")

print("\n[STATS]")
for k in sorted(stats.keys()):
    print(f"  {k}: {stats[k]}")