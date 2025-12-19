# file_py/step01-2_make_coco_from_dlidx_SAFE.py
import os
import json
from tqdm import tqdm

JSON_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터"
MAP_JSON  = "/home/ohs3201/6team_beginner_project/project_ver3/mappings/dlidx_to_trainid.json"

OUT_DIR   = "/home/ohs3201/6team_beginner_project/project_ver3/coco"
OUT_JSON  = os.path.join(OUT_DIR, "train_coco_trainid_safe.json")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# load mapping: dl_idx -> train_id
# -------------------------
with open(MAP_JSON, "r", encoding="utf-8") as f:
    dl2tid = {int(k): int(v) for k, v in json.load(f).items()}

# -------------------------
# collect json files
# -------------------------
json_files = []
for root, _, files in os.walk(JSON_ROOT):
    for fn in files:
        if fn.endswith(".json"):
            json_files.append(os.path.join(root, fn))

print(f"[STEP 01-2 SAFE] JSON files found: {len(json_files)}")

# -------------------------
# COCO containers
#   file_name(basename)로 images 유일하게 관리
# -------------------------
coco = {"images": [], "annotations": [], "categories": []}

# categories: train_id 기준
for dl, tid in sorted(dl2tid.items(), key=lambda x: x[1]):
    coco["categories"].append({
        "id": tid,
        "name": f"dl_{dl}",
        "supercategory": "pill",
    })

img_id_by_name = {}   # basename -> image_id
img_info_by_id = {}   # image_id -> (W,H)
ann_id = 1

stats = {
    "skip_load_err": 0,
    "skip_not_coco": 0,
    "skip_img_missing": 0,
    "skip_img_no_wh": 0,
    "skip_ann_bad_bbox": 0,
    "skip_ann_oob": 0,
    "skip_ann_unknown_dlidx": 0,
    "ann_added": 0,
    "img_added": 0,
}

def clamp_bbox(x, y, w, h, W, H):
    """bbox를 이미지 범위 내로 clamp하고 너무 작으면 None"""
    if w <= 1 or h <= 1:
        return None
    # clamp
    x1 = max(0.0, min(float(x), W - 1))
    y1 = max(0.0, min(float(y), H - 1))
    x2 = max(1.0, min(float(x + w), W))
    y2 = max(1.0, min(float(y + h), H))
    nw = x2 - x1
    nh = y2 - y1
    if nw <= 1 or nh <= 1:
        return None
    return [x1, y1, nw, nh]

# (선택) 중복 bbox 제거용: (image_id, cls, round(x),round(y),round(w),round(h))
seen_boxes = set()

for jp in tqdm(json_files):
    try:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        stats["skip_load_err"] += 1
        continue

    images = data.get("images")
    anns   = data.get("annotations")
    if not isinstance(images, list) or not isinstance(anns, list) or len(images) == 0:
        stats["skip_not_coco"] += 1
        continue

    # ✅ 이 JSON이 1장짜리든 여러 장짜리든 'images' 기준으로 처리
    # local image id -> global image id
    local2global = {}

    for im in images:
        file_name = im.get("file_name")
        if not file_name:
            stats["skip_img_missing"] += 1
            continue

        bn = os.path.basename(file_name)
        W = im.get("width")
        H = im.get("height")
        if not W or not H:
            stats["skip_img_no_wh"] += 1
            continue

        if bn not in img_id_by_name:
            gid = len(img_id_by_name) + 1
            img_id_by_name[bn] = gid
            coco["images"].append({
                "id": gid,
                "file_name": bn,
                "width": int(W),
                "height": int(H),
            })
            img_info_by_id[gid] = (int(W), int(H))
            stats["img_added"] += 1
        else:
            gid = img_id_by_name[bn]
            # width/height 불일치 체크(원하면 assert/로그)
            # 여기서는 최초값 유지

        local_id = im.get("id")
        if local_id is not None:
            local2global[local_id] = gid

    if not local2global:
        continue

    # annotations 처리: ✅ ann.category_id(dl_idx) 기반으로 train_id 결정
    for an in anns:
        lid = an.get("image_id")
        if lid not in local2global:
            continue
        gid = local2global[lid]
        W, H = img_info_by_id[gid]

        bbox = an.get("bbox")
        if (not isinstance(bbox, list)) or len(bbox) != 4:
            stats["skip_ann_bad_bbox"] += 1
            continue
        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            stats["skip_ann_bad_bbox"] += 1
            continue

        dl = an.get("category_id")
        try:
            dl = int(dl)
        except:
            stats["skip_ann_unknown_dlidx"] += 1
            continue

        if dl not in dl2tid:
            stats["skip_ann_unknown_dlidx"] += 1
            continue

        tid = dl2tid[dl]

        cb = clamp_bbox(x, y, w, h, W, H)
        if cb is None:
            stats["skip_ann_oob"] += 1
            continue

        # (선택) 중복 제거: 너무 공격적이면 꺼도 됨
        k = (gid, tid, round(cb[0]), round(cb[1]), round(cb[2]), round(cb[3]))
        if k in seen_boxes:
            continue
        seen_boxes.add(k)

        coco["annotations"].append({
            "id": ann_id,
            "image_id": gid,
            "category_id": tid,   # ✅ train_id로 통일
            "bbox": [float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3])],
            "area": float(cb[2] * cb[3]),
            "iscrowd": 0,
        })
        ann_id += 1
        stats["ann_added"] += 1

# save
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print("\n[DONE] STEP 01-2 SAFE completed")
print(f" - images: {len(coco['images'])}")
print(f" - annotations: {len(coco['annotations'])}")
print(f" - classes: {len(coco['categories'])}")
print(f" - saved to: {OUT_JSON}")

print("\n[STATS]")
for k in sorted(stats.keys()):
    print(f"  {k}: {stats[k]}")