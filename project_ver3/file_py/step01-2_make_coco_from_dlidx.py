# =========================================
# STEP 01-2 (FINAL)
# Make COCO from raw JSONs using dl_idx->train_id mapping
# - category_id = train_id (0-based)
# - preserve orig_dl_idx in each annotation
# - bbox clamp + invalid skip
# =========================================

import os
import json
from collections import defaultdict
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
JSON_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터"

MAP_JSON = (
    "/home/ohs3201/6team_beginner_project/project_ver3/"
    "mappings/dlidx_to_trainid.json"
)

OUT_DIR = "/home/ohs3201/6team_beginner_project/project_ver3/coco"
OUT_JSON = os.path.join(OUT_DIR, "train_coco_dlidx.json")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
MIN_BOX_SIZE = 2.0  # clamp 후 w/h가 이보다 작으면 스킵

# =========================
# UTIL
# =========================
def to_int(x):
    try:
        return int(x)
    except:
        return None

def parse_bbox(b):
    """
    bbox가 다양한 형태로 들어올 수 있어서 방어적으로 파싱.
    정상: [x,y,w,h]
    비정상: len<4, dict, string 등 -> None
    """
    if b is None:
        return None
    if isinstance(b, (list, tuple)):
        if len(b) >= 4:
            try:
                x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                return x, y, w, h
            except:
                return None
        return None
    if isinstance(b, dict):
        # 혹시 {"x":..,"y":..,"w":..,"h":..} 같은 형태
        keys = ["x", "y", "w", "h"]
        if all(k in b for k in keys):
            try:
                return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            except:
                return None
        return None
    return None

def clamp_bbox(x, y, w, h, W, H):
    """
    이미지 경계를 살짝 넘어간 bbox도 clamp로 살려서 쓰기.
    """
    if w <= 0 or h <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(W), x + w)
    y2 = min(float(H), y + h)

    nw = x2 - x1
    nh = y2 - y1
    if nw < MIN_BOX_SIZE or nh < MIN_BOX_SIZE:
        return None

    return x1, y1, nw, nh

def get_image_meta(images):
    """
    JSON 한 파일 안에 images가 여러 개일 수도 있으니,
    일반적으로 1개인 케이스를 우선 처리하되 방어적으로 첫번째만 사용.
    """
    if not isinstance(images, list) or len(images) == 0:
        return None

    im = images[0]
    file_name = im.get("file_name") or im.get("imgfile")
    W = im.get("width")
    H = im.get("height")
    dl_idx = im.get("dl_idx")  # ✅ 여기 값이 "진짜 dl_idx"일 가능성이 가장 높음

    return file_name, W, H, dl_idx

# =========================
# LOAD MAPPING (dl_idx -> train_id)
# =========================
with open(MAP_JSON, "r", encoding="utf-8") as f:
    dlidx_to_trainid = {int(k): int(v) for k, v in json.load(f).items()}

# categories: train_id 기준으로 정렬
categories = []
for dl, tid in sorted(dlidx_to_trainid.items(), key=lambda x: x[1]):
    categories.append({
        "id": tid,
        "name": f"dl_{dl}",
        "supercategory": "pill"
    })

# =========================
# SCAN JSON FILES
# =========================
json_files = []
for root, _, files in os.walk(JSON_ROOT):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        # mapping json 같은 건 제외(원하면 더 추가 가능)
        if fn == "category_id_mapping.json":
            continue
        json_files.append(os.path.join(root, fn))

print(f"[STEP 01-2] JSON files found: {len(json_files)}")

# =========================
# COCO CONTAINERS
# =========================
coco_images = []
coco_annotations = []

# file_name -> image_id (중복 방지/병합)
image_id_by_name = {}
image_wh_by_name = {}

next_image_id = 1
next_ann_id = 1

stats = defaultdict(int)

for jp in tqdm(json_files, mininterval=5):
    try:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        stats["skip_load_err"] += 1
        continue

    images = data.get("images", [])
    anns = data.get("annotations", [])

    if not images or not anns:
        stats["skip_empty"] += 1
        continue

    meta = get_image_meta(images)
    if meta is None:
        stats["skip_no_image_meta"] += 1
        continue

    file_name, W, H, dl_from_image = meta
    if not file_name or not W or not H:
        stats["skip_bad_image_fields"] += 1
        continue

    W = to_int(W)
    H = to_int(H)
    if W is None or H is None or W <= 0 or H <= 0:
        stats["skip_bad_wh"] += 1
        continue

    # ---------------------------------
    # dl_idx 결정 규칙 (핵심)
    # 1) images[0].dl_idx가 있으면 그걸 "진짜 dl_idx"로 최우선
    # 2) 없으면 annotations[*].category_id 중 dl_idx로 보이는 값(>100) 사용
    # ---------------------------------
    dl_idx = to_int(dl_from_image)
    if dl_idx is None:
        dl_candidates = []
        for a in anns:
            cid = to_int(a.get("category_id"))
            if cid is not None and cid > 100:
                dl_candidates.append(cid)
        dl_idx = dl_candidates[0] if dl_candidates else None

    if dl_idx is None:
        stats["skip_no_dl_idx"] += 1
        continue

    if dl_idx not in dlidx_to_trainid:
        # 매핑에 없는 dl_idx는 "학습 set 밖"으로 간주하고 스킵(안 꼬이게)
        stats["skip_dl_not_in_mapping"] += 1
        continue

    train_id = dlidx_to_trainid[dl_idx]

    # ---------------------------------
    # image 등록(중복 file_name이면 병합)
    # ---------------------------------
    new_image_created = False
    if file_name not in image_id_by_name:
        image_id_by_name[file_name] = next_image_id
        image_wh_by_name[file_name] = (W, H)
        coco_images.append({
            "id": next_image_id,
            "file_name": file_name,
            "width": W,
            "height": H
        })
        next_image_id += 1
        new_image_created = True
        stats["image_added"] += 1
    else:
        # 혹시 다른 JSON에서 같은 file_name 나오면 WH 일치 체크
        pw, ph = image_wh_by_name[file_name]
        if pw != W or ph != H:
            stats["skip_wh_mismatch_duplicate_name"] += 1
            continue

    image_id = image_id_by_name[file_name]

    # ---------------------------------
    # annotations 추가
    # - category_id는 train_id(0-based)
    # - orig_dl_idx로 dl_idx 보존
    # ---------------------------------
    added_any = False

    for a in anns:
        bbox_raw = a.get("bbox")
        bb = parse_bbox(bbox_raw)
        if bb is None:
            stats["skip_bad_bbox_format"] += 1
            continue

        x, y, bw, bh = bb
        clamped = clamp_bbox(x, y, bw, bh, W, H)
        if clamped is None:
            stats["skip_bad_bbox_value"] += 1
            continue

        x1, y1, nw, nh = clamped

        coco_annotations.append({
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": int(train_id),   # ✅ 학습용 cls
            "bbox": [float(x1), float(y1), float(nw), float(nh)],
            "area": float(nw * nh),
            "iscrowd": 0,
            "orig_dl_idx": int(dl_idx),     # ✅ 추적용(중요)
            "src_json": os.path.basename(jp)
        })
        next_ann_id += 1
        added_any = True
        stats["ann_added"] += 1

    if not added_any:
        # 이 JSON은 유효 bbox가 0개
        stats["skip_no_valid_ann_in_file"] += 1
        # 방금 추가한 이미지였는데 annotation 0개면 이미지도 되돌림
        if new_image_created:
            coco_images.pop()
            del image_id_by_name[file_name]
            del image_wh_by_name[file_name]
            next_image_id -= 1
            stats["rollback_image_no_ann"] += 1

# =========================
# SAVE COCO
# =========================
coco = {
    "info": {"description": "train_coco_dlidx (category_id=train_id, orig_dl_idx preserved)"},
    "licenses": [],
    "images": coco_images,
    "annotations": coco_annotations,
    "categories": categories,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print("\n[DONE] STEP 01-2 completed")
print(f" - images: {len(coco_images)}")
print(f" - annotations: {len(coco_annotations)}")
print(f" - classes: {len(categories)}")
print(f" - saved to: {OUT_JSON}")

print("\n[STATS]")
for k in sorted(stats.keys()):
    print(f"  {k}: {stats[k]}")