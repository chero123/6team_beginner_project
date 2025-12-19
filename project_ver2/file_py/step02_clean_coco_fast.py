import os, json
from tqdm import tqdm

IN_JSON = "/home/ohs3201/work/step1_unified_coco/unified.json"
OUT_DIR = "/home/ohs3201/work/step2_clean_coco"
OUT_JSON = os.path.join(OUT_DIR, "clean.json")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# load
# -------------------------
with open(IN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", [])
categories = data.get("categories", [])

# image_id -> image
img_by_id = {im["id"]: im for im in images}

# -------------------------
# 1) image 기본 검증 (FAST)
#   - STEP 1에서 이미지 존재/크기 확인 완료 가정
#   - 여기서는 width/height > 0만 확인
# -------------------------
valid_images = {}
for im in images:
    W = int(im.get("width", 0))
    H = int(im.get("height", 0))
    if W > 0 and H > 0:
        valid_images[im["id"]] = {
            "id": im["id"],
            "file_name": im["file_name"],
            "width": W,
            "height": H,
        }

print(f"[INFO] valid images (W,H>0): {len(valid_images)} / {len(images)}")

# -------------------------
# 2) bbox clamp + 정제 (FAST, 메모리만)
# -------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

clean_anns = []
ann_cnt = {}

for an in tqdm(annotations, desc="Clean annotations (FAST)"):
    iid = an.get("image_id")
    if iid not in valid_images:
        continue

    im = valid_images[iid]
    W, H = im["width"], im["height"]

    bbox = an.get("bbox")
    if not bbox or len(bbox) != 4:
        continue

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        continue

    # clamp to image
    x1 = clamp(x, 0, W - 1)
    y1 = clamp(y, 0, H - 1)
    x2 = clamp(x + w, 1, W)
    y2 = clamp(y + h, 1, H)

    nw = x2 - x1
    nh = y2 - y1
    if nw <= 1 or nh <= 1:
        continue

    clean_anns.append({
        "id": None,  # 재부여
        "image_id": iid,
        "category_id": int(an["category_id"]),  # dl_idx 유지
        "bbox": [float(x1), float(y1), float(nw), float(nh)],
        "area": float(nw * nh),
        "iscrowd": int(an.get("iscrowd", 0)),
    })
    ann_cnt[iid] = ann_cnt.get(iid, 0) + 1

# -------------------------
# 3) annotation 0개 이미지 제거 + ID 재부여
# -------------------------
new_images = []
old2new = {}
new_img_id = 1

for old_id, im in valid_images.items():
    if ann_cnt.get(old_id, 0) == 0:
        continue
    old2new[old_id] = new_img_id
    new_images.append({
        "id": new_img_id,
        "file_name": im["file_name"],
        "width": im["width"],
        "height": im["height"],
    })
    new_img_id += 1

new_anns = []
new_ann_id = 1
for an in clean_anns:
    oid = an["image_id"]
    if oid not in old2new:
        continue
    an["image_id"] = old2new[oid]
    an["id"] = new_ann_id
    new_anns.append(an)
    new_ann_id += 1

# -------------------------
# save
# -------------------------
out = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": new_images,
    "annotations": new_anns,
    "categories": categories,  # 클래스 유지
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"[DONE] images={len(new_images)} anns={len(new_anns)} cats={len(categories)}")
print(f"[DONE] saved -> {OUT_JSON}")