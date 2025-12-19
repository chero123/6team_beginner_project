import os
import json
from glob import glob
from tqdm import tqdm

# =========================
# PATH CONFIG (WSL 기준)
# =========================

IMG_ROOTS = [
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터",
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/yolo_dataset/images",
]

JSON_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터"
YOLO_LABEL_DIR = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/yolo_dataset/labels"
YOLO_CLASS_MAP = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"

OUT_DIR = "/home/ohs3201/work/step2_unified_coco"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_JSON = os.path.join(OUT_DIR, "unified.json")

# =========================
# LOAD YOLO CLASS MAP
# =========================

with open(YOLO_CLASS_MAP, encoding="utf-8") as f:
    raw_map = json.load(f)

# cls_id(int) → category_id(int)
YOLO_CLS_TO_CAT = {}
for cat_id, name in raw_map.items():
    cls = int(name.split("cls")[-1].strip(" )"))
    YOLO_CLS_TO_CAT[cls] = int(cat_id)

# =========================
# INDEX ALL IMAGES
# =========================

print("[1] Indexing images...")
image_index = {}

for root in IMG_ROOTS:
    for img in glob(f"{root}/**/*.png", recursive=True):
        image_index[os.path.basename(img)] = img

print(f"  - total images indexed: {len(image_index)}")

# =========================
# INIT COCO STRUCTURE
# =========================

images = []
annotations = []
categories = {}
img_id_map = {}
ann_id = 1
img_id = 1

# =========================
# LOAD JSON ANNOTATIONS
# =========================

print("[2] Loading JSON annotations...")

json_files = glob(f"{JSON_ROOT}/**/*.json", recursive=True)

for jf in tqdm(json_files):
    try:
        data = json.load(open(jf, encoding="utf-8"))
    except Exception:
        continue

    if "images" not in data or "annotations" not in data:
        continue

    img_info = data["images"][0]
    fname = img_info.get("file_name")
    if fname not in image_index:
        continue

    if fname not in img_id_map:
        img_id_map[fname] = img_id
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": img_info.get("width", 0),
            "height": img_info.get("height", 0)
        })
        img_id += 1

    for a in data["annotations"]:
        bbox = a.get("bbox", [])

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue

        cid = int(a["category_id"])
        categories[cid] = categories.get(cid, {
            "id": cid,
            "name": str(cid),
            "supercategory": "pill"
        })

        annotations.append({
            "id": ann_id,
            "image_id": img_id_map[fname],
            "category_id": cid,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id += 1

# =========================
# LOAD YOLO ANNOTATIONS
# =========================

print("[3] Loading YOLO annotations...")

yolo_labels = glob(f"{YOLO_LABEL_DIR}/**/*.txt", recursive=True)

for lbl in tqdm(yolo_labels):
    base = os.path.splitext(os.path.basename(lbl))[0]
    img_name = base + ".png"
    if img_name not in image_index:
        continue

    if img_name not in img_id_map:
        img_id_map[img_name] = img_id
        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": 0,
            "height": 0
        })
        img_id += 1

    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, cx, cy, bw, bh = map(float, parts)
            cls = int(cls)
            if cls not in YOLO_CLS_TO_CAT:
                continue

            if bw <= 0 or bh <= 0:
                continue

            cid = YOLO_CLS_TO_CAT[cls]
            categories[cid] = categories.get(cid, {
                "id": cid,
                "name": str(cid),
                "supercategory": "pill"
            })

            annotations.append({
                "id": ann_id,
                "image_id": img_id_map[img_name],
                "category_id": cid,
                "bbox": [cx, cy, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

# =========================
# SAVE
# =========================

coco = {
    "images": images,
    "annotations": annotations,
    "categories": list(categories.values())
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False)

print("\n[STEP 2 DONE]")
print(f"Images      : {len(images)}")
print(f"Annotations : {len(annotations)}")
print(f"Categories  : {len(categories)}")
print(f"Saved to    : {OUT_JSON}")