import os
import json
from collections import defaultdict

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"

COCO_JSON = f"{PROJECT_ROOT}/coco/train_coco_trainid_SAFE_FULL_IMAGES.json"
MAP_T2DL  = f"{PROJECT_ROOT}/mappings/trainid_to_dlidx.json"

IMG_ROOT  = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터"

# 1) load coco
coco = json.load(open(COCO_JSON, "r", encoding="utf-8"))
images = {im["id"]: im["file_name"] for im in coco["images"]}
anns = coco["annotations"]

# 2) load train_id -> dl_idx mapping
t2dl_raw = json.load(open(MAP_T2DL, "r", encoding="utf-8"))
t2dl = {int(k): int(v) for k, v in t2dl_raw.items()}

# 3) image_id -> set(dl_idx)  (중요: ann.category_id(train_id)를 dl_idx로 변환)
imgid_to_dl = defaultdict(set)
for a in anns:
    tid = int(a["category_id"])      # train_id
    dl = t2dl.get(tid, None)         # dl_idx
    if dl is None:
        continue
    imgid_to_dl[a["image_id"]].add(dl)

# 4) index all real png basenames
real_pngs = set()
for root, _, files in os.walk(IMG_ROOT):
    for f in files:
        if f.lower().endswith(".png"):
            real_pngs.add(f)

# 5) classify dl_idx by whether any image exists on disk
dl_has_real = set()
dl_only_missing = set()

missing_images = []  # (image_id, file_name, dl_idx_list)
for img_id, dl_set in imgid_to_dl.items():
    fname = images.get(img_id)
    if fname in real_pngs:
        dl_has_real |= dl_set
    else:
        dl_only_missing |= dl_set
        missing_images.append((img_id, fname, sorted(dl_set)))

dl_no_png = sorted(dl_only_missing - dl_has_real)

print("COCO images:", len(images))
print("COCO anns:", len(anns))
print("Missing image files:", len(missing_images))
print("❌ dl_idx with NO real PNG images:", len(dl_no_png))
print(dl_no_png[:200])

# (옵션) missing 이미지 샘플 20개 출력
print("\n[Sample missing images]")
for row in missing_images[:20]:
    print(row)