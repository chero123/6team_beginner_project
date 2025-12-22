import json, os
from collections import defaultdict

BASE = "/home/.../project_root"
ann = json.load(open(f"{BASE}/data/annotations.json"))
VALID = set(json.load(open(f"{BASE}/category_info.json"))["categories"])

def build(split_ids, out_path):
    new = {"images": [], "annotations": [], "categories": []}

    # categories 추가
    for cid in sorted(VALID):
        new["categories"].append({"id": cid, "name": str(cid)})

    for img in ann["images"]:
        if img["id"] in split_ids:
            new["images"].append(img)

    aid = 1
    for a in ann["annotations"]:
        if a["category_id"] not in VALID:
            continue
        if a["image_id"] not in split_ids:
            continue
        
        a["id"] = aid
        new["annotations"].append(a)
        aid += 1

    json.dump(new, open(out_path, "w"))

train_ids = set([...])
val_ids   = set([...])

build(train_ids, f"{BASE}/coco_dataset/train.json")
build(val_ids,   f"{BASE}/coco_dataset/val.json")
print(" COCO 변환 완료!")