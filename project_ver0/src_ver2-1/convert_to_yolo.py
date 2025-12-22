import json, os, shutil
from tqdm import tqdm

BASE = "/home/.../project_root"
ann_path = f"{BASE}/data/annotations.json"

with open(f"{BASE}/category_info.json") as f:
    VALID = set(json.load(f)["categories"])

os.makedirs(f"{BASE}/yolo_dataset/images/train", exist_ok=True)
os.makedirs(f"{BASE}/yolo_dataset/images/val", exist_ok=True)
os.makedirs(f"{BASE}/yolo_dataset/labels/train", exist_ok=True)
os.makedirs(f"{BASE}/yolo_dataset/labels/val", exist_ok=True)

ann = json.load(open(ann_path))
images = {img["id"]: img for img in ann["images"]}

train_ids = set([...])   # 기존 split 그대로 사용
val_ids   = set([...])

for a in tqdm(ann["annotations"]):
    cid = a["category_id"]
    if cid not in VALID:
        continue

    img_info = images[a["image_id"]]
    fname = img_info["file_name"]

    # YOLO 라벨 경로
    split = "train" if a["image_id"] in train_ids else "val"
    label_path = f"{BASE}/yolo_dataset/labels/{split}/{fname.replace('.jpg','.txt')}"

    # YOLO bbox 변환
    x,y,w,h = a["bbox"]
    cx = (x + w/2) / img_info["width"]
    cy = (y + h/2) / img_info["height"]
    nw = w / img_info["width"]
    nh = h / img_info["height"]

    with open(label_path, "a") as f:
        f.write(f"{cid} {cx} {cy} {nw} {nh}\n")

    # 이미지 복사
    shutil.copy(
        f"{BASE}/data/train_images/{fname}",
        f"{BASE}/yolo_dataset/images/{split}/{fname}"
    )

print(" YOLO dataset 생성 완료!")