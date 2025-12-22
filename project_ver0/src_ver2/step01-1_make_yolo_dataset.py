import os
import json
import glob
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from PIL import Image

HOME = os.path.expanduser("~")
BASE = f"{HOME}/6team_beginner_project"

RAW_IMG_DIR = f"{BASE}/data/train_images"
RAW_JSON_DIR = f"{BASE}/data/train_annotations"
TEST_IMG_DIR = f"{BASE}/data/test_images"

YOLO_DIR = f"{BASE}/yolo_dataset"
os.makedirs(YOLO_DIR, exist_ok=True)

IMG_OUT = f"{YOLO_DIR}/images"
LBL_OUT = f"{YOLO_DIR}/labels"
os.makedirs(IMG_OUT + "/train", exist_ok=True)
os.makedirs(IMG_OUT + "/val", exist_ok=True)
os.makedirs(LBL_OUT + "/train", exist_ok=True)
os.makedirs(LBL_OUT + "/val", exist_ok=True)

print("📌 Step01 — YOLO Dataset 생성 시작")
print("- RAW_IMG_DIR:", RAW_IMG_DIR)
print("- RAW_JSON_DIR:", RAW_JSON_DIR)

# 1) 모든 JSON 파일 재귀적으로 수집
print("\n[1] JSON 파일 스캔 중...")

json_files = glob.glob(f"{RAW_JSON_DIR}/**/*.json", recursive=True)
image_list = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith(".png")])

json_map = {}
category_set = set()

print(f" - 이미지 개수: {len(image_list)}")
print(f" - JSON 파일 개수: {len(json_files)}")

for jp in tqdm(json_files):
    try:
        with open(jp, "r") as f:
            data = json.load(f)
    except:
        continue

    if "images" not in data or "annotations" not in data:
        continue

    img_name = data["images"][0]["file_name"]

    # annotation 없는 이미지 → 스킵
    if img_name not in image_list:
        continue

    # 한 이미지에 대해 JSON 여러 개가 존재하면 가장 첫 번째만 사용
    if img_name not in json_map:
        json_map[img_name] = jp

    # 카테고리 수집
    for ann in data["annotations"]:
        category_set.add(ann["category_id"])

print(f" - 매칭된 이미지 수: {len(json_map)}")
print(f" - 고유 category 수: {len(category_set)}")

# 2) category_id → YOLO index (0~N−1)
#    YOLO index → original category_id (역매핑)
print("\n[2] Category Mapping 생성")

sorted_categories = sorted(list(category_set))
cat2yolo = {cat: idx for idx, cat in enumerate(sorted_categories)}
yolo2cat = {idx: cat for idx, cat in enumerate(sorted_categories)}

with open(f"{BASE}/category_mapping.json", "w") as f:
    json.dump({
        "cat2yolo": cat2yolo,
        "yolo2cat": yolo2cat
    }, f, indent=2, ensure_ascii=False)

print(" - 저장됨:", f"{BASE}/category_mapping.json")

# 3) Train/Val Split
print("\n[3] Train/Val Split")

valid_imgs = sorted(list(json_map.keys()))
np.random.seed(0)
np.random.shuffle(valid_imgs)

split_idx = int(len(valid_imgs) * 0.9)
train_imgs = valid_imgs[:split_idx]
val_imgs = valid_imgs[split_idx:]

print(f" - Train: {len(train_imgs)}")
print(f" - Val:   {len(val_imgs)}")

# 4) YOLO bbox 변환 함수
def convert_bbox(img_w, img_h, bbox):
    x, y, bw, bh = bbox

    # boundary clipping
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    bw = max(1, min(bw, img_w - x))
    bh = max(1, min(bh, img_h - y))

    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h

    return cx, cy, nw, nh

# 5) YOLO Dataset 생성
print("\n[4] YOLO Dataset 생성 중...")

def process_image(img_name, split):
    img_path = f"{RAW_IMG_DIR}/{img_name}"
    json_path = json_map[img_name]

    try:
        img = Image.open(img_path)
        w, h = img.size
    except:
        print("❌ 이미지 열기 실패:", img_name)
        return

    # 이미지 복사
    shutil.copy(img_path, f"{IMG_OUT}/{split}/{img_name}")

    # 라벨 생성
    with open(json_path, "r") as f:
        data = json.load(f)

    label_path = f"{LBL_OUT}/{split}/{img_name.replace('.png', '.txt')}"
    with open(label_path, "w") as f:
        for ann in data["annotations"]:
            cid = ann["category_id"]
            x_c, y_c, w_n, h_n = convert_bbox(w, h, ann["bbox"])
            f.write(f"{cat2yolo[cid]} {x_c} {y_c} {w_n} {h_n}\n")

for img in tqdm(train_imgs, desc="Train"):
    process_image(img, "train")

for img in tqdm(val_imgs, desc="Val"):
    process_image(img, "val")

# 6) YOLO data.yaml 생성
print("\n[5] data.yaml 생성")

yaml_text = f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\n\nnames:\n"
for yolo_idx, catid in yolo2cat.items():
    yaml_text += f"  {yolo_idx}: '{catid}'\n"

with open(f"{YOLO_DIR}/data.yaml", "w") as f:
    f.write(yaml_text)

print(" - 저장됨:", f"{YOLO_DIR}/data.yaml")

# 7) 5-Fold 생성
print("\n[6] 5-Fold split 생성")

valid_imgs = np.array(valid_imgs)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

folds = {}
for fold_idx, (tr, va) in enumerate(kf.split(valid_imgs)):
    folds[fold_idx] = {
        "train": valid_imgs[tr].tolist(),
        "val":   valid_imgs[va].tolist()
    }

with open(f"{BASE}/folds_5.json", "w") as f:
    json.dump(folds, f, indent=2, ensure_ascii=False)

print(" - 저장됨:", f"{BASE}/folds_5.json")
print("\n🎉 Step01 완료 — YOLO Dataset 생성 성공!")