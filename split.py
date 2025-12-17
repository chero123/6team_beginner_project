import os
import json
from pathlib import Path
from collections import defaultdict
import random

BASE = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

ANN_DIR = BASE / "data_ai06" / "train_annotations"

# 1) 이미지 기준으로 json 목록 묶기
img_to_jsons = defaultdict(list)

for root, dirs, files in os.walk(ANN_DIR):
    for fname in files:
        if not fname.endswith(".json"):
            continue

        full_path = Path(root) / fname
        rel_path = full_path.relative_to(ANN_DIR)  # split.json에 저장할 상대경로

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_name = data["images"][0]["file_name"]
        img_file = os.path.basename(img_name)

        img_to_jsons[img_file].append(str(rel_path))

# 2) 이미지 전체 리스트 섞어서 8:2 나누기
all_imgs = list(img_to_jsons.keys())
random.seed(42)
random.shuffle(all_imgs)

n_total = len(all_imgs)
train_count = int(n_total * 0.8)

train_imgs = set(all_imgs[:train_count])
val_imgs   = set(all_imgs[train_count:])

train_jsons = []
val_jsons   = []

# 3) 이미지 기준으로 json들을 train/val로 분리
for img_file, json_list in img_to_jsons.items():
    if img_file in train_imgs:
        train_jsons.extend(json_list)
    else:
        val_jsons.extend(json_list)

# 4) split.json 저장
split_info = {
    "train": train_jsons,
    "val": val_jsons,
}

out_path = BASE / "split.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print("총 이미지 수 :", n_total)
print("train 이미지 :", len(train_imgs))
print("val 이미지   :", len(val_imgs))
print("train json   :", len(train_jsons))
print("val json     :", len(val_jsons))
print("✔ 이미지 기준 split.json 생성 완료 →", out_path)
