import os
import json
from collections import defaultdict
from tqdm import tqdm

# =========================
# 0. 경로
# =========================
DATA_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터"
OUT_DIR = "./work_step1"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_EXT = (".png", ".jpg", ".jpeg")
JSON_EXT = ".json"
TXT_EXT = ".txt"

# =========================
# 1. 파일 수집
# =========================
images = {}
jsons = {}
txts = {}

print("[STEP 1] Scanning files...")

for root, _, files in os.walk(DATA_ROOT):
    for f in files:
        fp = os.path.join(root, f)
        name, ext = os.path.splitext(f)

        if ext.lower() in IMG_EXT:
            images[f] = fp
        elif ext.lower() == JSON_EXT:
            jsons[f] = fp
        elif ext.lower() == TXT_EXT:
            txts[f] = fp

print(f"Images: {len(images)}")
print(f"JSONs : {len(jsons)}")
print(f"TXTs  : {len(txts)}")

# =========================
# 2. 이미지 기준 매칭
# =========================
index = defaultdict(dict)

for fname, img_path in images.items():
    index[fname]["image_path"] = img_path

    base = os.path.splitext(fname)[0]

    json_name = base + ".json"
    txt_name = base + ".txt"

    if json_name in jsons:
        index[fname]["json"] = jsons[json_name]

    if txt_name in txts:
        index[fname]["txt"] = txts[txt_name]

# =========================
# 3. 통계
# =========================
stats = {
    "total_images": len(index),
    "image_only": 0,
    "image_with_json": 0,
    "image_with_txt": 0,
    "image_with_json_and_txt": 0,
}

for v in index.values():
    has_json = "json" in v
    has_txt = "txt" in v

    if has_json and has_txt:
        stats["image_with_json_and_txt"] += 1
    elif has_json:
        stats["image_with_json"] += 1
    elif has_txt:
        stats["image_with_txt"] += 1
    else:
        stats["image_only"] += 1

print("\n===== STEP 1 RESULT =====")
for k, v in stats.items():
    print(f"{k}: {v}")

# =========================
# 4. 저장
# =========================
with open(os.path.join(OUT_DIR, "image_index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved to {OUT_DIR}")