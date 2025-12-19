import os
import json
from pathlib import Path

# ========= 설정 =========
UNIFIED_JSON = "/home/ohs3201/work/step2_unified_coco/unified.json"

# 🔹 원천 이미지들이 흩어져 있는 루트 (여러 개 가능)
IMAGE_ROOTS = [
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터",
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/yolo_dataset/images",
]

OUT_IMG_DIR = "/home/ohs3201/work/step2_unified_coco/images"
# ========================

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# 1️⃣ unified.json 로드
with open(UNIFIED_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

# 2️⃣ 원천 이미지 전체 인덱싱 (파일명 → 실제 경로)
print("[1] Indexing source images...")
image_index = {}

for root in IMAGE_ROOTS:
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                if fn not in image_index:
                    image_index[fn] = os.path.join(r, fn)

print(f"  - indexed images: {len(image_index)}")

# 3️⃣ symlink 생성
print("[2] Creating symlinks...")
linked = 0
missing = 0

for img in coco["images"]:
    fname = img["file_name"]
    dst = os.path.join(OUT_IMG_DIR, fname)

    if os.path.exists(dst):
        linked += 1
        continue

    src = image_index.get(fname)
    if src is None:
        missing += 1
        continue

    os.symlink(src, dst)
    linked += 1

print("\n[STEP 2.5 DONE]")
print(f"  - linked images : {linked}")
print(f"  - missing images: {missing}")
print(f"  - output dir    : {OUT_IMG_DIR}")