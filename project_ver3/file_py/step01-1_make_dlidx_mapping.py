# STEP 01-1
# 모든 JSON + YOLO txt 스캔해서 dl_idx 기준 train_id 매핑 생성
# ❗ bbox / COCO 생성 안 함 (순수 매핑만)

import os
import json
import re
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
JSON_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터"
YOLO_TXT_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/YOLO_TXT"
YOLO_CLS_MAP = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"

OUT_DIR = "/home/ohs3201/6team_beginner_project/project_ver3/mappings"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DL2T = os.path.join(OUT_DIR, "dlidx_to_trainid.json")
OUT_T2DL = os.path.join(OUT_DIR, "trainid_to_dlidx.json")

# =========================
# UTIL
# =========================
def safe_int(x):
    try:
        return int(x)
    except:
        return None

def extract_dl_idx_from_str(s):
    if not isinstance(s, str):
        return None
    nums = re.findall(r"\d+", s)
    return int(nums[0]) if nums else None

# =========================
# 1. JSON에서 dl_idx 수집
# =========================
dlidx_set = set()
json_files = []

print("[STEP 01-1] Scan JSON files...")
for root, _, files in os.walk(JSON_ROOT):
    for f in files:
        if f.endswith(".json"):
            json_files.append(os.path.join(root, f))

for jp in tqdm(json_files):
    try:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        continue

    # images[].dl_idx
    for img in data.get("images", []):
        dl = img.get("dl_idx")
        dl = safe_int(dl)
        if dl is not None:
            dlidx_set.add(dl)

    # annotations[].category_id (dl_idx인 경우만)
    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        cid = safe_int(cid)
        if cid is not None and cid > 100:  # dl_idx는 항상 큼
            dlidx_set.add(cid)

print(f"[INFO] dl_idx from JSON: {len(dlidx_set)}")

# =========================
# 2. YOLO cls mapping에서 dl_idx 수집
# =========================
print("[STEP 01-1] Scan YOLO cls mapping...")

with open(YOLO_CLS_MAP, "r", encoding="utf-8") as f:
    cls_map = json.load(f)

cls_to_dlidx = {}

for dl_key, desc in cls_map.items():
    dl = safe_int(dl_key)
    if dl is None:
        continue

    # "(cls X)" 에서 cls 번호 추출
    m = re.search(r"\(cls\s*(\d+)\)", desc)
    if not m:
        continue

    cls = int(m.group(1))
    cls_to_dlidx[cls] = dl
    dlidx_set.add(dl)

print(f"[INFO] dl_idx after cls mapping: {len(dlidx_set)}")

# =========================
# 3. YOLO txt 보완 (JSON 없는 이미지용)
# =========================
print("[STEP 01-1] Scan YOLO txt files...")

for root, _, files in os.walk(YOLO_TXT_ROOT):
    for f in files:
        if not f.endswith(".txt"):
            continue
        path = os.path.join(root, f)
        try:
            with open(path) as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    cls = int(line.split()[0])
                    if cls in cls_to_dlidx:
                        dlidx_set.add(cls_to_dlidx[cls])
        except:
            continue

print(f"[INFO] dl_idx after YOLO txt merge: {len(dlidx_set)}")

# =========================
# 4. dl_idx → train_id (0-based)
# =========================
dlidx_list = sorted(dlidx_set)
dlidx_to_trainid = {str(dl): i for i, dl in enumerate(dlidx_list)}
trainid_to_dlidx = {str(i): dl for dl, i in dlidx_to_trainid.items()}

with open(OUT_DL2T, "w", encoding="utf-8") as f:
    json.dump(dlidx_to_trainid, f, indent=2, ensure_ascii=False)

with open(OUT_T2DL, "w", encoding="utf-8") as f:
    json.dump(trainid_to_dlidx, f, indent=2, ensure_ascii=False)

# =========================
# DONE
# =========================
print("\n[DONE] STEP 01-1 mapping created")
print(f" - total classes: {len(dlidx_list)}")
print(f" - {OUT_DL2T}")
print(f" - {OUT_T2DL}")

print("\n[Sample]")
for dl in dlidx_list[:10]:
    print(f" dl_idx {dl} -> train_id {dlidx_to_trainid[str(dl)]}")