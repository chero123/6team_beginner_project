# project_ver4/file_py/step01_1_make_dlidx_mapping_SAFE_FINAL.py
import os
import json
import re
from glob import glob
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"

JSON_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터"

# (선택) YOLO 자산
YOLO_TXT_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/YOLO_TXT"
YOLO_CLS_MAP  = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"

OUT_DIR = os.path.join(PROJECT_ROOT, "mappings")
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

def try_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def parse_cls_map(path):
    """
    YOLO cls mapping json에서
    cls -> dl_idx 매핑 추출
    """
    obj = try_load_json(path)
    if not obj:
        return {}

    cls2dl = {}
    for dl_key, desc in obj.items():
        dl = safe_int(dl_key)
        if dl is None:
            continue

        s = str(desc)
        m = re.search(r"\(cls\s*(\d+)\)", s)
        if m:
            cls2dl[int(m.group(1))] = dl
            continue

        m2 = re.search(r"cls\s*(\d+)", s)
        if m2:
            cls2dl[int(m2.group(1))] = dl

    return cls2dl

# =========================
# STEP 01-1 SAFE FINAL
# =========================
dlidx_set = set()
stats = {
    "json_load_err": 0,
    "dl_from_images": 0,
    "dl_from_annotations": 0,
    "dl_from_yolo_cls": 0,
    "dl_from_yolo_txt": 0,
}

json_paths = glob(os.path.join(JSON_ROOT, "**", "*.json"), recursive=True)
print(f"[STEP 01-1 SAFE FINAL] JSON files found: {len(json_paths)}")

# --------------------------------------------------
# 1) JSON 전체 스캔 (images + annotations)
# --------------------------------------------------
for jp in tqdm(json_paths, desc="Scan JSON"):
    data = try_load_json(jp)
    if data is None:
        stats["json_load_err"] += 1
        continue

    # (A) images[].dl_idx
    for im in data.get("images", []):
        dl = safe_int(im.get("dl_idx"))
        if dl is not None:
            dlidx_set.add(dl)
            stats["dl_from_images"] += 1

    # (B) annotations[].category_id  ⭐ 핵심
    for an in data.get("annotations", []):
        cid = safe_int(an.get("category_id"))
        if cid is not None:
            dlidx_set.add(cid)
            stats["dl_from_annotations"] += 1

print(f"[INFO] dl_idx after JSON scan: {len(dlidx_set)}")

# --------------------------------------------------
# 2) YOLO cls mapping 보완
# --------------------------------------------------
cls2dl = {}
if os.path.exists(YOLO_CLS_MAP):
    cls2dl = parse_cls_map(YOLO_CLS_MAP)
    for dl in cls2dl.values():
        if dl not in dlidx_set:
            stats["dl_from_yolo_cls"] += 1
        dlidx_set.add(dl)
    print(f"[INFO] YOLO cls mapping loaded: {len(cls2dl)}")

# --------------------------------------------------
# 3) YOLO txt 보완 (선택)
# --------------------------------------------------
if os.path.exists(YOLO_TXT_ROOT) and cls2dl:
    txt_paths = glob(os.path.join(YOLO_TXT_ROOT, "**", "*.txt"), recursive=True)
    for tp in tqdm(txt_paths, desc="Scan YOLO txt"):
        try:
            with open(tp) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    cls = safe_int(ln.split()[0])
                    if cls in cls2dl:
                        dl = cls2dl[cls]
                        if dl not in dlidx_set:
                            stats["dl_from_yolo_txt"] += 1
                        dlidx_set.add(dl)
        except:
            pass

# --------------------------------------------------
# 4) dl_idx -> train_id (0-based)
# --------------------------------------------------
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
print("\n[DONE] STEP 01-1 SAFE FINAL")
print(f" - total dl_idx classes: {len(dlidx_list)}")
print(f" - saved: {OUT_DL2T}")
print(f" - saved: {OUT_T2DL}")

print("\n[STATS]")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n[SAMPLE]")
for dl in dlidx_list[:10]:
    print(f"  dl_idx {dl} -> train_id {dlidx_to_trainid[str(dl)]}")