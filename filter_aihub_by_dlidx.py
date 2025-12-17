# AI-Hub 원본 데이터(aihub_data)에서 
# category_id_mapping.json에 들어있는 내가 원하는 dl_idx(약)만 등장하는 이미지만 골라서
# 이미지 + 해당 json 라벨들을 통째로 복사

import json
import shutil
from pathlib import Path
from collections import defaultdict

# -----------------------------
# 설정
# -----------------------------
BASE_DIR = Path(r"C:\Users\sangj\workspace\6team_beginner_project")
AIHUB_ROOT = BASE_DIR / "aihub_data"
IMAGES_ROOT = AIHUB_ROOT / "images"
LABELS_ROOT = AIHUB_ROOT / "labels"

# 네가 이미 만들어둔 dl_idx 목록(키가 dl_idx라고 가정)
CAT_MAP_PATH = BASE_DIR / "category_id_mapping.json"

# 결과 저장 폴더
OUT_ROOT = BASE_DIR / "aihub_filtered_for_my_model_all"
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"

# 엄격 모드: 이미지에 등장하는 모든 dl_idx가 target 안에 있어야만 keep
STRICT_ALL_IN_TARGET = True

# -----------------------------
# 1) target dl_idx 집합 로드
# -----------------------------
with open(CAT_MAP_PATH, "r", encoding="utf-8") as f:
    cat_map = json.load(f)

# keys: "1899", "2482", ... 라고 했으니 keys를 target으로 사용
target_dlidx = set(str(k) for k in cat_map.keys())

print(f"[INFO] target dl_idx count = {len(target_dlidx)}")

# -----------------------------
# 2) labels 구조를 훑어서 (image_file_name -> 관련 json 파일들) 모으기
# -----------------------------
# image_file_name 기준으로, 해당 이미지에 매칭되는 json 경로들을 모음
image_to_json_paths = defaultdict(list)
image_to_dlidx_set = defaultdict(set)
image_to_combo_folder = {}  # 이미지가 속한 조합 폴더 추적용

# labels 아래는: <combo>_json / <K-xxxxx> / *.json
combo_json_folders = [p for p in LABELS_ROOT.iterdir() if p.is_dir() and p.name.endswith("_json")]

for combo_json in combo_json_folders:
    # 예: K-000250-...-006192_json
    drug_folders = [d for d in combo_json.iterdir() if d.is_dir()]
    for drug_dir in drug_folders:
        for js in drug_dir.glob("*.json"):
            try:
                with open(js, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 예시처럼 images[0]에 file_name, dl_idx 존재한다고 가정
                img_info = data["images"][0]
                img_file = img_info["file_name"]
                dl_idx = str(img_info["dl_idx"])

                image_to_json_paths[img_file].append(js)
                image_to_dlidx_set[img_file].add(dl_idx)

                # 이 img_file이 어느 조합 폴더에 속하는지 기록(대부분 1개로 고정)
                if img_file not in image_to_combo_folder:
                    image_to_combo_folder[img_file] = combo_json.name.replace("_json", "")

            except Exception as e:
                print(f"[WARN] failed to parse {js}: {e}")

print(f"[INFO] discovered images (from labels) = {len(image_to_json_paths)}")

# -----------------------------
# 3) 조건 검사 후 복사
# -----------------------------
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

kept = 0
skipped = 0

for img_file, json_paths in image_to_json_paths.items():
    dlset = image_to_dlidx_set[img_file]
    combo_folder = image_to_combo_folder.get(img_file)

    if combo_folder is None:
        skipped += 1
        continue

    # 엄격 모드: 이미지에 등장하는 약 전부가 target 안에 있어야 keep
    if STRICT_ALL_IN_TARGET:
        keep = dlset.issubset(target_dlidx)
    else:
        # 부분 모드: target과 교집합이 있으면 keep (비추천)
        keep = len(dlset.intersection(target_dlidx)) > 0

    if not keep:
        skipped += 1
        continue

    # -----------------------------
    # 이미지 복사
    # images/<combo_folder>/<img_file> -> out/images/<combo_folder>/<img_file>
    # -----------------------------
    src_img = IMAGES_ROOT / combo_folder / img_file
    dst_img_dir = OUT_IMAGES / combo_folder
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_img = dst_img_dir / img_file

    if not src_img.exists():
        # 혹시 다른 구조면 여기서 경고 뜸
        print(f"[WARN] image not found: {src_img}")
        skipped += 1
        continue

    shutil.copy2(src_img, dst_img)

    # -----------------------------
    # 해당 이미지에 매칭되는 json들 복사
    # labels/<combo>_json/<drug>/<json> -> out/labels 동일 구조
    # -----------------------------
    combo_json_name = combo_folder + "_json"
    for js in json_paths:
        # 원본 labels에서 combo_json_name 하위 경로를 그대로 유지
        rel = js.relative_to(LABELS_ROOT)  # 예: K-..._json/K-000250/xxxx.json
        dst_js = OUT_LABELS / rel
        dst_js.parent.mkdir(parents=True, exist_ok=True)

        # 엄격 모드에서는 그냥 다 복사하면 됨 (이미 dlset이 전부 target임)
        # 부분 모드였다면 여기서 dl_idx가 target인 json만 복사하도록 조건 걸어야 함
        shutil.copy2(js, dst_js)

    kept += 1

print("-------- RESULT --------")
print(f"kept images = {kept}")
print(f"skipped images = {skipped}")
print(f"output = {OUT_ROOT}")
