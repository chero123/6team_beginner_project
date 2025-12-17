#ai 허브 데이터 중에서 내 클래스에 맞는 약들만 모아 data.yml을 만들어줌

import re
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image  # pip install pillow

# =============================
# 설정
# =============================
BASE_DIR = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

# 입력(이미 정제된 폴더)
SRC_ROOT = BASE_DIR / "aihub_filtered_for_my_model_all"
SRC_IMAGES = SRC_ROOT / "images"
SRC_LABELS = SRC_ROOT / "labels"

# dl_idx -> cls 매핑
CAT_MAP_PATH = BASE_DIR / "category_id_mapping.json"

# 출력(여기에 YOLO 형식으로 저장)
OUT_ROOT = BASE_DIR / "aihub_filtered_all"
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"

VAL_RATIO = 0.1
SEED = 42

# 기존 있으면 스킵(덮어쓰기 싫으면 True)
SKIP_EXISTING = False

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# =============================
# Helper: mapping 로드 (dl_idx -> cls)
# category_id_mapping.json value가 "... (cls 7)" 형태라고 가정
# =============================
def load_dlidx_to_cls(cat_map_path: Path):
    raw = json.loads(cat_map_path.read_text(encoding="utf-8"))
    pat = re.compile(r"\(cls\s*(\d+)\)", re.IGNORECASE)

    dlidx_to_cls = {}
    for dlidx, v in raw.items():
        m = pat.search(str(v))
        if not m:
            raise ValueError(f"[ERROR] mapping value에서 cls 못 찾음: dl_idx={dlidx}, value={v}")
        dlidx_to_cls[str(dlidx)] = int(m.group(1))

    return dlidx_to_cls

# =============================
# Helper: bbox 추출 (COCO/변형 대응)
# returns x,y,w,h (pixels)
# =============================
def extract_bbox(ann: dict):
    if "bbox" in ann:
        b = ann["bbox"]

        # [x,y,w,h]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            x, y, w, h = b[:4]
            return float(x), float(y), float(w), float(h)

        # dict 형태
        if isinstance(b, dict):
            if all(k in b for k in ("x", "y", "w", "h")):
                return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            if all(k in b for k in ("x", "y", "width", "height")):
                return float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])
            if all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                xmin, ymin, xmax, ymax = map(float, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
                return xmin, ymin, (xmax - xmin), (ymax - ymin)

        # "x,y,w,h"
        if isinstance(b, str):
            parts = re.split(r"[,\s]+", b.strip())
            if len(parts) >= 4:
                x, y, w, h = parts[:4]
                return float(x), float(y), float(w), float(h)

    # bbox가 풀려있는 케이스
    if all(k in ann for k in ("x", "y", "width", "height")):
        return float(ann["x"]), float(ann["y"]), float(ann["width"]), float(ann["height"])

    if all(k in ann for k in ("xmin", "ymin", "xmax", "ymax")):
        xmin, ymin, xmax, ymax = map(float, (ann["xmin"], ann["ymin"], ann["xmax"], ann["ymax"]))
        return xmin, ymin, (xmax - xmin), (ymax - ymin)

    raise KeyError(f"[ERROR] bbox 형식 인식 실패. bbox_type={type(ann.get('bbox'))}, bbox={ann.get('bbox')}")

# =============================
# Helper: YOLO line
# =============================
def to_yolo_line(cls_id: int, x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    # clip
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)

    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if SKIP_EXISTING and dst.exists():
        return False
    shutil.copy2(src, dst)
    return True

# =============================
# 1) 매핑 로드
# =============================
dlidx_to_cls = load_dlidx_to_cls(CAT_MAP_PATH)
num_classes = max(dlidx_to_cls.values()) + 1
print(f"[INFO] dl_idx mapping loaded: {len(dlidx_to_cls)} items, num_classes={num_classes}")

# =============================
# 2) SRC_LABELS 아래 json 전부 읽어서 이미지별로 bbox 병합
# key는 "combo/filename" 형태로 (충돌 방지)
# =============================
imgkey_to_records = defaultdict(list)  # key -> [(cls, x,y,w,h, iw, ih), ...]
json_paths = list(SRC_LABELS.rglob("*.json"))
print(f"[INFO] json files found in filtered folder = {len(json_paths)}")

failed = 0

for js in json_paths:
    # 맥 메타/숨김파일 스킵
    if js.name.startswith("._") or js.name.startswith("."):
        continue

    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # 혹시 인코딩 섞이면 cp949 fallback
        data = json.loads(js.read_text(encoding="cp949"))
    except Exception as e:
        failed += 1
        continue

    try:
        img_info = data["images"][0]
        file_name = img_info["file_name"]
        dl_idx = str(img_info.get("dl_idx", ""))

        cls_id = dlidx_to_cls.get(dl_idx, None)
        if cls_id is None:
            continue

        # labels 최상위 폴더가 "<combo>_json"인 구조를 이용
        combo_json = js.relative_to(SRC_LABELS).parts[0]  # "<combo>_json"
        combo = combo_json.replace("_json", "")
        img_rel = Path(combo) / file_name  # images/<combo>/<file_name>

        # 이미지 크기
        iw = img_info.get("width")
        ih = img_info.get("height")

        if not iw or not ih:
            img_path = SRC_IMAGES / img_rel
            if not img_path.exists():
                # fallback: 파일명으로 검색
                hits = list(SRC_IMAGES.rglob(file_name))
                if not hits:
                    continue
                img_path = hits[0]
                try:
                    img_rel = img_path.relative_to(SRC_IMAGES)
                except Exception:
                    pass

            with Image.open(img_path) as im:
                iw, ih = im.size

        anns = data.get("annotations", [])
        if not anns:
            continue

        for ann in anns:
            x, y, w, h = extract_bbox(ann)
            imgkey_to_records[str(img_rel)].append((cls_id, x, y, w, h, int(iw), int(ih)))

    except Exception:
        failed += 1
        continue

print(f"[INFO] images with merged labels = {len(imgkey_to_records)}")
print(f"[INFO] json parse/label failures (skipped json) = {failed}")

# =============================
# 3) train/val split
# =============================
keys = list(imgkey_to_records.keys())
random.Random(SEED).shuffle(keys)
val_count = int(len(keys) * VAL_RATIO)
val_keys = set(keys[:val_count])

print(f"[INFO] split: train={len(keys)-len(val_keys)}, val={len(val_keys)} (VAL_RATIO={VAL_RATIO})")

# =============================
# 4) 출력 폴더 생성 + 복사 + txt 생성
# OUT_ROOT/images/train, OUT_ROOT/labels/train ...
# =============================
for split in ["train", "val"]:
    (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (OUT_LABELS / split).mkdir(parents=True, exist_ok=True)

copied_img = 0
made_txt = 0
missing_img = 0

for img_rel_str, recs in imgkey_to_records.items():
    img_rel = Path(img_rel_str)
    split = "val" if img_rel_str in val_keys else "train"

    src_img = SRC_IMAGES / img_rel
    if not src_img.exists():
        # fallback
        hits = list(SRC_IMAGES.rglob(img_rel.name))
        if not hits:
            missing_img += 1
            continue
        src_img = hits[0]

    # 파일명 충돌 방지: "__<combo>" 붙여 저장
    unique_name = f"{src_img.stem}__{img_rel.parent.name}{src_img.suffix}"
    dst_img = OUT_IMAGES / split / unique_name
    safe_copy(src_img, dst_img)
    copied_img += 1

    dst_txt = OUT_LABELS / split / (Path(unique_name).stem + ".txt")
    if SKIP_EXISTING and dst_txt.exists():
        continue

    lines = []
    for cls_id, x, y, w, h, iw, ih in recs:
        lines.append(to_yolo_line(cls_id, x, y, w, h, iw, ih))

    dst_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    made_txt += 1

print("-------- RESULT --------")
print(f"OUT_ROOT = {OUT_ROOT}")
print(f"copied images = {copied_img}")
print(f"made txt     = {made_txt}")
print(f"missing img  = {missing_img}")

# =============================
# 5) data.yml 생성 (OUT_ROOT/data.yml)
# =============================
names = [f"pill_{i}" for i in range(num_classes)]
data_yml = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val

nc: {num_classes}
names: {names}
"""
(OUT_ROOT / "data.yml").write_text(data_yml, encoding="utf-8")
print(f"[INFO] wrote data.yml -> {OUT_ROOT / 'data.yml'}")
