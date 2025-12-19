import os, json
from glob import glob
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# =========================
# PATH CONFIG (WSL 기준, 고정)
# =========================
JSON_ROOTS = [
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/라벨링데이터",
    "/home/ohs3201/6team_beginner_project/data/train_annotations",  # 기존 것도 같이 (있으면)
]

IMAGE_ROOTS = [
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/1.Training/원천데이터",
    "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/yolo_dataset/images",
    "/home/ohs3201/6team_beginner_project/data/train_images",  # 기존 것도 같이 (있으면)
]

YOLO_LABEL_ROOT = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/yolo_dataset/labels"
MAPPING_JSON = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"

OUT_JSON = "/home/ohs3201/work/step1_unified_coco/unified.json"
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg")


# =========================
# mapping: cls -> dl_idx (유일 기준)
# =========================
mapping = json.load(open(MAPPING_JSON, encoding="utf-8"))

cls_to_dlidx = {}
for dl_str, v in mapping.items():
    dl = int(dl_str)
    if isinstance(v, dict) and "cls" in v:
        cls_to_dlidx[int(v["cls"])] = dl
    else:
        # 예: "동아가바펜틴정 800mg (cls 16)"
        s = str(v)
        if "cls" in s:
            cls_part = s.split("cls")[-1].strip().strip("()").strip()
            cls_to_dlidx[int(cls_part)] = dl

if not cls_to_dlidx:
    raise RuntimeError("cls_to_dlidx mapping is empty. category_id_mapping.json 형식을 확인하세요.")


# =========================
# 1) 이미지 인덱싱 (한 번만! - 성능 핵심)
#   basename -> fullpath
# =========================
print("[1/5] Indexing images (once)...")
image_index = {}
dup_image_names = 0

for r in IMAGE_ROOTS:
    if not os.path.exists(r):
        continue
    for ext in IMG_EXTS:
        for p in glob(os.path.join(r, "**", f"*{ext}"), recursive=True):
            bn = os.path.basename(p)
            if bn in image_index:
                dup_image_names += 1
                # 첫 발견 우선(원천데이터가 보통 먼저 오도록 IMAGE_ROOTS 순서 조정 가능)
                continue
            image_index[bn] = p

print(f"  - indexed images: {len(image_index)} (dup name ignored: {dup_image_names})")


# =========================
# 2) 이미지 크기 캐시
# =========================
size_cache = {}

def get_size_by_basename(bn):
    """bn: filename with extension"""
    if bn in size_cache:
        return size_cache[bn]
    p = image_index.get(bn)
    if not p:
        size_cache[bn] = (None, None)
        return None, None
    try:
        with Image.open(p) as im:
            W, H = im.size
        size_cache[bn] = (W, H)
        return W, H
    except:
        size_cache[bn] = (None, None)
        return None, None


# =========================
# 3) YOLO 라벨 인덱싱 (한 번만!)
#   basename.png -> txtpath
# =========================
print("[2/5] Indexing YOLO labels (once)...")
yolo_index = {}
if os.path.exists(YOLO_LABEL_ROOT):
    for p in glob(os.path.join(YOLO_LABEL_ROOT, "**", "*.txt"), recursive=True):
        bn = os.path.basename(p)
        base = os.path.splitext(bn)[0]
        # 이미지 확장자는 png가 대부분이라 가정하되, 없으면 나중에 image_index로 매칭
        yolo_index[base] = p

print(f"  - indexed yolo txt: {len(yolo_index)}")


# =========================
# COCO containers
# =========================
images = {}          # bn -> {"id", "file_name", "width", "height"}
img_id = 1
annotations = []
ann_id = 1
dlidx_set = set()

# 어떤 이미지가 YOLO로 커버됐는지 표시 (JSON 중복 방지)
yolo_covered = set()

# 통계
stats = defaultdict(int)


# =========================
# helper: 이미지 등록
# =========================
def ensure_image(bn):
    global img_id
    if bn in images:
        return images[bn]["id"]
    W, H = get_size_by_basename(bn)
    if not W or not H:
        return None
    images[bn] = {"id": img_id, "file_name": bn, "width": int(W), "height": int(H)}
    img_id += 1
    return images[bn]["id"]


# =========================
# 4) YOLO -> COCO (우선)
# =========================
print("[3/5] Parse YOLO first (priority)...")

for base, txt_path in tqdm(list(yolo_index.items()), desc="YOLO"):
    # 이미지 파일 찾기: base + 확장자 후보
    img_bn = None
    # 가장 흔한 확장자 우선
    for ext in (".png", ".jpg", ".jpeg"):
        cand = base + ext
        if cand in image_index:
            img_bn = cand
            break
    if not img_bn:
        stats["yolo_skip_no_image"] += 1
        continue

    iid = ensure_image(img_bn)
    if iid is None:
        stats["yolo_skip_bad_image"] += 1
        continue

    W, H = images[img_bn]["width"], images[img_bn]["height"]

    ok_any = False
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) != 5:
                    stats["yolo_skip_bad_line"] += 1
                    continue

                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])
                except:
                    stats["yolo_skip_parse_err"] += 1
                    continue

                if cls not in cls_to_dlidx:
                    stats["yolo_skip_unknown_cls"] += 1
                    continue

                if w <= 0 or h <= 0:
                    stats["yolo_skip_bad_bbox"] += 1
                    continue

                # normalized 범위(관대한 체크)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    stats["yolo_skip_oob_bbox"] += 1
                    continue

                bw = w * W
                bh = h * H
                bx = x * W - bw / 2
                by = y * H - bh / 2

                # pixel bbox 최소 검증
                if bw <= 1 or bh <= 1:
                    stats["yolo_skip_tiny_bbox"] += 1
                    continue

                dl = cls_to_dlidx[cls]

                annotations.append({
                    "id": None,  # 나중에 재부여
                    "image_id": iid,
                    "category_id": int(dl),          # ✅ dl_idx
                    "bbox": [float(bx), float(by), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                dlidx_set.add(int(dl))
                ok_any = True
                stats["yolo_ann_ok"] += 1
    except:
        stats["yolo_skip_open_err"] += 1
        continue

    if ok_any:
        yolo_covered.add(img_bn)
        stats["yolo_img_ok"] += 1
    else:
        # yolo txt는 있는데 유효 bbox가 하나도 없으면 "커버"로 치지 않음
        stats["yolo_img_no_valid_bbox"] += 1


# =========================
# 5) JSON -> COCO (YOLO 없는 이미지만)
# =========================
print("[4/5] Parse JSON (only images not covered by YOLO)...")

json_paths = []
for r in JSON_ROOTS:
    if not os.path.exists(r):
        continue
    json_paths += glob(os.path.join(r, "**", "*.json"), recursive=True)

print(f"  - JSON files found: {len(json_paths)}")

for jp in tqdm(json_paths, desc="JSON"):
    # 너무 흔한 mapping 파일 등은 제외
    if os.path.basename(jp) == "category_id_mapping.json":
        continue

    try:
        data = json.load(open(jp, "r", encoding="utf-8"))
    except:
        stats["json_skip_load_err"] += 1
        continue

    imgs = data.get("images")
    anns = data.get("annotations")
    if not isinstance(imgs, list) or not isinstance(anns, list):
        stats["json_skip_not_coco"] += 1
        continue
    if len(imgs) == 0 or len(anns) == 0:
        stats["json_skip_empty"] += 1
        continue

    # local image id -> global image id
    local_map = {}

    # 1) images 등록 (YOLO 커버된 이미지는 아예 스킵)
    for im in imgs:
        fn = im.get("file_name", "")
        bn = os.path.basename(fn) if fn else ""
        if not bn:
            continue

        if bn in yolo_covered:
            stats["json_skip_img_because_yolo"] += 1
            continue

        if bn not in image_index:
            stats["json_skip_img_not_found"] += 1
            continue

        gid = ensure_image(bn)
        if gid is None:
            stats["json_skip_img_bad_image"] += 1
            continue

        local_id = im.get("id")
        if local_id is not None:
            local_map[local_id] = gid

    if not local_map:
        # 이 JSON은 결과적으로 사용할 이미지가 없음
        continue

    # 2) annotations 추가
    for an in anns:
        try:
            lid = an.get("image_id")
            if lid not in local_map:
                continue

            bbox = an.get("bbox")
            if (not isinstance(bbox, list)) or len(bbox) != 4:
                stats["json_skip_bad_bbox"] += 1
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                stats["json_skip_bad_bbox"] += 1
                continue

            dl = int(an.get("category_id"))
        except:
            stats["json_skip_parse_err"] += 1
            continue

        # JSON의 dl_idx가 mapping에 없을 수도 있음. (원하면 여기서 거를 수도 있는데, 지금은 "쓸 수 있으면 다"라서 통과)
        # 다만 나중 STEP4에서 cls로 못 바꾸면 제외될 수 있음.
        annotations.append({
            "id": None,  # 나중에 재부여
            "image_id": local_map[lid],
            "category_id": dl,                 # ✅ dl_idx 유지
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": int(an.get("iscrowd", 0)) if an.get("iscrowd") is not None else 0,
        })
        dlidx_set.add(int(dl))
        stats["json_ann_ok"] += 1


# =========================
# finalize: ann id 재부여
# =========================
print("[5/5] Finalizing & saving...")

# 이미지 리스트
image_list = list(images.values())

# annotation id 재부여
for i, an in enumerate(annotations, start=1):
    an["id"] = i

out = {
    "info": {"description": "Unified COCO FULL COVER (YOLO priority, FAST index)"},
    "licenses": [],
    "images": image_list,
    "annotations": annotations,
    "categories": [{"id": d, "name": str(d)} for d in sorted(dlidx_set)],
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"[DONE] saved -> {OUT_JSON}")
print(f"  images: {len(image_list)}")
print(f"  anns  : {len(annotations)}")
print(f"  cats  : {len(dlidx_set)}")

# 간단 통계
print("\n[STATS]")
for k in sorted(stats.keys()):
    print(f"  {k}: {stats[k]}")